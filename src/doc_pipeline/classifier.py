"""
Page-level document classification.

`PageClassifier` is the interface the pipeline calls. `ThresholdClassifier`
implements it with the hand-tuned decision tree this module shipped with:
cheap, deterministic signals first, falling back to OCR-confidence quality
checks only where those signals are ambiguous.

Classification is two-stage. `classify()` picks a route from cheap signals;
for a page routed to DC3, `refine_after_ocr()` then settles DC3 against DC5
from real OCR output. Implementations needing no second stage inherit a no-op.

Route labels match schema.ProcessingRoute:
    DC1             text-heavy / semi-structured prose
    DC2             tabular
    DC3             scanned / flattened image, layout-aware OCR
    DC4             native CAD/DXF (not reachable from PDF pages; see note)
    DC5             low quality / handwritten
    DC6             engineering diagram (P&ID, floor plan, sensor layout)
    unknown         page carries no extractable content of its own
"""

from __future__ import annotations
import abc
import os
import sys
from dataclasses import asdict, dataclass

import pymupdf as fitz

from .schema import DocumentType, ProcessingRoute


@dataclass(frozen=True)
class Thresholds:
    """Tuning constants for `ThresholdClassifier`.

    The defaults reproduce the hand-tuned values this classifier shipped
    with, so `Thresholds()` is the baseline any alternative implementation is
    measured against. Pass overrides to retune one number without touching
    the decision tree, and give two instances different overrides to run two
    configurations side by side in one process.
    """

    min_text_chars_for_prose: int = 400
    large_page_area_pt2: float = 1_500_000.0
    high_vector_path_count: int = 5_000
    table_vector_path_count: int = 100
    low_ocr_confidence: float = 60.0
    text_layer_coverage_ratio: float = 0.02
    # A title block is close to a universal marker across drafting standards,
    # so these words are a document-type-agnostic proxy for "this page is a
    # formal engineering drawing sheet".
    title_block_keywords: tuple[str, ...] = (
        "Drawing No.",
        "Sheet:",
        "Rev.",
        "Previous:",
        "Next:",
    )
    title_block_min_hits: int = 3
    schematic_min_vector_paths: int = 150
    # A "table" spanning most of the page is the sheet's own border and
    # zone-reference grid rather than data. Caps what share of the page a
    # counted table may occupy.
    max_table_area_ratio: float = 0.6


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class DocumentContext:
    """Document-level facts a per-page decision might want.

    `classify()` accepts this, and the pipeline passes nothing for it:
    nothing reads it yet, and populating an object no implementation consults
    would commit to a shape nobody has tested. It is in the signature so that
    the first implementation needing document context does not have to change
    every caller to get it.
    """

    document_type: DocumentType | None = None
    page_count: int = 0
    page_index: int = 0


class PageClassifier(abc.ABC):
    """Assigns a `ProcessingRoute` to one page.

    Implementations must not raise: a page they cannot decide comes back as
    `ProcessingRoute.UNKNOWN` at low confidence. `process_pdf` guards the
    call as well, so a raising implementation costs one record rather than a
    whole document.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier recorded on every record this classifier emits."""

    @abc.abstractmethod
    def classify(
        self, signals: PageSignals, context: DocumentContext | None = None
    ) -> ClassificationResult:
        """Assign a route to one page from its signals."""

    def refine_after_ocr(
        self,
        signals: PageSignals,
        result: ClassificationResult,
        ocr_result: dict,
    ) -> ClassificationResult:
        """Revise a decision once OCR has run.

        Returns the whole result rather than a route and a tier so that an
        implementation can also revise confidence: a page that looked like a
        0.6-confidence DC3 and then OCRs cleanly is now a much more certain
        DC3. The default does nothing, which is the right answer for an
        implementation that never routes a page to DC3.
        """
        return result


@dataclass
class PageSignals:
    page_number: int
    width_pt: float
    height_pt: float
    area_pt2: float
    text_chars: int
    image_count: int
    vector_path_count: int
    has_table_via_pdfplumber: bool
    embedded_font_count: int
    title_block_hits: int = 0
    # Real page text, truncated. A classifier deciding between a drawing and a
    # table needs the page's words, not just its measurements -- a thin-text
    # floor plan and a thin-text index look identical by measurement alone.
    # This is a feature for the classifier only; it is popped before any
    # result's `signals` dict is built, so client document text never reaches
    # an output record or the committed routing snapshot.
    text_sample: str = ""


@dataclass
class ClassificationResult:
    route: ProcessingRoute
    quality_tier: str
    confidence: float
    reason: str
    signals: dict


def collect_signals(
    page: "fitz.Page",
    plumber_page=None,
    thresholds: Thresholds | None = None,
) -> PageSignals:
    """Measure one page.

    `thresholds` only affects the two signals that are themselves
    threshold-shaped: `title_block_hits` and `has_table_via_pdfplumber`.
    Omitting it yields the `Thresholds` defaults.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    rect = page.rect
    text = page.get_text()
    images = page.get_images(full=True)
    drawings = page.get_drawings()
    fonts = page.get_fonts()

    # A single-row bordered box (a footer metadata strip, a title-block
    # cell) is technically a "table" to a ruling-line detector but is not
    # the tabular content this route exists for. Require >=3 rows AND that
    # the table covers a non-trivial share of the page before it counts --
    # otherwise a prose page with a footer box gets misrouted wholesale to
    # DC2. This is a page-level proxy; true region-level splitting is a
    # later refinement.
    has_table = False
    if plumber_page is not None:
        try:
            page_area = rect.width * rect.height
            for table in plumber_page.find_tables():
                x0, top, x1, bottom = table.bbox
                table_area = max(0, x1 - x0) * max(0, bottom - top)
                ratio = table_area / max(page_area, 1)
                if len(table.rows) >= 3 and 0.05 <= ratio <= t.max_table_area_ratio:
                    has_table = True
                    break
        except Exception:
            has_table = False

    title_block_hits = sum(1 for kw in t.title_block_keywords if kw in text)

    return PageSignals(
        page_number=page.number,
        width_pt=rect.width,
        height_pt=rect.height,
        area_pt2=rect.width * rect.height,
        text_chars=len(text.strip()),
        image_count=len(images),
        vector_path_count=len(drawings),
        has_table_via_pdfplumber=has_table,
        embedded_font_count=len(fonts),
        title_block_hits=title_block_hits,
        text_sample=text.strip()[:2000],
    )


class ThresholdClassifier(PageClassifier):
    """Routes a page by hand-tuned thresholds over its signals.

    This is the classifier the pipeline shipped with, unchanged. It is the
    baseline every alternative implementation is measured against.
    """

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    @property
    def name(self) -> str:
        return "threshold"

    def classify(self, sig, context=None) -> ClassificationResult:
        """Evaluate the decision tree top to bottom; each branch returns."""
        t = self.thresholds
        # asdict rather than sig.__dict__: the pipeline writes a classifier
        # name into this dictionary and must not reach back into the record.
        signals_dict = asdict(sig)
        # `asdict` copies every field, including the page text sample that
        # classifiers read. Records are audit data and must not carry client
        # document content, so drop it here. Every other field is unchanged,
        # which keeps the committed routing snapshot byte-identical.
        signals_dict.pop("text_sample", None)

        # --- Tier 0: native / oversized engineering drawing --------------------
        # Large sheet size plus very dense vector line-work is the signature
        # of a CAD-exported drawing, even when the page carries a real text
        # layer -- title blocks, tags and room labels are all text.
        if sig.area_pt2 >= t.large_page_area_pt2 and (
            sig.vector_path_count >= t.high_vector_path_count
        ):
            return ClassificationResult(
                route=ProcessingRoute.DC6,
                quality_tier="clean",
                confidence=0.9,
                reason=f"large sheet ({sig.width_pt:.0f}x{sig.height_pt:.0f}pt) with "
                f"{sig.vector_path_count} vector paths -> CAD-exported engineering diagram",
                signals=signals_dict,
            )

        # --- Tier 0b: standard-size engineering schematic ----------------------
        # The same conclusion, reached by a size-independent signal. Electrical
        # and loop schematics are routinely printed at A4/A3 rather than at the
        # oversized sheets a floor plan ships at, so page size alone
        # under-detects this class.
        if sig.title_block_hits >= t.title_block_min_hits and (
            sig.vector_path_count >= t.schematic_min_vector_paths
        ):
            return ClassificationResult(
                route=ProcessingRoute.DC6,
                quality_tier="clean",
                confidence=0.85,
                reason=f"title block detected ({sig.title_block_hits} markers) with "
                f"{sig.vector_path_count} vector paths -> engineering schematic at normal sheet size",
                signals=signals_dict,
            )

        # --- Tier 1: no usable text layer --------------------------------------
        coverage = sig.text_chars / max(sig.area_pt2, 1)
        if (
            sig.text_chars < t.min_text_chars_for_prose
            and coverage < t.text_layer_coverage_ratio
        ):
            # Could still be a diagram at ordinary page size, or a scan.
            if sig.vector_path_count >= t.high_vector_path_count:
                return ClassificationResult(
                    route=ProcessingRoute.DC6,
                    quality_tier="clean",
                    confidence=0.7,
                    reason="vector-dense page with negligible text -> diagram at normal page size",
                    signals=signals_dict,
                )
            if sig.image_count > 0:
                return ClassificationResult(
                    route=ProcessingRoute.DC3,
                    # Settled by refine_after_ocr once OCR confidence is known.
                    quality_tier="unknown",
                    confidence=0.6,
                    reason="image-bearing page with no usable text layer -> scanned page, route to OCR",
                    signals=signals_dict,
                )
            return ClassificationResult(
                route=ProcessingRoute.UNKNOWN,
                quality_tier="clean",
                confidence=0.4,
                reason="no text, no images, low vector count -> likely blank or decorative page",
                signals=signals_dict,
            )

        # --- Tier 1: has text -> table versus prose ----------------------------
        if sig.has_table_via_pdfplumber or (
            t.table_vector_path_count
            <= sig.vector_path_count
            < t.high_vector_path_count
            and sig.text_chars >= t.min_text_chars_for_prose
        ):
            return ClassificationResult(
                route=ProcessingRoute.DC2,
                quality_tier="clean",
                confidence=0.75,
                reason="ruled table detected alongside a real text layer -> tabular route",
                signals=signals_dict,
            )

        return ClassificationResult(
            route=ProcessingRoute.DC1,
            quality_tier="clean",
            confidence=0.8,
            reason="text-dominant page, no table structure, page size normal -> prose/semantic route",
            signals=signals_dict,
        )

    def refine_after_ocr(self, sig, result, ocr_result):
        """Second pass for DC3 pages: DC3 or DC5, decided on real OCR."""
        handwriting = bool(ocr_result.get("handwriting_detected", False))
        mean_confidence = float(ocr_result.get("mean_confidence", 0.0))

        if handwriting or mean_confidence < self.thresholds.low_ocr_confidence:
            route = ProcessingRoute.DC5
            tier = "handwritten" if handwriting else "degraded"
        else:
            route = ProcessingRoute.DC3
            tier = "clean"

        return ClassificationResult(
            route=route,
            quality_tier=tier,
            confidence=result.confidence,
            reason=(
                f"OCR mean confidence {mean_confidence:.1f}"
                + (" with handwriting detected" if handwriting else "")
                + f" -> {tier}"
            ),
            signals=result.signals,
        )


#: Environment variable naming the classifier to build.
CLASSIFIER_ENV_VAR = "DOC_PIPELINE_CLASSIFIER"

#: Every name `resolve_classifier` accepts.
KNOWN_CLASSIFIERS = ("threshold", "jev")


def _build_known_classifier(name: str) -> PageClassifier:
    """Build the classifier registered under `name`."""
    if name == "threshold":
        return ThresholdClassifier()
    if name == "jev":
        # Imported here rather than at module scope because jev_classifier
        # imports PageClassifier and PageSignals from this module, and a
        # top-level import in both directions is a circular import.
        from .jev_classifier import JevClassifier

        return JevClassifier()
    raise ValueError(f"no builder registered for classifier {name!r}")


def resolve_classifier() -> PageClassifier:
    """Build the classifier named by `DOC_PIPELINE_CLASSIFIER`.

    Unset gives the threshold classifier. An unrecognised name raises: it is a
    mistake in the caller's shell, and stopping before the first PDF is opened
    beats attributing a whole corpus to a classifier nobody chose. An empty
    string is not a name and raises with the rest.

    A recognised name that cannot be built is a different failure -- a missing
    credential, an unreachable service -- and falls back to the threshold
    classifier with a warning, so one broken dependency does not cost a batch
    run.

    This reads the environment and nothing else. Loading a `.env` file belongs
    to the command-line entry points rather than here, so that library code
    and tests see only what the caller explicitly exported.
    """
    requested = os.environ.get(CLASSIFIER_ENV_VAR)
    if requested is None:
        return ThresholdClassifier()

    if requested not in KNOWN_CLASSIFIERS:
        valid = ", ".join(KNOWN_CLASSIFIERS)
        raise ValueError(
            f"{CLASSIFIER_ENV_VAR}={requested!r} is not a known classifier. "
            f"Valid values: {valid}."
        )

    try:
        return _build_known_classifier(requested)
    except Exception as exc:
        print(
            f"{CLASSIFIER_ENV_VAR}={requested!r} is unavailable ({exc!r}); "
            f"falling back to the threshold classifier",
            file=sys.stderr,
        )
        return ThresholdClassifier()
