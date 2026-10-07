"""
Page-level document classification.

`PageClassifier` is the interface the pipeline calls. The hand-tuned
decision tree and its signal collection live in `threshold_classifier`.

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
from dataclasses import dataclass

from .schema import DocumentType, ProcessingRoute


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


#: Environment variable naming the classifier to build.
CLASSIFIER_ENV_VAR = "DOC_PIPELINE_CLASSIFIER"

#: Every name `resolve_classifier` accepts.
KNOWN_CLASSIFIERS = ("threshold", "jev")


def _build_known_classifier(name: str) -> PageClassifier:
    """Build the classifier registered under `name`."""
    if name == "threshold":
        from .threshold_classifier import ThresholdClassifier

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
    from .threshold_classifier import ThresholdClassifier

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
