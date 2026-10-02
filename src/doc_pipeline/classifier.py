"""
Page-level document classifier.

Implements the decision tree: cheap, deterministic signals first
falling back to OCR-confidence-based quality checks only
where the cheap signals are ambiguous.

Route labels match schema.ProcessingRoute:
    E1              text-heavy / semi-structured prose
    E2              tabular
    E3              scanned / flattened image, layout-aware OCR
    E4              native CAD/DXF (not reachable from PDF pages; see note)
    E5              low quality / handwritten
    E6              engineering diagram (P&ID, floor plan, sensor layout)
    unknown         page carries no extractable content of its own
"""

from __future__ import annotations
from dataclasses import dataclass
import pymupdf as fitz

from .schema import ProcessingRoute

# ---- Tunable thresholds (Tier 0/1 signals) ---------------------------------
# These were sanity-checked against the three sample PDFs in this workspace;
# treat them as a starting point; the accompanying script prints the raw
# signals for every page so the framework can be re-tuned once you build the
# 200-300 page labelled sample the design doc calls for.

MIN_TEXT_CHARS_FOR_PROSE = 400  # below this, page is probably not text-first
LARGE_PAGE_AREA_PT2 = 1_500_000  # ~A2 and larger (drawings are usually printed big)
HIGH_VECTOR_PATH_COUNT = 5_000  # CAD/engineering line-work is vector-dense
TABLE_VECTOR_PATH_COUNT = 100  # ordinary ruled tables still show up as paths
LOW_OCR_CONFIDENCE = 60  # tesseract mean confidence, 0-100
TEXT_LAYER_COVERAGE_RATIO = (
    0.02  # chars per page-area unit, below = "no usable text layer"
)

# A standard-size (A4/A3) engineering schematic -- electrical, P&ID, loop
# diagrams -- does NOT trip LARGE_PAGE_AREA_PT2, yet is still a diagram, not
# prose or a data table. Its title block (drawing no., sheet/rev, previous/
# next sheet navigation) is close to a universal marker across drafting
# standards (ISO 5457, IEC 61082) regardless of discipline or page size.
# This is a cheap, document-type-agnostic proxy for "this page is a formal
# engineering drawing sheet" -- re-tune the keyword list against your own
# title-block conventions once you have more sample drawings.
TITLE_BLOCK_KEYWORDS = ["Drawing No.", "Sheet:", "Rev.", "Previous:", "Next:"]
TITLE_BLOCK_MIN_HITS = 3
SCHEMATIC_MIN_VECTOR_PATHS = (
    150  # low bar -- title-block hit already carries most of the signal
)

# A "table" whose bbox covers most of the page is almost always the sheet's
# own outer border + zone-reference grid (e.g. the numbered columns 1-12
# along the top of an IEC-style drawing), not a real data table. Cap how
# much of the page a counted table may occupy, alongside the row-count
# floor that already filters out single-row footer boxes.
MAX_TABLE_AREA_RATIO = 0.6


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


@dataclass
class ClassificationResult:
    route: ProcessingRoute
    quality_tier: str
    confidence: float
    reason: str
    signals: dict


def collect_signals(page: "fitz.Page", pdfplumber_page=None) -> PageSignals:
    rect = page.rect
    text = page.get_text()
    images = page.get_images(full=True)
    drawings = page.get_drawings()
    fonts = page.get_fonts()

    # A single-row bordered box (a footer metadata strip, a title-block cell)
    # is technically a "table" to a ruling-line detector but is not the
    # tabular content this route exists for. Require >=3 rows AND that the
    # table covers a non-trivial share of the page before it counts --
    # otherwise a prose page with a footer box gets misrouted wholesale to
    # E2. This is a page-level proxy; true region-level splitting (routing
    # the footer strip as E2 and the body as E1 on the same page) is the
    # next refinement once this runs against the labelled sample.
    has_table = False
    if pdfplumber_page is not None:
        try:
            page_area = rect.width * rect.height
            for t in pdfplumber_page.find_tables():
                x0, top, x1, bottom = t.bbox
                table_area = max(0, x1 - x0) * max(0, bottom - top)
                ratio = table_area / max(page_area, 1)
                if len(t.rows) >= 3 and 0.05 <= ratio <= MAX_TABLE_AREA_RATIO:
                    has_table = True
                    break
        except Exception:
            has_table = False

    title_block_hits = sum(1 for kw in TITLE_BLOCK_KEYWORDS if kw in text)

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
    )


def classify_page(sig: PageSignals) -> ClassificationResult:
    """
    Decision tree, evaluated top to bottom. Each branch returns immediately.
    """
    signals_dict = sig.__dict__

    # --- Tier 0: native / oversized engineering drawing --------------------
    # Large sheet size + very dense vector line-work is the signature of a
    # CAD-exported drawing (P&ID, floor plan, sensor layout) even when it
    # carries a real text layer (title blocks, tags, room labels all have
    # text -- see the BMS sensor layout sample, 7665 chars but 126k vectors).
    if (
        sig.area_pt2 >= LARGE_PAGE_AREA_PT2
        and sig.vector_path_count >= HIGH_VECTOR_PATH_COUNT
    ):
        return ClassificationResult(
            route=ProcessingRoute.E6,
            quality_tier="clean",
            confidence=0.9,
            reason=f"large sheet ({sig.width_pt:.0f}x{sig.height_pt:.0f}pt) with "
            f"{sig.vector_path_count} vector paths -> CAD-exported engineering diagram",
            signals=signals_dict,
        )

    # --- Tier 0b: standard-size engineering schematic ----------------------
    # Same conclusion as the large-sheet branch above, reached via a
    # size-independent signal: a recognizable title block plus non-trivial
    # vector line-work. Needed because electrical/loop/wiring schematics are
    # routinely printed at A4/A3, not the oversized sheets a floor plan or
    # P&ID typically ships at -- page size alone under-detects this class.
    if (
        sig.title_block_hits >= TITLE_BLOCK_MIN_HITS
        and sig.vector_path_count >= SCHEMATIC_MIN_VECTOR_PATHS
    ):
        return ClassificationResult(
            route=ProcessingRoute.E6,
            quality_tier="clean",
            confidence=0.85,
            reason=f"title block detected ({sig.title_block_hits} markers) with "
            f"{sig.vector_path_count} vector paths -> engineering schematic at normal sheet size",
            signals=signals_dict,
        )

    # --- Tier 1: no usable text layer -> image branch -----------------------
    coverage = sig.text_chars / max(sig.area_pt2, 1)
    if (
        sig.text_chars < MIN_TEXT_CHARS_FOR_PROSE
        and coverage < TEXT_LAYER_COVERAGE_RATIO
    ):
        # Could still be a diagram at ordinary page size, or a scan.
        if sig.vector_path_count >= HIGH_VECTOR_PATH_COUNT:
            return ClassificationResult(
                route=ProcessingRoute.E6,
                quality_tier="clean",
                confidence=0.7,
                reason="vector-dense page with negligible text -> diagram at normal page size",
                signals=signals_dict,
            )
        if sig.image_count > 0:
            return ClassificationResult(
                route=ProcessingRoute.E3,
                quality_tier="unknown",  # resolved by OCR-confidence pass, see ocr_quality_check
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

    # --- Tier 1: has text -> table vs. prose --------------------------------
    if sig.has_table_via_pdfplumber or (
        TABLE_VECTOR_PATH_COUNT <= sig.vector_path_count < HIGH_VECTOR_PATH_COUNT
        and sig.text_chars >= MIN_TEXT_CHARS_FOR_PROSE
    ):
        return ClassificationResult(
            route=ProcessingRoute.E2,
            quality_tier="clean",
            confidence=0.75,
            reason="ruled table detected alongside a real text layer -> tabular route",
            signals=signals_dict,
        )

    return ClassificationResult(
        route=ProcessingRoute.E1,
        quality_tier="clean",
        confidence=0.8,
        reason="text-dominant page, no table structure, page size normal -> prose/semantic route",
        signals=signals_dict,
    )


def ocr_quality_check(
    mean_confidence: float, handwriting_detected: bool = False
) -> tuple[ProcessingRoute, str]:
    """
    Tier 2 refinement for pages routed to E3: decide E3 vs E5 and set the
    quality tier, based on actual OCR confidence rather than page-level
    proxies. Call this after running tesseract on the rendered page image.
    """
    if handwriting_detected or mean_confidence < LOW_OCR_CONFIDENCE:
        return (
            ProcessingRoute.E5,
            "degraded" if not handwriting_detected else "handwritten",
        )
    return ProcessingRoute.E3, "clean"
