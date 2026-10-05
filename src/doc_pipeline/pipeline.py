"""
PDF -> classified, extracted -> records out

Usage:
    uv run doc-pipeline <path-to-pdf> [<path-to-pdf> ...]

Writes one JSON file per input PDF to ./output/<stub>.records.json

Every processing route must emit records in the schema shape. This is what makes the cross-document reasoning possible later. The consumer of these records never needs to know which route produced them.

"""

import sys
import os
import json
import pymupdf as fitz
import pdfplumber
from dotenv import load_dotenv

from .classifier import (
    ClassificationResult,
    PageClassifier,
    collect_signals,
    resolve_classifier,
)
from .schema import ProcessingRoute, make_record
from .guessers import DocumentTypeGuesser, DEFAULT_GUESSER

# Extractor modules are named after the route they serve. DC5 has no module --
# ocr_quality_check() reclassifies a weak DC3 page as DC5 and it keeps dc3's
# OCR output rather than being re-extracted.
from .extractors import dc1, dc2, dc3, dc6


def _failed_result(classifier: PageClassifier, exc: Exception) -> ClassificationResult:
    """The result recorded for a page whose classifier raised.

    The page is kept rather than dropped: the record still points at the
    source page, and `classifier_error` is what distinguishes this from a
    page that was genuinely blank.
    """
    return ClassificationResult(
        route=ProcessingRoute.UNKNOWN,
        quality_tier="clean",
        confidence=0.0,
        reason=f"classifier {classifier.name} raised: {exc}",
        signals={"classifier": classifier.name, "classifier_error": repr(exc)},
    )


def _classify_safely(classifier: PageClassifier, sig, page_index: int) -> ClassificationResult:
    """Call `classifier.classify`, degrading one page on an unexpected error.

    The interface says implementations must not raise and should answer
    `UNKNOWN` at low confidence instead; this is the belt to that braces. A
    classifier that raises is a bug in that classifier, not a reason to throw
    away the rest of the document.
    """
    try:
        # No `context` argument: no implementation reads it yet, and passing
        # an empty context would imply one that does.
        return classifier.classify(sig)
    except Exception as exc:
        print(
            f"  page {page_index + 1:>3}: classifier {classifier.name} failed "
            f"({exc!r}); recording an unknown record",
            file=sys.stderr,
        )
        return _failed_result(classifier, exc)


def _refine_safely(
    classifier: PageClassifier, sig, result, ocr_result, page_index: int
) -> ClassificationResult:
    """The same guard around the post-OCR stage."""
    try:
        return classifier.refine_after_ocr(sig, result, ocr_result)
    except Exception as exc:
        print(
            f"  page {page_index + 1:>3}: classifier {classifier.name} failed "
            f"during refinement ({exc!r}); recording an unknown record",
            file=sys.stderr,
        )
        return _failed_result(classifier, exc)


def process_pdf(
    pdf_path: str,
    guesser: DocumentTypeGuesser = DEFAULT_GUESSER,
    classifier: PageClassifier | None = None,
) -> list[dict]:
    """Classify and extract every page of one PDF.

    `classifier` defaults to None and is resolved at call time rather than
    bound at import time, so it need not be constructible when the module
    loads. It is resolved once per document, not once per page: a classifier
    holding a client or a session is built once.
    """
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]
    records = []

    doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber_doc:
        first_page_text = doc[0].get_text() if len(doc) else ""
        document_type = guesser.guess(pdf_path, first_page_text)
        # DocumentContext is deliberately not built: nothing reads it yet.
        classifier = classifier or resolve_classifier()

        for i, page in enumerate(doc):
            plumber_page = plumber_doc.pages[i] if i < len(plumber_doc.pages) else None
            sig = collect_signals(page, plumber_page)
            result = _classify_safely(classifier, sig, i)

            route = result.route
            quality_tier = result.quality_tier
            extracted = {}
            extraction_confidence = result.confidence

            if route == ProcessingRoute.DC1:
                extracted = dc1.extract(page.get_text())
                extraction_confidence = 0.85

            elif route == ProcessingRoute.DC2:
                extracted = (
                    dc2.extract(plumber_page)
                    if plumber_page
                    else {"table_count": 0, "tables": []}
                )
                # Text-heavy pages routed to DC2 usually have prose around the
                # table too -- capture it so nothing is lost.
                extracted["surrounding_text"] = page.get_text().strip()
                extraction_confidence = 0.75

            elif route == ProcessingRoute.DC3:
                ocr = dc3.extract(page)
                # The classifier settles DC3 against DC5 using real OCR output
                # and returns a whole result, so the pipeline takes its route,
                # tier and revised confidence together.
                result = _refine_safely(classifier, sig, result, ocr, i)
                route, quality_tier = result.route, result.quality_tier
                extracted = ocr
                extraction_confidence = ocr["mean_confidence"] / 100.0

            elif route == ProcessingRoute.DC6:
                extracted = dc6.extract(page)
                extraction_confidence = 0.7  # geometry extraction is deterministic;
                # confidence caps here because symbol ID / graph build hasn't run

            elif route == ProcessingRoute.UNKNOWN:
                extracted = {"note": "no extractable content on this page"}
                extraction_confidence = 0.3

            record = make_record(
                document_id=document_id,
                document_type=document_type,
                page=i + 1,
                processing_route=route,
                quality_tier=quality_tier,
                # The classifier name is what makes two experiment runs
                # distinguishable after the fact, so it is recorded even when
                # the classification failed.
                signals={**result.signals, "classifier": classifier.name},
                confidence=result.confidence,
                raw_ref=f"{pdf_path}#page={i + 1}",
                extracted=extracted,
                extraction_confidence=round(extraction_confidence, 2),
            )
            records.append(record.to_dict())

            print(
                f"  page {i + 1:>3}: route={route.value:<15} quality={quality_tier:<10} "
                f"conf={result.confidence:.2f}  ({result.reason})"
            )

    doc.close()
    return records


def main():
    """
    Expects the pdf file path as the first or next arguments to the method
    """

    # Loaded here rather than inside the library so that importing this module
    # has no side effect on the environment, and a `.env` in the project can
    # only ever influence a real command-line run.
    load_dotenv()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for pdf_path in sys.argv[1:]:
        print(f"\n=== {pdf_path} ===")
        records = process_pdf(pdf_path)
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join("output", f"{stem}.records.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  -> wrote {len(records)} record(s) to {out_path}")


if __name__ == "__main__":
    main()
