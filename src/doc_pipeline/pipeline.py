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

from .classifier import collect_signals, classify_page, ocr_quality_check
from .schema import ProcessingRoute, make_record
from .guessers import DocumentTypeGuesser, DEFAULT_GUESSER

# Extractor modules are named after the route they serve. E5 has no module --
# ocr_quality_check() reclassifies a weak E3 page as E5 and it keeps e3's
# OCR output rather than being re-extracted.
from .extractors import e1, e2, e3, e6


def process_pdf(pdf_path: str, guesser: DocumentTypeGuesser = DEFAULT_GUESSER) -> list[dict]:
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]
    records = []

    doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber_doc:
        first_page_text = doc[0].get_text() if len(doc) else ""
        document_type = guesser.guess(pdf_path, first_page_text)

        for i, page in enumerate(doc):
            plumber_page = plumber_doc.pages[i] if i < len(plumber_doc.pages) else None
            sig = collect_signals(page, plumber_page)
            result = classify_page(sig)

            route = result.route
            quality_tier = result.quality_tier
            extracted = {}
            extraction_confidence = result.confidence

            if route == ProcessingRoute.E1:
                extracted = e1.extract(page.get_text())
                extraction_confidence = 0.85

            elif route == ProcessingRoute.E2:
                extracted = (
                    e2.extract(plumber_page)
                    if plumber_page
                    else {"table_count": 0, "tables": []}
                )
                # Text-heavy pages routed to E2 usually have prose around the
                # table too -- capture it so nothing is lost.
                extracted["surrounding_text"] = page.get_text().strip()
                extraction_confidence = 0.75

            elif route == ProcessingRoute.E3:
                ocr = e3.extract(page)
                final_route, final_tier = ocr_quality_check(ocr["mean_confidence"])
                route, quality_tier = final_route, final_tier
                extracted = ocr
                extraction_confidence = ocr["mean_confidence"] / 100.0

            elif route == ProcessingRoute.E6:
                extracted = e6.extract(page)
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
                signals=result.signals,
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
