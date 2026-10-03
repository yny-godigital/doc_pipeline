"""Routing-decision digest over four corpus PDFs.

The digest holds only what the page classifier decides: the route, the quality
tier, the confidence and the raw signals. Extracted content and page text are
excluded on purpose, so committing the baseline never copies client documents
into the repository.

The four PDFs were picked because together they exercise every route the
current classifier emits, including `unknown`. They live outside the
repository, so every consumer of this module must skip when they are absent.
"""

import json
import os

CORPUS_ROOT = (
    "/home/ccw/.hermes/home-project-scale/workspace/project-scale/Discovery"
    "/Discovery Session"
)

CORPUS_PDFS = [
    f"{CORPUS_ROOT}/AI-004 Technical Proposal Drafting"
    f"/Sample Input (Confidential)/Drawing and Specification"
    f"/Vantage-Vendor Supplier List.pdf",
    f"{CORPUS_ROOT}/AI-008 Populate FAT Test Forms"
    f"/Sample 2 - Electrical Panel FAT/input"
    f"/SP25856-MAS-CAS200-IO LIST REV.B.pdf",
    f"{CORPUS_ROOT}/AI-008 Populate FAT Test Forms"
    f"/Sample 2 - Electrical Panel FAT/input/Singe line diagram.pdf",
    f"{CORPUS_ROOT}/AI-004 Technical Proposal Drafting"
    f"/Sample Input (Confidential)/Specifications"
    f"/KUL23-XX-NV5-XX-SH-U-006001-WS4_BMS_EPMS_CPMS_Control_Panel_Schedule.pdf",
]

SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "classifier_baseline.json"
)


def build_digest(pdf_path: str) -> dict:
    """Route every page of one PDF and return the decision-only digest."""
    # Imported inside the function so that an absent corpus does not stop the
    # module from importing.
    from doc_pipeline.pipeline import process_pdf

    records = json.loads(json.dumps(process_pdf(pdf_path)))

    pages = []
    for record in records:
        classification = record["classification"]
        pages.append(
            {
                "page": record["provenance"]["page"],
                "route": classification["processing_route"],
                "quality_tier": classification["quality_tier"],
                "classifier_confidence": classification["classifier_confidence"],
                "signals": classification["signals"],
            }
        )

    return {"basename": os.path.basename(pdf_path), "pages": pages}


def write_snapshot(path: str) -> None:
    """Write the digest of every corpus PDF to `path` as JSON."""
    missing = [p for p in CORPUS_PDFS if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Corpus PDFs are missing, cannot regenerate the baseline:\n  "
            + "\n  ".join(missing)
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    snapshot = {os.path.basename(p): build_digest(p) for p in CORPUS_PDFS}
    with open(path, "w") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    write_snapshot(SNAPSHOT_PATH)
    print(f"wrote {SNAPSHOT_PATH}")