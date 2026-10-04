"""Report where two page classifiers disagree.

This measures divergence, not accuracy. There is no hand-labelled ground truth
for page routes in this repository: the committed routing snapshot records what
the threshold classifier *does*, not what is *right*. Two classifiers can agree
on every page of a corpus and both be wrong about all of it.

So the agreement percentage is not the deliverable. The deliverable is the list
of disagreements and the histogram of how they break down, because a human
reading twenty disagreeing pages learns far more than a reader of one number.
The pairing histogram is the part worth glancing at: it shows the *character*
of the disagreement at a glance, such as one classifier being systematically
willing to call a thin-text page a diagram.
"""

from __future__ import annotations
import json
import os
import sys
from collections import Counter
from dataclasses import asdict

import pdfplumber
import pymupdf

from .classifier import PageClassifier, collect_signals


def _decision(result) -> dict:
    """The three fields worth comparing, plus where the decision came from."""
    return {
        "route": result.route.value,
        "quality_tier": result.quality_tier,
        "confidence": round(result.confidence, 3),
        "source": result.signals.get("jev_route_source"),
    }


def _audit_signals(signals) -> dict:
    """The measurements behind a decision, without any page text.

    Page text is excluded for the same reason it is excluded from output
    records: this report gets written to disk, and client document content
    does not belong in a file that gets committed or shared.
    """
    recorded = asdict(signals)
    recorded.pop("text_sample", None)
    return recorded


def compare_classifiers(
    pdf_paths, first: PageClassifier, second: PageClassifier
) -> dict:
    """Route every page of every PDF with both classifiers and diff the results.

    Signals are collected once per page and passed to both classifiers. Running
    the full pipeline twice would instead re-OCR every scanned page, which is
    the most expensive part of a run and tells us nothing the signals do not
    already say.
    """
    documents = []
    pair_counter: Counter = Counter()
    total_pages = 0
    total_agreements = 0

    for pdf_path in pdf_paths:
        name = os.path.basename(pdf_path)
        disagreements = []
        pages = 0
        agreements = 0

        document = pymupdf.open(pdf_path)
        try:
            with pdfplumber.open(pdf_path) as plumber_document:
                for index, page in enumerate(document):
                    plumber_page = (
                        plumber_document.pages[index]
                        if index < len(plumber_document.pages)
                        else None
                    )
                    signals = collect_signals(page, plumber_page)
                    first_result = first.classify(signals)
                    second_result = second.classify(signals)

                    pages += 1
                    if (
                        first_result.route == second_result.route
                        and first_result.quality_tier == second_result.quality_tier
                    ):
                        agreements += 1
                        continue

                    disagreements.append(
                        {
                            "page": index + 1,
                            "first": _decision(first_result),
                            "second": _decision(second_result),
                            "signals": _audit_signals(signals),
                        }
                    )
                    pair_counter[
                        f"{first_result.route.value}->{second_result.route.value}"
                    ] += 1
        finally:
            document.close()

        documents.append(
            {
                "pdf": name,
                "pages": pages,
                "agreements": agreements,
                "rate": round(agreements / pages, 3) if pages else 0.0,
                "disagreements": disagreements,
            }
        )
        total_pages += pages
        total_agreements += agreements

    return {
        "first_classifier": first.name,
        "second_classifier": second.name,
        "documents": documents,
        "totals": {
            "pages": total_pages,
            "agreements": total_agreements,
            "rate": round(total_agreements / total_pages, 3) if total_pages else 0.0,
            "by_route_pair": dict(pair_counter.most_common()),
        },
    }


def format_report(report: dict) -> str:
    """Render a report for a terminal, leading with what the number is not."""
    totals = report["totals"]
    lines = [
        f"Comparing {report['first_classifier']} against {report['second_classifier']}",
        "",
        "This is divergence, not accuracy: two classifiers can agree on every",
        "page and both be wrong. Read the disagreements; the rate is context.",
        "",
        f"Pages: {totals['pages']}  agreeing: {totals['agreements']}  "
        f"rate: {totals['rate']:.1%}",
        "",
    ]

    if totals["by_route_pair"]:
        lines.append("Disagreements by route pair:")
        for pair, count in totals["by_route_pair"].items():
            lines.append(f"  {pair:<20} {count}")
        lines.append("")

    for entry in report["documents"]:
        lines.append(
            f"{entry['pdf']}: {entry['agreements']}/{entry['pages']} pages agree"
        )
        for disagreement in entry["disagreements"]:
            first = disagreement["first"]
            second = disagreement["second"]
            lines.append(
                f"  page {disagreement['page']:>3}: "
                f"{first['route']} ({first['confidence']}) -> "
                f"{second['route']} ({second['confidence']})"
            )

    return "\n".join(lines)


def main() -> None:
    """Entry point for the `doc-pipeline-agreement` console script."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: doc-pipeline-agreement <pdf> [<pdf> ...]")
        sys.exit(1)

    from .classifier import ThresholdClassifier, resolve_classifier

    report = compare_classifiers(
        sys.argv[1:], ThresholdClassifier(), resolve_classifier()
    )
    print(format_report(report))
    with open("agreement-report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print("\n  -> wrote agreement-report.json")


if __name__ == "__main__":
    main()