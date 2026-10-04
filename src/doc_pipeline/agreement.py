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
from dotenv import load_dotenv

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
    source_counter: Counter = Counter()
    tier_only = 0
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
                    # Counted across every page, not just disagreeing ones, so
                    # that a run in which every call fell back to the
                    # thresholds is visible instead of reading as a perfect
                    # score.
                    source = second_result.signals.get("jev_route_source")
                    if source:
                        source_counter[source] += 1

                    same_route = first_result.route == second_result.route
                    same_tier = first_result.quality_tier == second_result.quality_tier
                    if same_route and same_tier:
                        agreements += 1
                        continue

                    if same_route:
                        tier_only += 1

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
            "tier_only_disagreements": tier_only,
            "decided_by_source": dict(source_counter),
        },
    }


def format_report(report: dict) -> str:
    """Render a report for a terminal, leading with what the number is not.

    Every printed value is labelled with the classifier it came from and
    whether it is a route or a quality tier. Agreement is computed on the pair
    of them, so an unlabelled comparison is ambiguous in two ways at once.
    """
    totals = report["totals"]
    first = report["first_classifier"]
    second = report["second_classifier"]

    lines = [
        f"Comparing {first} against {second}   (left -> right in every line below)",
        "",
        "This is divergence, not accuracy: two classifiers can agree on every",
        "page and both be wrong. Read the disagreements; the rate is context.",
        "",
        f"Pages: {totals['pages']}  agreeing: {totals['agreements']}  "
        f"rate: {totals['rate']:.1%}",
    ]

    sources = totals.get("decided_by_source") or {}
    if sources:
        decided = sources.get("jev", 0)
        fell_back = sources.get("threshold_fallback", 0)
        if decided or fell_back:
            lines.append(
                f"{second} decided {decided} page(s); fell back to thresholds "
                f"on {fell_back}."
            )
            if fell_back:
                lines.append(
                    "  A page that fell back is NOT a page the two classifiers "
                    "independently agreed on."
                )

    if totals.get("tier_only_disagreements"):
        lines.append(
            f"Tier-only disagreements (same route, different quality): "
            f"{totals['tier_only_disagreements']}"
        )

    lines.append("")

    if totals["by_route_pair"]:
        lines.append(f"Disagreements by route pair ({first} -> {second}):")
        for pair, count in totals["by_route_pair"].items():
            lines.append(f"  {pair:<20} {count}")
        lines.append("")

    for entry in report["documents"]:
        lines.append(
            f"{entry['pdf']}: {entry['agreements']}/{entry['pages']} pages agree"
        )
        for disagreement in entry["disagreements"]:
            left = disagreement["first"]
            right = disagreement["second"]
            lines.append(
                f"  page {disagreement['page']:>3}: "
                f"{first} {left['route']}/{left['quality_tier']} "
                f"({left['confidence']})  ->  "
                f"{second} {right['route']}/{right['quality_tier']} "
                f"({right['confidence']})"
            )

    return "\n".join(lines)


def main() -> None:
    """Entry point for the `doc-pipeline-agreement` console script."""

    # Loaded here rather than inside the library so that importing this module
    # has no side effect on the environment. Without it a `.env` in the
    # project cannot silently switch a test run onto a classifier that spends
    # money.
    load_dotenv()

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