"""The divergence report: where two page classifiers disagree, and how they disagree.

There is no hand-labelled ground truth for page routes in this repository, so
this measures the distance between two classifiers and nothing more. Two
classifiers can agree and both be wrong.
"""

import pymupdf

from doc_pipeline.agreement import compare_classifiers, format_report
from doc_pipeline.classifier import (
    ClassificationResult,
    PageClassifier,
    collect_signals,
)
from doc_pipeline.schema import ProcessingRoute


def one_page_pdf(tmp_path, text="CONTROL POINT SCHEDULE"):
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 144), text)
    document.save(str(path))
    document.close()
    return str(path)


class FixedClassifier(PageClassifier):
    """A classifier that returns a canned route, so the report can be driven."""

    def __init__(self, label, route):
        self._label = label
        self._route = route

    @property
    def name(self):
        return self._label

    def classify(self, signals, context=None):
        return ClassificationResult(
            route=self._route,
            quality_tier="clean",
            confidence=0.8,
            reason=f"{self._label} fixed route",
            signals={"classifier": self._label},
        )


def test_identical_classifiers_agree_on_every_page(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.E1), FixedClassifier("b", ProcessingRoute.E1)
    )

    assert report["totals"]["agreements"] == report["totals"]["pages"]
    assert report["totals"]["rate"] == 1.0
    assert report["documents"][0]["disagreements"] == []


def test_a_disagreement_is_reported_with_both_decisions(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.E1), FixedClassifier("b", ProcessingRoute.E6)
    )

    entry = report["documents"][0]["disagreements"][0]
    assert entry["first"]["route"] == "E1"
    assert entry["second"]["route"] == "E6"
    assert entry["page"] == 1


def test_the_pair_histogram_names_the_direction_of_change(tmp_path):
    # The histogram is the useful part of the report: it shows the character of
    # the disagreement at a glance, which a bare agreement percentage cannot.
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.UNKNOWN), FixedClassifier("b", ProcessingRoute.E6)
    )

    assert report["totals"]["by_route_pair"] == {"unknown->E6": 1}


def test_the_pair_histogram_counts_every_disagreement(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.E1), FixedClassifier("b", ProcessingRoute.E6)
    )

    total = sum(report["totals"]["by_route_pair"].values())
    assert total == report["totals"]["pages"] - report["totals"]["agreements"]


def test_a_disagreement_carries_the_signals_that_produced_it(tmp_path):
    path = one_page_pdf(tmp_path, text=None)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.E1), FixedClassifier("b", ProcessingRoute.E6)
    )

    entry = report["documents"][0]["disagreements"][0]
    assert entry["signals"]["text_chars"] == 0


def test_the_report_names_both_classifiers(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("threshold-x", ProcessingRoute.E1), FixedClassifier("jev-x", ProcessingRoute.E1)
    )

    assert report["first_classifier"] == "threshold-x"
    assert report["second_classifier"] == "jev-x"


def test_format_report_warns_that_the_number_is_not_accuracy(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.E1), FixedClassifier("b", ProcessingRoute.E1)
    )

    text = format_report(report)

    assert "not" in text.lower()
    assert "accuracy" in text.lower()


def test_format_report_shows_the_pair_histogram(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.E1), FixedClassifier("b", ProcessingRoute.E2)
    )

    assert "E1->E2" in format_report(report)


def test_compare_collects_signals_once_per_page(tmp_path, monkeypatch):
    # Running the whole pipeline twice would double the tesseract cost on every
    # scanned page. Collecting signals once and classifying twice is the same
    # information for half the work.
    path = one_page_pdf(tmp_path)
    calls = []
    original = collect_signals

    def counting_collect(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("doc_pipeline.agreement.collect_signals", counting_collect)
    compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.E1), FixedClassifier("b", ProcessingRoute.E2)
    )

    assert len(calls) == 1