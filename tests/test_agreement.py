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
        [path], FixedClassifier("a", ProcessingRoute.DC1), FixedClassifier("b", ProcessingRoute.DC1)
    )

    assert report["totals"]["agreements"] == report["totals"]["pages"]
    assert report["totals"]["rate"] == 1.0
    assert report["documents"][0]["disagreements"] == []


def test_a_disagreement_is_reported_with_both_decisions(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.DC1), FixedClassifier("b", ProcessingRoute.DC6)
    )

    entry = report["documents"][0]["disagreements"][0]
    assert entry["first"]["route"] == "DC1"
    assert entry["second"]["route"] == "DC6"
    assert entry["page"] == 1


def test_the_pair_histogram_names_the_direction_of_change(tmp_path):
    # The histogram is the useful part of the report: it shows the character of
    # the disagreement at a glance, which a bare agreement percentage cannot.
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.UNKNOWN), FixedClassifier("b", ProcessingRoute.DC6)
    )

    assert report["totals"]["by_route_pair"] == {"unknown->DC6": 1}


def test_the_pair_histogram_counts_every_disagreement(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.DC1), FixedClassifier("b", ProcessingRoute.DC6)
    )

    total = sum(report["totals"]["by_route_pair"].values())
    assert total == report["totals"]["pages"] - report["totals"]["agreements"]


def test_a_disagreement_carries_the_signals_that_produced_it(tmp_path):
    path = one_page_pdf(tmp_path, text=None)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.DC1), FixedClassifier("b", ProcessingRoute.DC6)
    )

    entry = report["documents"][0]["disagreements"][0]
    assert entry["signals"]["text_chars"] == 0


def test_the_report_names_both_classifiers(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("threshold-x", ProcessingRoute.DC1), FixedClassifier("jev-x", ProcessingRoute.DC1)
    )

    assert report["first_classifier"] == "threshold-x"
    assert report["second_classifier"] == "jev-x"


def test_format_report_warns_that_the_number_is_not_accuracy(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.DC1), FixedClassifier("b", ProcessingRoute.DC1)
    )

    text = format_report(report)

    assert "not" in text.lower()
    assert "accuracy" in text.lower()


def test_format_report_shows_the_pair_histogram(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("a", ProcessingRoute.DC1), FixedClassifier("b", ProcessingRoute.DC2)
    )

    assert "DC1->DC2" in format_report(report)


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
        [path], FixedClassifier("a", ProcessingRoute.DC1), FixedClassifier("b", ProcessingRoute.DC2)
    )

    assert len(calls) == 1

class TieredClassifier(FixedClassifier):
    """A fixed-route classifier that also fixes the quality tier."""

    def __init__(self, label, route, tier):
        super().__init__(label, route)
        self._tier = tier

    def classify(self, signals, context=None):
        result = super().classify(signals, context)
        result.quality_tier = self._tier
        return result


class SourcedClassifier(FixedClassifier):
    """A classifier that claims a given decision source, as Jev does."""

    def __init__(self, label, route, source):
        super().__init__(label, route)
        self._source = source

    def classify(self, signals, context=None):
        result = super().classify(signals, context)
        result.signals["jev_route_source"] = self._source
        return result


def test_a_tier_only_disagreement_is_still_a_disagreement(tmp_path):
    # Agreement is computed on route *and* quality tier, so two classifiers
    # returning the same route with different tiers disagree.
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path],
        TieredClassifier("a", ProcessingRoute.DC6, "clean"),
        TieredClassifier("b", ProcessingRoute.DC6, "degraded"),
    )

    assert report["totals"]["agreements"] == 0
    assert report["totals"]["tier_only_disagreements"] == 1


def test_a_tier_only_disagreement_prints_both_tiers(tmp_path):
    # Printing only the route makes a tier-only disagreement look like
    # agreement, because both sides read as the same value.
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path],
        TieredClassifier("a", ProcessingRoute.DC6, "clean"),
        TieredClassifier("b", ProcessingRoute.DC6, "degraded"),
    )

    text = format_report(report)
    assert "clean" in text
    assert "degraded" in text


def test_each_disagreement_line_names_which_classifier_is_which(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("threshold-x", ProcessingRoute.DC6), FixedClassifier("jev-x", ProcessingRoute.DC2)
    )

    line = [
        row
        for row in format_report(report).splitlines()
        if "->" in row and "threshold-x" in row
    ][0]

    assert "threshold-x" in line
    assert "jev-x" in line


def test_the_histogram_heading_names_the_direction(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path], FixedClassifier("threshold-x", ProcessingRoute.DC6), FixedClassifier("jev-x", ProcessingRoute.DC2)
    )

    heading = [
        row
        for row in format_report(report).splitlines()
        if "route pair" in row
    ][0]

    assert "threshold-x" in heading
    assert "jev-x" in heading


def test_the_report_counts_pages_the_second_classifier_decided(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path],
        FixedClassifier("a", ProcessingRoute.DC1),
        SourcedClassifier("jev-x", ProcessingRoute.DC1, "jev"),
    )

    assert report["totals"]["decided_by_source"] == {"jev": 1}


def test_the_report_counts_pages_that_fell_back_even_when_agreeing(tmp_path):
    # A run where every call fell back is indistinguishable from a perfect
    # score unless the fallbacks are counted across all pages, not just the
    # disagreeing ones.
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path],
        FixedClassifier("a", ProcessingRoute.DC1),
        SourcedClassifier("jev-x", ProcessingRoute.DC1, "threshold_fallback"),
    )

    assert report["totals"]["agreements"] == 1
    assert report["totals"]["decided_by_source"] == {"threshold_fallback": 1}


def test_the_source_line_says_how_many_pages_jev_actually_decided(tmp_path):
    path = one_page_pdf(tmp_path)
    report = compare_classifiers(
        [path],
        FixedClassifier("a", ProcessingRoute.DC1),
        SourcedClassifier("jev-x", ProcessingRoute.DC1, "threshold_fallback"),
    )

    text = format_report(report)
    assert "fell back" in text.lower()
