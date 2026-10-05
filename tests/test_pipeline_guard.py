"""A classifier that blows up costs one page, not the whole document."""

import pymupdf as fitz
import pytest

from doc_pipeline.classifier import ClassificationResult, PageClassifier
from doc_pipeline.pipeline import process_pdf
from doc_pipeline.schema import ProcessingRoute


def dc3_result(sig):
    return ClassificationResult(
        route=ProcessingRoute.DC3,
        quality_tier="unknown",
        confidence=0.6,
        reason="test double",
        signals=dict(sig.__dict__),
    )


def dc1_result(sig):
    return ClassificationResult(
        route=ProcessingRoute.DC1,
        quality_tier="clean",
        confidence=0.8,
        reason="test double",
        signals=dict(sig.__dict__),
    )


@pytest.fixture
def two_page_pdf(tmp_path):
    doc = fitz.open()
    for number in range(2):
        doc.new_page().insert_text((72, 72), f"page {number + 1}")
    path = tmp_path / "two_pages.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


class Exploding(PageClassifier):
    @property
    def name(self):
        return "exploding"

    def classify(self, sig, context=None):
        raise RuntimeError("classifier unavailable")


class ExplodingRefiner(PageClassifier):
    @property
    def name(self):
        return "exploding-refiner"

    def classify(self, sig, context=None):
        return dc3_result(sig)

    def refine_after_ocr(self, sig, result, ocr_result):
        raise RuntimeError("refiner unavailable")


class Named(PageClassifier):
    @property
    def name(self):
        return "named-double"

    def classify(self, sig, context=None):
        return dc1_result(sig)


def test_failing_classification_becomes_an_unknown_record(two_page_pdf):
    records = process_pdf(two_page_pdf, classifier=Exploding())

    assert len(records) == 2, "the run must continue past the failing page"
    first = records[0]["classification"]
    assert first["processing_route"] == "unknown"
    assert first["classifier_confidence"] == 0.0
    assert "classifier unavailable" in first["signals"]["classifier_error"]
    assert first["signals"]["classifier"] == "exploding"


def test_failing_classification_warns_on_stderr(two_page_pdf, capsys):
    process_pdf(two_page_pdf, classifier=Exploding())
    captured = capsys.readouterr()
    assert "classifier unavailable" in captured.err
    assert "exploding" in captured.err


def test_failing_refinement_becomes_an_unknown_record(two_page_pdf, monkeypatch):
    # Stub the OCR so the test needs no tesseract binary; what is under test is
    # the guard around refine_after_ocr, not the OCR itself.
    monkeypatch.setattr(
        "doc_pipeline.pipeline.dc3.extract",
        lambda page: {"text": "x", "word_count": 1, "mean_confidence": 91.0},
    )
    records = process_pdf(two_page_pdf, classifier=ExplodingRefiner())

    assert len(records) == 2
    first = records[0]["classification"]
    assert first["processing_route"] == "unknown"
    assert "refiner unavailable" in first["signals"]["classifier_error"]


def test_clean_run_carries_the_classifier_name(two_page_pdf):
    records = process_pdf(two_page_pdf, classifier=Named())
    assert records[0]["classification"]["signals"]["classifier"] == "named-double"