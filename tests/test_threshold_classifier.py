"""Branch coverage for the routing decision tree.

These tests use synthetic signals rather than PDFs: the tree is pure
arithmetic over a `PageSignals` record, so a real document adds runtime and
adds nothing. `test_regression_snapshot.py` covers the end-to-end path
against real documents.
"""

import pytest

from doc_pipeline.classifier import (
    ClassificationResult,
    PageClassifier,
)
from doc_pipeline.threshold_classifier import Thresholds, ThresholdClassifier
from doc_pipeline.schema import ProcessingRoute


def signals(**overrides):
    """A prose page with a usable text layer, then overridden field by field.

    The default page is 200x200pt carrying 1200 characters, a text density of
    0.03 characters per square point -- above the 0.02 floor below which a page
    is treated as having no usable text layer at all. Tests that need the
    textless branch lower `text_chars` and the tree handles the rest.
    """
    from doc_pipeline.classifier import PageSignals

    base = dict(
        page_number=0,
        width_pt=200.0,
        height_pt=200.0,
        area_pt2=200.0 * 200.0,
        text_chars=1200,
        image_count=0,
        vector_path_count=10,
        has_table_via_pdfplumber=False,
        embedded_font_count=3,
    )
    base.update(overrides)
    return PageSignals(**base)


@pytest.fixture
def classifier():
    return ThresholdClassifier()


def test_large_dense_sheet_is_an_engineering_diagram(classifier):
    result = classifier.classify(signals(area_pt2=1_600_000, vector_path_count=5_000))
    assert result.route == ProcessingRoute.DC6
    assert result.confidence == 0.9


def test_title_block_with_linework_is_an_engineering_schematic(classifier):
    # An A4 electrical schematic: ordinary page size, but a title block and
    # enough vector work to rule out prose.
    result = classifier.classify(signals(title_block_hits=3, vector_path_count=150))
    assert result.route == ProcessingRoute.DC6
    assert result.confidence == 0.85


def test_vector_dense_page_without_text_is_a_diagram(classifier):
    result = classifier.classify(signals(text_chars=100, vector_path_count=5_000))
    assert result.route == ProcessingRoute.DC6
    assert result.confidence == 0.7


def test_image_page_without_text_layer_is_routed_to_ocr(classifier):
    result = classifier.classify(signals(text_chars=100, image_count=1))
    assert result.route == ProcessingRoute.DC3
    # The tier is a placeholder here: refine_after_ocr settles it once OCR
    # confidence is known.
    assert result.quality_tier == "unknown"


def test_empty_page_is_unknown(classifier):
    result = classifier.classify(signals(text_chars=0, image_count=0))
    assert result.route == ProcessingRoute.UNKNOWN
    assert result.confidence == 0.4


def test_detected_table_is_tabular(classifier):
    result = classifier.classify(signals(has_table_via_pdfplumber=True))
    assert result.route == ProcessingRoute.DC2
    assert result.confidence == 0.75


def test_ruled_lines_around_prose_are_tabular(classifier):
    # No ruling-line table detected, but the vector count sits in the band
    # ordinary ruled tables occupy.
    result = classifier.classify(signals(vector_path_count=150))
    assert result.route == ProcessingRoute.DC2


def test_text_dominant_page_is_prose(classifier):
    result = classifier.classify(signals())
    assert result.route == ProcessingRoute.DC1
    assert result.confidence == 0.8


def test_result_carries_a_copy_of_the_signals_that_decided_it(classifier):
    page = signals(text_chars=0)
    result = classifier.classify(page)
    assert result.signals["text_chars"] == 0
    # A copy, not the PageSignals instance's own __dict__: the pipeline writes
    # a classifier name into this dictionary and must not mutate the record.
    assert result.signals is not page.__dict__


class TestRefineAfterOcr:
    def _dc3_result(self, classifier):
        return classifier.classify(signals(text_chars=100, image_count=1))

    def test_low_confidence_ocr_becomes_degraded(self, classifier):
        result = self._dc3_result(classifier)
        refined = classifier.refine_after_ocr(
            signals(), result, {"mean_confidence": 40.0}
        )
        assert (refined.route, refined.quality_tier) == (ProcessingRoute.DC5, "degraded")

    def test_handwriting_is_its_own_tier(self, classifier):
        result = self._dc3_result(classifier)
        refined = classifier.refine_after_ocr(
            signals(), result, {"mean_confidence": 95.0, "handwriting_detected": True}
        )
        assert (refined.route, refined.quality_tier) == (
            ProcessingRoute.DC5,
            "handwritten",
        )

    def test_good_ocr_stays_clean(self, classifier):
        result = self._dc3_result(classifier)
        refined = classifier.refine_after_ocr(
            signals(), result, {"mean_confidence": 92.0}
        )
        assert (refined.route, refined.quality_tier) == (ProcessingRoute.DC3, "clean")
        # The pre-OCR confidence survives the refinement.
        assert refined.confidence == result.confidence


class TestInterfaceDefaults:
    def test_refine_after_ocr_is_a_no_op_by_default(self):
        class PassThrough(PageClassifier):
            @property
            def name(self):
                return "passthrough"

            def classify(self, sig, context=None):
                return ClassificationResult(
                    route=ProcessingRoute.DC1,
                    quality_tier="clean",
                    confidence=0.5,
                    reason="test double",
                    signals=dict(sig.__dict__),
                )

        page = signals()
        double = PassThrough()
        result = double.classify(page)
        assert double.refine_after_ocr(page, result, {"mean_confidence": 1.0}) is result

    def test_thresholds_can_be_overridden_per_instance(self):
        tuned = ThresholdClassifier(Thresholds(min_text_chars_for_prose=250))
        page = signals(text_chars=300)
        # 300 characters clears the tuned 250 floor but not the default 400 one,
        # so the same page is prose under the tuned thresholds and unknown
        # under the defaults.
        assert tuned.classify(page).route == ProcessingRoute.DC1
        assert ThresholdClassifier().classify(page).route == ProcessingRoute.UNKNOWN