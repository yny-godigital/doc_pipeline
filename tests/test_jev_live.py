"""One real call to the Jev API, to prove the request we build is accepted.

Every other test in this suite injects a fake transport, which cannot catch a
wrong field name or a wrong question type. This one sends a real request.

It is opt-in twice over: it needs both `TYPESAFE_API_KEY` and an explicit
`JEV_LIVE_TESTS=1`. `resolve_classifier` loads the project's `.env` file, so the
credential is normally present in the environment; requiring `JEV_LIVE_TESTS`
as well is what stops an ordinary `pytest` run from spending money against a
billable API.
"""

import os
import shutil

import pytest

from doc_pipeline.classifier import PageSignals
from doc_pipeline.jev_classifier import API_KEY_ENV_VAR, JevClassifier, JevConfig
from doc_pipeline.schema import ProcessingRoute

LIVE_TESTS_ENV_VAR = "JEV_LIVE_TESTS"

pytestmark = pytest.mark.skipif(
    not (os.environ.get(API_KEY_ENV_VAR) and os.environ.get(LIVE_TESTS_ENV_VAR) == "1"),
    reason=(
        f"set {LIVE_TESTS_ENV_VAR}=1 (and provide {API_KEY_ENV_VAR}) to run "
        f"live API tests"
    ),
)


def test_a_tabular_page_is_routed_by_the_real_api():
    signals = PageSignals(
        page_number=0,
        width_pt=792.0,
        height_pt=612.0,
        area_pt2=484704.0,
        text_chars=2759,
        image_count=0,
        vector_path_count=2,
        has_table_via_pdfplumber=True,
        embedded_font_count=2,
        title_block_hits=0,
        text_sample=(
            "BMS_EPMS_CPMS_Control_Panel_Schedule   POINT ADDRESS   "
            "DESCRIPTION   DEVICE TYPE   TRIP SETPOINT   RANGE"
        ),
    )

    result = JevClassifier(config=JevConfig.from_env()).classify(signals)

    # This page is unambiguously tabular: a ruled table, repeating column
    # headers, no images and almost no line-work. Anything other than E2 means
    # the request was malformed or the rubrics are not doing their job.
    assert result.route == ProcessingRoute.E2
    assert result.signals["jev_route_source"] == "jev"
    assert result.signals["jev_model"]
    assert result.signals["jev_input_tokens"] > 0


def test_the_second_stage_is_accepted_by_the_real_api():
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")

    from doc_pipeline.classifier import ClassificationResult

    signals = PageSignals(
        page_number=0,
        width_pt=595.0,
        height_pt=842.0,
        area_pt2=500990.0,
        text_chars=0,
        image_count=1,
        vector_path_count=0,
        has_table_via_pdfplumber=False,
        embedded_font_count=0,
    )
    first_stage = ClassificationResult(
        route=ProcessingRoute.E3,
        quality_tier="unknown",
        confidence=0.6,
        reason="scanned page, route to OCR",
        signals={"text_chars": 0},
    )
    ocr = {
        "text": "BMS PANEL NETWORK DRAWING LEVEL 01 SENSOR LAYOUT SHEET 1 OF 4",
        "word_count": 11,
        "mean_confidence": 93.5,
    }

    result = JevClassifier(config=JevConfig.from_env()).refine_after_ocr(
        signals, first_stage, ocr
    )

    # Clean OCR of a full sentence of machine-printed text should stay E3.
    assert result.route == ProcessingRoute.E3
    assert result.quality_tier == "clean"