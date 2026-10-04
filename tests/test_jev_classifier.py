"""The Jev page classifier, exercised through an injected transport.

Every test here runs with no network and no API key. The transport is a plain
callable taking (endpoint, payload, headers, timeout) and returning a
pre-canned response body, which is the seam that keeps these tests fast and
free.
"""

import pytest

from doc_pipeline.classifier import PageSignals
from doc_pipeline.jev_classifier import JevClassifier, JevConfig
from doc_pipeline.schema import ProcessingRoute


def make_signals(**overrides) -> PageSignals:
    base = dict(
        page_number=0,
        width_pt=595.0,
        height_pt=842.0,
        area_pt2=500990.0,
        text_chars=2759,
        image_count=0,
        vector_path_count=2,
        has_table_via_pdfplumber=True,
        embedded_font_count=2,
        title_block_hits=0,
        text_sample="POINT ADDRESS DESCRIPTION TRIP SETPOINT",
    )
    base.update(overrides)
    return PageSignals(**base)


def response(choice="E2", confidence=0.81, quality="clean", **extra):
    body = {
        "model": "jev-1.13.0",
        "answers": {
            "route": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": {choice: 0.81, "E1": 0.19},
            },
            "quality": {
                "type": "choice",
                "choice": quality,
                "confidence": 0.7,
                "probabilities": {quality: 0.7},
            },
        },
        "usage": {"input_tokens": 512, "output_tokens": 24},
    }
    body.update(extra)
    return body


def classifier_returning(body, calls=None, fallback=None):
    """Build a JevClassifier whose transport returns a fixed response body."""

    def transport(endpoint, payload, headers, timeout):
        if calls is not None:
            calls.append(payload)
        if isinstance(body, Exception):
            raise body
        return body

    return JevClassifier(
        config=JevConfig(api_key="test-key"),
        transport=transport,
        fallback=fallback,
    )


# --- the request -----------------------------------------------------------


def test_request_offers_exactly_five_route_options():
    calls = []
    classifier_returning(response(), calls=calls).classify(make_signals())

    options = calls[0]["questions"]["route"]["criteria"]
    assert set(options) == {"E1", "E2", "E3", "E6", "unknown"}


def test_request_asks_for_route_and_quality_together():
    # Jev answers every question in one request in parallel, so asking for the
    # quality tier alongside the route costs no extra round trip.
    calls = []
    classifier_returning(response(), calls=calls).classify(make_signals())

    assert set(calls[0]["questions"]) == {"route", "quality"}
    assert calls[0]["questions"]["route"]["type"] == "choice"
    assert calls[0]["questions"]["quality"]["type"] == "choice"


def test_request_carries_the_configured_model():
    calls = []
    config = JevConfig(api_key="test-key", model="jev-1.13.0")
    transport = lambda endpoint, payload, headers, timeout: calls.append(payload) or response()
    JevClassifier(config=config, transport=transport).classify(make_signals())

    assert calls[0]["model"] == "jev-1.13.0"


def test_request_bears_the_api_key_as_a_bearer_token():
    seen = {}

    def transport(endpoint, payload, headers, timeout):
        seen.update(headers)
        return response()

    JevClassifier(config=JevConfig(api_key="secret-key"), transport=transport).classify(
        make_signals()
    )

    assert seen["Authorization"] == "Bearer secret-key"
    assert seen["Content-Type"] == "application/json"


def test_state_contains_the_page_text():
    state = JevClassifier(config=JevConfig(api_key="k"))._build_state(
        make_signals(text_sample="SENSOR LAYOUT LEVEL 01")
    )

    assert "SENSOR LAYOUT LEVEL 01" in state


def test_state_contains_the_measurements():
    state = JevClassifier(config=JevConfig(api_key="k"))._build_state(make_signals())

    assert "ruled table detected" in state
    assert "2759 chars" in state
    assert "vector paths" in state


def test_state_truncates_page_text_to_the_configured_cap():
    classifier = JevClassifier(config=JevConfig(api_key="k", text_char_cap=50))
    state = classifier._build_state(make_signals(text_sample="Y" * 5000))

    assert state.count("Y") == 50


def test_state_survives_a_page_with_no_text():
    # A scanned page has no text layer at all. The measurements must still
    # describe the page, or the model has nothing to work from.
    state = JevClassifier(config=JevConfig(api_key="k"))._build_state(
        make_signals(text_chars=0, text_sample="")
    )

    assert "No native text layer" in state


# --- the answer ------------------------------------------------------------


def test_chosen_option_becomes_the_route():
    result = classifier_returning(response(choice="E2")).classify(make_signals())

    assert result.route == ProcessingRoute.E2


def test_unknown_option_becomes_the_unknown_route():
    result = classifier_returning(response(choice="unknown")).classify(make_signals())

    assert result.route == ProcessingRoute.UNKNOWN


def test_confidence_comes_from_the_model_not_a_constant():
    result = classifier_returning(response(confidence=0.93)).classify(make_signals())

    assert result.confidence == pytest.approx(0.93)


def test_quality_answer_becomes_the_quality_tier():
    result = classifier_returning(response(quality="degraded")).classify(make_signals())

    assert result.quality_tier == "degraded"


def test_result_records_the_model_that_answered():
    # The response reports the version that actually answered, which can differ
    # from the alias that was requested. Recording it is what makes an output
    # file from months ago interpretable.
    result = classifier_returning(response()).classify(make_signals())

    assert result.signals["jev_model"] == "jev-1.13.0"


def test_result_records_token_usage():
    result = classifier_returning(response()).classify(make_signals())

    assert result.signals["jev_input_tokens"] == 512


def test_result_records_that_jev_decided():
    result = classifier_returning(response()).classify(make_signals())

    assert result.signals["jev_route_source"] == "jev"


def test_result_carries_the_page_signals_for_audit():
    result = classifier_returning(response()).classify(make_signals())

    assert result.signals["text_chars"] == 2759


def test_page_text_never_lands_in_the_result_signals():
    result = classifier_returning(response()).classify(
        make_signals(text_sample="CONFIDENTIAL CLIENT CONTENT")
    )

    assert "text_sample" not in result.signals


def test_name_is_jev():
    assert classifier_returning(response()).name == "jev"


# --- failure ---------------------------------------------------------------


def test_server_error_is_retried_then_succeeds():
    attempts = []
    bodies = [RuntimeError("Jev API returned 503"), response(choice="E1")]

    def transport(endpoint, payload, headers, timeout):
        attempts.append(1)
        body = bodies.pop(0)
        if isinstance(body, Exception):
            raise body
        return body

    config = JevConfig(api_key="k", base_backoff_seconds=0.0)
    result = JevClassifier(config=config, transport=transport).classify(make_signals())

    assert len(attempts) == 2
    assert result.route == ProcessingRoute.E1
    assert result.signals["jev_route_source"] == "jev"


def test_persistent_failure_falls_back_after_three_attempts():
    attempts = []

    def transport(endpoint, payload, headers, timeout):
        attempts.append(1)
        raise RuntimeError("Jev API returned 529")

    config = JevConfig(api_key="k", max_attempts=3, base_backoff_seconds=0.0)
    result = JevClassifier(config=config, transport=transport).classify(make_signals())

    assert len(attempts) == 3
    assert result.signals["jev_route_source"] == "threshold_fallback"


def test_unauthorized_is_not_retried():
    # An invalid key fails identically on every attempt. Retrying it turns one
    # misconfiguration into three slow failures on every page of the corpus.
    attempts = []

    def transport(endpoint, payload, headers, timeout):
        attempts.append(1)
        raise RuntimeError("Jev API returned 401")

    config = JevConfig(api_key="k", max_attempts=3, base_backoff_seconds=0.0)
    JevClassifier(config=config, transport=transport).classify(make_signals())

    assert len(attempts) == 1


def test_transport_failure_falls_back_to_the_threshold_decision():
    def transport(endpoint, payload, headers, timeout):
        raise RuntimeError("connection reset")

    config = JevConfig(api_key="k", max_attempts=1, base_backoff_seconds=0.0)
    signals = make_signals(text_chars=2759, has_table_via_pdfplumber=True)
    result = JevClassifier(config=config, transport=transport).classify(signals)

    # The page keeps the decision the threshold tree would have made, rather
    # than becoming a blank `unknown` record.
    assert result.route == ProcessingRoute.E2
    assert result.signals["jev_route_source"] == "threshold_fallback"


def test_failure_records_the_error_for_audit():
    def transport(endpoint, payload, headers, timeout):
        raise RuntimeError("connection reset")

    config = JevConfig(api_key="k", max_attempts=1, base_backoff_seconds=0.0)
    result = JevClassifier(config=config, transport=transport).classify(make_signals())

    assert "connection reset" in result.signals["jev_error"]


def test_missing_answers_key_falls_back():
    result = classifier_returning({"model": "jev-1.13.0"}).classify(make_signals())

    assert result.signals["jev_route_source"] == "threshold_fallback"


def test_missing_route_answer_falls_back():
    body = {"model": "jev-1.13.0", "answers": {"quality": {"choice": "clean"}}}
    result = classifier_returning(body).classify(make_signals())

    assert result.signals["jev_route_source"] == "threshold_fallback"


def test_option_outside_the_vocabulary_falls_back():
    result = classifier_returning(response(choice="E7")).classify(make_signals())

    assert result.signals["jev_route_source"] == "threshold_fallback"


def test_constructing_with_a_blank_api_key_raises():
    # The pipeline builds this classifier with no arguments and lets
    # `JevConfig.from_env` supply the key. A missing credential must surface at
    # construction, where the caller can degrade to a local classifier, rather
    # than on the first page of a corpus.
    with pytest.raises(RuntimeError):
        JevClassifier(config=JevConfig())