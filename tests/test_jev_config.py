"""Configuration, credentials, and the route vocabulary Jev is offered."""

import pytest

from doc_pipeline.jev_classifier import (
    API_KEY_ENV_VAR,
    DEFAULT_ENDPOINT,
    ROUTE_CRITERIA,
    JevConfig,
)


def test_default_endpoint_is_the_documented_systemone_url():
    assert DEFAULT_ENDPOINT == "https://api.typesafe.ai/v1/systemone"


def test_route_vocabulary_is_exactly_five_options():
    # DC5 is absent on purpose: pipeline.py has no DC5 extraction branch, so a
    # first-stage DC5 would be written with an empty `extracted` dict and no OCR
    # output. DC4 is absent because it is unreachable from PDF pages.
    assert set(ROUTE_CRITERIA) == {"DC1", "DC2", "DC3", "DC6", "unknown"}


def test_every_route_option_carries_a_rubric():
    for option, rubric in ROUTE_CRITERIA.items():
        assert rubric.strip(), f"{option} has an empty rubric"


def test_from_env_requires_an_api_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        JevConfig.from_env()

    assert API_KEY_ENV_VAR in str(excinfo.value)


def test_from_env_rejects_a_blank_api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "   ")

    with pytest.raises(RuntimeError):
        JevConfig.from_env()


def test_from_env_defaults_the_model_alias(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    monkeypatch.delenv("JEV_MODEL", raising=False)

    assert JevConfig.from_env().model == "jev-latest"


def test_from_env_reads_the_model_pin(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    monkeypatch.setenv("JEV_MODEL", "jev-1.13.0")

    assert JevConfig.from_env().model == "jev-1.13.0"


def test_config_is_frozen(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    config = JevConfig.from_env()

    with pytest.raises(Exception):
        config.model = "something-else"


def test_text_caps_are_bounded(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    config = JevConfig.from_env()

    assert config.text_char_cap == 2000
    assert config.ocr_text_char_cap == 1000