"""`DOC_PIPELINE_CLASSIFIER` decides which classifier the pipeline builds."""

import pytest

from doc_pipeline.classifier import ThresholdClassifier, resolve_classifier


def test_defaults_to_the_threshold_classifier(monkeypatch):
    monkeypatch.delenv("DOC_PIPELINE_CLASSIFIER", raising=False)
    assert isinstance(resolve_classifier(), ThresholdClassifier)


def test_threshold_is_spelled_out(monkeypatch):
    monkeypatch.setenv("DOC_PIPELINE_CLASSIFIER", "threshold")
    assert isinstance(resolve_classifier(), ThresholdClassifier)


@pytest.mark.parametrize("value", ["jevv", "Jev", "JEV", "", " threshold"])
def test_unrecognised_values_stop_the_run(monkeypatch, value):
    # A misspelled name means the caller made a mistake, and continuing would
    # attribute a corpus of output to a classifier nobody chose.
    monkeypatch.setenv("DOC_PIPELINE_CLASSIFIER", value)
    with pytest.raises(ValueError) as excinfo:
        resolve_classifier()
    assert "threshold" in str(excinfo.value)


def test_recognised_name_that_cannot_be_built_falls_back(monkeypatch, capsys):
    # A missing credential or an unreachable service must not cost a batch
    # run, so a name that is valid but unbuildable degrades to the baseline.
    attempts = []

    def flaky(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("no api key configured")
        return ThresholdClassifier()

    monkeypatch.setattr("doc_pipeline.classifier._build_known_classifier", flaky)
    monkeypatch.setenv("DOC_PIPELINE_CLASSIFIER", "threshold")

    assert isinstance(resolve_classifier(), ThresholdClassifier)
    assert "no api key configured" in capsys.readouterr().err

def test_jev_is_a_known_classifier_name(monkeypatch):
    monkeypatch.setenv("DOC_PIPELINE_CLASSIFIER", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("JEV_MODEL", raising=False)

    from doc_pipeline.jev_classifier import JevClassifier

    assert isinstance(resolve_classifier(), JevClassifier)


def test_jev_without_a_key_falls_back_to_the_thresholds(monkeypatch):
    # A missing credential must not cost a whole batch run, so a valid name
    # that cannot be built degrades instead of raising.
    monkeypatch.setenv("DOC_PIPELINE_CLASSIFIER", "jev")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    assert isinstance(resolve_classifier(), ThresholdClassifier)


def test_resolve_classifier_ignores_a_dotenv_file(monkeypatch, tmp_path):
    # `.env` loading belongs to the command-line entry points, not to library
    # code. If `resolve_classifier` read it, a project `.env` selecting `jev`
    # would silently switch every test run onto a classifier that makes
    # billable API calls.
    (tmp_path / ".env").write_text(
        "DOC_PIPELINE_CLASSIFIER=jev\nTYPESAFE_API_KEY=should-not-be-read\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOC_PIPELINE_CLASSIFIER", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    assert isinstance(resolve_classifier(), ThresholdClassifier)


def test_the_valid_names_error_lists_both_classifiers(monkeypatch):
    monkeypatch.setenv("DOC_PIPELINE_CLASSIFIER", "jevv")

    with pytest.raises(ValueError) as excinfo:
        resolve_classifier()

    message = str(excinfo.value)
    assert "threshold" in message
    assert "jev" in message
