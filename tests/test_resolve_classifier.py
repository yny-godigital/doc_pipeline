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