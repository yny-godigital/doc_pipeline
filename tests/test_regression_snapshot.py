"""The page-routing refactor must not change any routing decision.

`classifier_baseline.json` was generated from the classifier as it stood
before `PageClassifier` existed. If this test fails, a branch of the decision
tree moved.
"""

import json
import os
import shutil

import pytest

# pytest prepends the test file's own directory to sys.path, so the helper is
# importable by module name rather than as `tests.snapshot_helper`.
from snapshot_helper import CORPUS_PDFS, SNAPSHOT_PATH, build_digest

pytestmark = [
    # Pages routed to DC3 are OCR'd with tesseract, which is a system binary
    # rather than a Python package and is not present on every machine.
    pytest.mark.skipif(
        shutil.which("tesseract") is None, reason="tesseract binary not installed"
    ),
    pytest.mark.skipif(
        not all(os.path.exists(p) for p in CORPUS_PDFS),
        reason="corpus PDFs are outside this repository",
    ),
]


def _without_classifier_key(digest: dict) -> dict:
    """Drop `signals["classifier"]`, which the pipeline adds during the refactor."""
    stripped = json.loads(json.dumps(digest))
    for page in stripped["pages"]:
        page["signals"].pop("classifier", None)
    return stripped


def test_routing_matches_the_committed_baseline():
    with open(SNAPSHOT_PATH) as handle:
        baseline = json.load(handle)

    for pdf_path in CORPUS_PDFS:
        digest = _without_classifier_key(build_digest(pdf_path))
        expected = baseline[os.path.basename(pdf_path)]
        assert digest == expected, f"routing changed for {os.path.basename(pdf_path)}"