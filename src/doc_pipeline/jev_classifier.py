"""Jev-backed page classification.

Jev is TypeSafe AI's decision model. It takes a `state` of text and a map of
typed questions, and returns a typed answer per question with a probability
distribution. Unlike an LLM it returns no prose to parse.

Jev accepts text only: images, audio and video are not supported. That shapes
everything here. A drawing is recognised from its page text plus the
measurements in `PageSignals`, because the model cannot be shown the page.
"""

from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

#: Environment variable holding the bearer token for api.typesafe.ai.
API_KEY_ENV_VAR = "TYPESAFE_API_KEY"

#: Environment variable overriding which Jev alias or pinned version is called.
MODEL_ENV_VAR = "JEV_MODEL"

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

#: HTTP statuses worth trying again. 401 and 422 are deliberately absent: an
#: invalid key or a malformed body fails identically every time, so retrying
#: them only turns one misconfiguration into a slow full-corpus hang.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})

#: The rubric Jev reads for each route option.
#:
#: These cannot be derived from the enum. `schema.py` documents routes with
#: trailing `#` comments, and `ProcessingRoute.E1.__doc__` returns the class
#: docstring -- identical text for all seven members -- so reading a docstring
#: here would hand the model the same paragraph five times. They are written by
#: hand to push against each other, since the hard separations are E6 against
#: E2, E6 against E1, and E1 against `unknown`.
ROUTE_CRITERIA = {
    "E1": (
        "Continuous prose: specification text, narrative notes, or "
        "semi-structured paragraphs. No ruled table structure and no diagram."
    ),
    "E2": (
        "Tabular content: rows and columns of values under a header row -- a "
        "schedule, a register, an IO point list. Detected by ruling lines or a "
        "repeating column structure."
    ),
    "E3": (
        "A scanned or flattened page with no usable native text layer. The "
        "content exists as raster images and must be OCR'd before it can be "
        "read. Includes handwritten pages, whatever their legibility."
    ),
    "E6": (
        "An engineering diagram or drawing: a floor plan, panel or sensor "
        "layout, P&ID, wiring or loop schematic. The content is in vector "
        "line-work and scattered labels, not in sentences."
    ),
    "unknown": (
        "The page carries no extractable content of its own: blank, decorative, "
        "a divider, or a fragment too small to classify. Choosing this is not "
        "a failure -- choose it when the page genuinely has nothing to offer."
    ),
}

#: The rubric for settling a scanned page as readable or as needing a human.
LEGIBILITY_CRITERIA = {
    "E3": (
        "tesseract read the page reliably; the recognized text is coherent "
        "and complete"
    ),
    "E5": (
        "tesseract struggled; the text is garbled, partial, or not machine "
        "printed, so a human must re-read this page"
    ),
}


@dataclass(frozen=True)
class JevConfig:
    """Connection and request-shape settings for the Jev API.

    Frozen so a second configuration can be built for comparison without
    either mutating. Every field has a default; `from_env` is the constructor
    the pipeline uses because the API key has no sensible default.
    """

    endpoint: str = DEFAULT_ENDPOINT
    api_key: str = ""
    model: str = "jev-latest"
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    base_backoff_seconds: float = 1.0
    text_char_cap: int = 2000
    ocr_text_char_cap: int = 1000

    @classmethod
    def from_env(cls) -> "JevConfig":
        """Build a config from the environment.

        Raises `RuntimeError` when the API key is missing or blank. The caller
        is expected to degrade to a local classifier rather than let this
        propagate: one missing credential should not cost a whole batch run.
        """
        api_key = (os.environ.get(API_KEY_ENV_VAR) or "").strip()
        if not api_key:
            raise RuntimeError(
                f"{API_KEY_ENV_VAR} is not set; it holds the bearer token for "
                f"{DEFAULT_ENDPOINT}"
            )
        return cls(
            api_key=api_key,
            model=os.environ.get(MODEL_ENV_VAR) or "jev-latest",
        )


def _post(
    endpoint: str,
    payload: dict,
    headers: dict,
    timeout: float,
) -> dict:
    """POST a JSON payload and return the decoded response body.

    Raises `RuntimeError` for a non-2xx status or a body that is not JSON, so
    callers have one failure type to handle regardless of what went wrong.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Jev API returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach the Jev API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Jev API returned a body that is not JSON: {exc}") from exc