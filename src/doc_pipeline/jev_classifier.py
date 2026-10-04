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

from .classifier import (
    ClassificationResult,
    PageClassifier,
    PageSignals,
    ThresholdClassifier,
)
from .schema import ProcessingRoute

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

def _is_retryable(error: Exception) -> bool:
    """Whether a failed request is worth sending again.

    The status code is embedded in the message `_post` raises, because
    `RuntimeError` is the single failure type callers handle.
    """
    message = str(error)
    return any(f"returned {code}" in message for code in RETRYABLE_STATUS_CODES)


class JevClassifier(PageClassifier):
    """Routes a page by asking Jev, degrading to thresholds on any failure.

    One request per page carries two questions -- the route and the quality
    tier -- which Jev answers in parallel, so the second costs no extra round
    trip.

    Nothing here raises. A missing key, an unreachable API, an exhausted retry
    budget or a malformed response all degrade that single page to the decision
    `ThresholdClassifier` would have made, because a blank `unknown` record
    loses the page entirely while a threshold decision at least preserves it.
    """

    def __init__(
        self,
        config: JevConfig | None = None,
        *,
        transport=None,
        fallback: ThresholdClassifier | None = None,
    ) -> None:
        """
        `config` defaults to `JevConfig.from_env()`, which raises when
        `TYPESAFE_API_KEY` is unset. A blank key is rejected here too, so a
        misconfigured credential surfaces at construction, where the caller can
        degrade to a local classifier, rather than on the first page of a
        corpus.

        `transport` is the injection point for tests. It takes
        `(endpoint, payload, headers, timeout)` and returns the decoded
        response body.
        """
        self.config = config if config is not None else JevConfig.from_env()
        if not self.config.api_key.strip():
            raise RuntimeError(
                f"{API_KEY_ENV_VAR} is empty; it holds the bearer token for "
                f"{self.config.endpoint}"
            )
        self._transport = transport if transport is not None else _post
        self._fallback = fallback if fallback is not None else ThresholdClassifier()

    @property
    def name(self) -> str:
        return "jev"

    # --- request ------------------------------------------------------------

    def _build_state(self, signals: PageSignals) -> str:
        """Describe one page as text: its measurements, then its own words.

        The measurements come first because a page can be almost entirely
        diagram, in which case the text sample is a handful of stray labels
        and the measurements are the evidence.
        """
        cap = self.config.text_char_cap
        table_note = (
            "ruled table detected" if signals.has_table_via_pdfplumber else "no ruled table"
        )
        if signals.text_sample:
            text_block = f"Text of page:\n{signals.text_sample[:cap]}"
        else:
            text_block = "No native text layer on this page."

        return (
            f"Page {signals.page_number + 1} | "
            f"{signals.width_pt:.0f}x{signals.height_pt:.0f}pt | "
            f"{signals.text_chars} chars native text | "
            f"{signals.image_count} images | "
            f"{signals.vector_path_count} vector paths | "
            f"{table_note} | "
            f"title-block markers: {signals.title_block_hits} | "
            f"embedded fonts: {signals.embedded_font_count}\n\n"
            f"{text_block}"
        )

    def _build_payload(self, signals: PageSignals) -> dict:
        return {
            "state": self._build_state(signals),
            "model": self.config.model,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": (
                        "Which processing route fits this page of a PDF "
                        "document?"
                    ),
                    "criteria": ROUTE_CRITERIA,
                },
                "quality": {
                    "type": "choice",
                    "instructions": "How readable is this page?",
                    "criteria": {
                        "clean": "Fully legible, machine-printed, nothing degraded.",
                        "degraded": (
                            "Legible but poor: faint, skewed, low contrast, or "
                            "partly obscured."
                        ),
                        "handwritten": "Contains handwriting rather than machine-printed text.",
                    },
                },
            },
        }

    def _call(self, payload: dict) -> dict:
        """POST a payload, retrying only the failures worth retrying."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None

        for attempt in range(self.config.max_attempts):
            try:
                return self._transport(
                    self.config.endpoint, payload, headers, self.config.timeout_seconds
                )
            except RuntimeError as exc:
                if not _is_retryable(exc):
                    raise
                last_error = exc
                if attempt < self.config.max_attempts - 1:
                    time.sleep(self.config.base_backoff_seconds * (2**attempt))

        raise last_error if last_error else RuntimeError("Jev call failed")

    # --- answer -------------------------------------------------------------

    def _fallback_result(
        self, signals: PageSignals, error: Exception
    ) -> ClassificationResult:
        """The decision the threshold tree would have made, marked as a fallback."""
        result = self._fallback.classify(signals)
        signals_dict = dict(result.signals)
        signals_dict["classifier"] = self.name
        signals_dict["jev_route_source"] = "threshold_fallback"
        signals_dict["jev_error"] = repr(error)
        return ClassificationResult(
            route=result.route,
            quality_tier=result.quality_tier,
            confidence=result.confidence,
            reason=f"Jev unavailable ({error}); fell back to the threshold classifier",
            signals=signals_dict,
        )

    def _parse(self, body: dict) -> tuple[ProcessingRoute, str, float, str, int]:
        """Pull the route, tier, confidence, model and token count out of a response."""
        answers = (body or {}).get("answers") or {}
        route_answer = answers.get("route")
        if not isinstance(route_answer, dict) or "choice" not in route_answer:
            raise RuntimeError("Jev response has no route answer")

        option = route_answer["choice"]
        if option not in ROUTE_CRITERIA:
            raise RuntimeError(f"Jev chose {option!r}, which is not a known route")

        quality_answer = answers.get("quality") or {}
        quality = quality_answer.get("choice")
        if quality not in ("clean", "degraded", "handwritten"):
            quality = "clean"

        usage = body.get("usage") or {}
        return (
            ProcessingRoute(option),
            quality,
            float(route_answer.get("confidence", 0.0)),
            str(body.get("model", "")),
            int(usage.get("input_tokens", 0)),
        )

    def _parse_legibility(
        self, body: dict
    ) -> tuple[ProcessingRoute, str, float, str, int]:
        """Read the E3-or-E5 decision, the tier and the confidence out of a response."""
        answers = (body or {}).get("answers") or {}
        legibility = answers.get("legibility")
        if not isinstance(legibility, dict) or "choice" not in legibility:
            raise RuntimeError("Jev response has no legibility answer")

        option = legibility["choice"]
        if option not in LEGIBILITY_CRITERIA:
            raise RuntimeError(
                f"Jev chose {option!r}, which is not a known legibility option"
            )

        handwritten = (answers.get("handwritten") or {}).get("noul")
        if option == "E3":
            tier = "clean"
        elif isinstance(handwritten, (int, float)) and handwritten >= 0.5:
            tier = "handwritten"
        else:
            tier = "degraded"

        usage = body.get("usage") or {}
        return (
            ProcessingRoute(option),
            tier,
            float(legibility.get("confidence", 0.0)),
            str(body.get("model", "")),
            int(usage.get("input_tokens", 0)),
        )

    def _build_ocr_state(self, page_number: int, ocr_result: dict) -> str:
        text = (ocr_result.get("text") or "").strip()
        body = (
            text[: self.config.ocr_text_char_cap]
            if text
            else "tesseract recognized no words on this page."
        )
        return (
            f"OCR of page {page_number + 1} | "
            f"mean confidence {ocr_result.get('mean_confidence', 0.0)} | "
            f"word count {ocr_result.get('word_count', 0)}\n\n"
            f"Recognized text:\n{body}"
        )

    def refine_after_ocr(
        self, signals: PageSignals, result: ClassificationResult, ocr_result: dict
    ) -> ClassificationResult:
        """Settle E3 against E5 using tesseract's actual output.

        A page can only reach E5 from here. The pipeline's extraction chain has
        no E5 branch, so a classifier that returned E5 from `classify` would
        produce a record whose `extracted` dict is empty and which carries no
        OCR output at all.

        On failure the first-stage route is kept. The OCR result is already
        attached to the page by then, and losing the legibility call is a much
        smaller loss than discarding it.
        """
        try:
            body = self._call(
                {
                    "state": self._build_ocr_state(signals.page_number, ocr_result),
                    "model": self.config.model,
                    "questions": {
                        "legibility": {
                            "type": "choice",
                            "instructions": (
                                "Could tesseract read this page reliably enough "
                                "to use its text without a human re-reading it?"
                            ),
                            "criteria": LEGIBILITY_CRITERIA,
                        },
                        "handwritten": {
                            "type": "noul",
                            "instructions": (
                                "Is this page handwritten rather than machine "
                                "printed?"
                            ),
                        },
                    },
                }
            )
            route, tier, confidence, model, tokens = self._parse_legibility(body)
        except Exception as exc:
            return ClassificationResult(
                route=result.route,
                quality_tier=result.quality_tier,
                confidence=result.confidence,
                reason=f"Jev unavailable for the OCR legibility check ({exc})",
                signals={
                    **result.signals,
                    "classifier": self.name,
                    "jev_route_source": "threshold_fallback",
                    "jev_error": repr(exc),
                },
            )

        return ClassificationResult(
            route=route,
            quality_tier=tier,
            confidence=confidence,
            reason=(
                f"OCR mean confidence {ocr_result.get('mean_confidence', 0.0)} -> {tier}"
            ),
            signals={
                **result.signals,
                "classifier": self.name,
                "jev_route_source": "jev",
                "jev_model": model,
                "jev_input_tokens": tokens,
            },
        )

    def classify(self, signals: PageSignals, context=None) -> ClassificationResult:
        """Ask Jev to route one page.

        Sequential by necessity: this method sees one page at a time and cannot
        see what is coming, so there is nothing to batch. That costs roughly a
        second per page -- a forty-page drawing takes about forty seconds that
        the threshold classifier spent in microseconds. Acceptable while
        measuring whether Jev is worth its latency; if it is, batching becomes
        a pipeline change rather than a classifier one.
        """
        try:
            body = self._call(self._build_payload(signals))
            route, quality, confidence, model, tokens = self._parse(body)
        except Exception as exc:
            return self._fallback_result(signals, exc)

        return ClassificationResult(
            route=route,
            quality_tier=quality,
            confidence=confidence,
            reason=f"Jev chose {route.value} at {confidence:.2f} confidence",
            signals={
                **{
                    k: v
                    for k, v in asdict(signals).items()
                    if k != "text_sample"
                },
                "jev_route_source": "jev",
                "jev_model": model,
                "jev_input_tokens": tokens,
            },
        )
