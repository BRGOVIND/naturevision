"""Evidence-grounded interpretation via a hosted language model.

The language layer performs no measurement. It receives a finished evidence
package, writes prose about it, and its output is schema-validated and
grounding-checked before it is allowed anywhere near a report.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.core.errors import InterpretationProviderError, InterpretationUnavailableError
from app.core.logging import get_logger
from app.interpretation.evidence import EvidencePackage
from app.interpretation.prompts import (
    SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    build_interpretation_prompt,
    build_vision_prompt,
)
from app.interpretation.schemas import (
    Interpretation,
    InterpretationEnvelope,
    VisualInterpretation,
)
from app.interpretation.validation import validate_interpretation

logger = get_logger(__name__)

PROVIDER_NAME = "Groq"

#: One retry with an explicit correction turn. Beyond that a provider that
#: cannot follow the schema is treated as unusable rather than retried forever.
MAX_ATTEMPTS = 2

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


class LanguageInterpretationService:
    """Generates and validates structured environmental interpretation."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None
        self._model_cache: set[str] | None = None

    @property
    def available(self) -> bool:
        return settings.language_enabled

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=settings.groq_base_url,
                timeout=httpx.Timeout(settings.language_timeout_seconds),
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # --- capability probing --------------------------------------------------
    async def available_models(self) -> set[str] | None:
        """Model identifiers this account can actually serve.

        Returns None when the catalogue cannot be reached. Cached for the life
        of the service, since an account's model list does not change during a
        process lifetime.
        """
        if not self.available:
            return set()
        if self._model_cache is not None:
            return self._model_cache
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            self._model_cache = {str(entry.get("id")) for entry in response.json().get("data", [])}
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("model_catalogue_unavailable", error=str(exc)[:200])
            return None
        return self._model_cache

    async def vision_supported(self) -> bool:
        """Whether visual interpretation can genuinely run on this deployment.

        Vision needs three things: a configured provider, the feature enabled,
        and a vision-capable model the account can actually call. Reporting the
        capability without the third check would offer the user a button that
        is guaranteed to fail.
        """
        if not self.available or not settings.enable_vision_interpretation:
            return False
        models = await self.available_models()
        if models is None:
            return False
        return settings.vision_model in models

    # --- text interpretation ------------------------------------------------
    async def interpret(self, evidence: EvidencePackage) -> InterpretationEnvelope:
        """Generate an interpretation, or explain why one is unavailable.

        A missing provider key is a configuration state, not an error: the rest
        of the analysis is fully valid without it, so the envelope records the
        absence instead of failing the run.
        """
        if not self.available:
            return InterpretationEnvelope(
                available=False,
                unavailable_reason=(
                    "No language provider is configured, so the automated "
                    "interpretation section was not generated. All measured "
                    "results are unaffected."
                ),
            )

        payload = evidence.to_dict()
        allowed_numbers = evidence.numeric_claims()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_interpretation_prompt(payload)},
        ]

        last_error: str | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            raw = await self._complete(messages, settings.language_model)
            try:
                interpretation = Interpretation.model_validate(_extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_error = f"schema: {exc}"
                logger.warning(
                    "interpretation_schema_invalid", attempt=attempt, error=str(exc)[:400]
                )
                messages.append({"role": "assistant", "content": raw[:4000]})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required JSON schema. "
                            "Return only a JSON object with keys summary, observations, "
                            "interpretation, uncertainty, limitations, confidence_qualifier."
                        ),
                    }
                )
                continue

            grounding = validate_interpretation(interpretation, allowed_numbers)
            if grounding.passed:
                return InterpretationEnvelope(
                    interpretation=interpretation,
                    provider=PROVIDER_NAME,
                    model=settings.language_model,
                    generated_at=dt.datetime.now(dt.UTC).isoformat(),
                    grounding=grounding.to_dict(),
                    available=True,
                )

            last_error = f"grounding: unsupported numbers {grounding.unsupported_numbers[:5]}"
            messages.append({"role": "assistant", "content": raw[:4000]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "These numbers do not appear in the evidence package: "
                        f"{grounding.unsupported_numbers[:10]}. Rewrite using only "
                        "values present in the evidence, or omit the figure entirely."
                    ),
                }
            )

        logger.error("interpretation_failed", reason=last_error)
        raise InterpretationProviderError(
            "The interpretation could not be validated against the measured "
            "evidence and was discarded.",
            details={"reason": "failed_grounding_or_schema_validation"},
        )

    # --- vision interpretation ----------------------------------------------
    async def interpret_image(
        self, image_data_url: str, layer_label: str, context: str
    ) -> VisualInterpretation | None:
        """Describe a rendered analysis layer with a vision-capable model."""
        if not await self.vision_supported():
            logger.info("vision_interpretation_skipped", model=settings.vision_model)
            return None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_vision_prompt(layer_label, context)},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
        try:
            raw = await self._complete(messages, settings.vision_model)
            return VisualInterpretation.model_validate(_extract_json(raw))
        except (InterpretationProviderError, ValidationError, ValueError) as exc:
            # Visual interpretation is supplementary; its absence must never
            # invalidate the measured analysis.
            logger.warning("vision_interpretation_failed", error=str(exc)[:300])
            return None

    # --- transport ----------------------------------------------------------
    async def _complete(self, messages: list[dict[str, Any]], model: str) -> str:
        if not settings.groq_api_key:
            raise InterpretationUnavailableError()

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": settings.language_temperature,
            "max_tokens": settings.language_max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self.client.post("/chat/completions", json=body)
            response.raise_for_status()
            document = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "language_provider_http_error",
                status=exc.response.status_code,
                body=exc.response.text[:400],
            )
            raise InterpretationProviderError(
                "The interpretation provider rejected the request.",
                details={"status": exc.response.status_code},
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("language_provider_transport_error", error=str(exc))
            raise InterpretationProviderError(
                "The interpretation provider is unreachable."
            ) from exc

        try:
            return document["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise InterpretationProviderError(
                "The interpretation provider returned an unexpected payload."
            ) from exc


def _extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating code fences."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty response from the interpretation provider.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _JSON_BLOCK.search(text)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No JSON object found in the interpretation response.")
