"""OpenAI-compatible chat-completion backend.

Credentials are read from an environment variable at construction time and
are never accepted through a checked-in configuration file.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from ..schemas import GenerationRequest, GenerationResult


class OpenAICompatibleGenerator:
    """Generate one chat completion per request with strict ID preservation.

    Parameters are deliberately backend-only. Prompt rendering and response
    parsing belong to the task components, which keeps this adapter reusable.
    An already constructed client can be injected for testing.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: int | None = 42,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not api_key_env.strip():
            raise ValueError("api_key_env must not be empty")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self.model = model.strip()
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self._client = client or self._create_client(api_key_env, base_url)

    @staticmethod
    def _create_client(api_key_env: str, base_url: str | None) -> Any:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set the {api_key_env} environment variable before creating the backend"
            )
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Bactrainus with the 'api' extra") from error

        options: dict[str, object] = {"api_key": api_key}
        if base_url is not None:
            options["base_url"] = base_url
        return OpenAI(**options)

    def generate(self, requests: Sequence[GenerationRequest]) -> tuple[GenerationResult, ...]:
        """Generate completions in request order without changing identifiers."""

        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate request IDs are not allowed within a batch")
        results: list[GenerationResult] = []
        for request in requests:
            options: dict[str, object] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            if self.seed is not None:
                options["seed"] = self.seed
            response = self._client.chat.completions.create(**options)
            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError(f"backend returned no choice for {request.request_id!r}")
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"backend returned empty text for {request.request_id!r}")
            results.append(GenerationResult(request.request_id, content))
        return tuple(results)
