"""Local Hugging Face text-generation backend with lazy dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..schemas import GenerationRequest, GenerationResult

ChatRenderer = Callable[[GenerationRequest], str]
TextGenerationCallable = Callable[..., object]


def _plain_chat(request: GenerationRequest) -> str:
    """Render a backend-neutral fallback transcript."""

    return f"System:\n{request.system_prompt}\n\nUser:\n{request.user_prompt}\n\nAssistant:\n"


class HuggingFaceGenerator:
    """Wrap a local Transformers text-generation pipeline.

    Use :meth:`from_pretrained` for normal operation. The public constructor
    accepts an injected generation callable and renderer so the adapter can be
    tested without loading model weights.
    """

    def __init__(
        self,
        generate_text: TextGenerationCallable,
        *,
        render_chat: ChatRenderer = _plain_chat,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be positive when provided")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1] when provided")
        if not do_sample and (temperature is not None or top_p is not None):
            raise ValueError("temperature and top_p require do_sample=True")
        self._generate_text = generate_text
        self._render_chat = render_chat
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> HuggingFaceGenerator:
        """Load a model, tokenizer, and model-native chat template lazily."""

        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Bactrainus with the 'inference' extra") from error

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        def render_chat(request: GenerationRequest) -> str:
            messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ]
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return cls(
            generator,
            render_chat=render_chat,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    @staticmethod
    def _extract_text(raw: object, prompt: str) -> str:
        if isinstance(raw, str):
            generated = raw
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
            first = raw[0]
            if not isinstance(first, Mapping):
                raise RuntimeError("local backend returned an unsupported result item")
            candidate = first.get("generated_text")
            if not isinstance(candidate, str):
                raise RuntimeError("local backend result has no generated_text string")
            generated = candidate
        else:
            raise RuntimeError("local backend returned an unsupported result")
        continuation = generated[len(prompt) :] if generated.startswith(prompt) else generated
        if not continuation.strip():
            raise RuntimeError("local backend returned empty generated text")
        return continuation.strip()

    def generate(self, requests: Sequence[GenerationRequest]) -> tuple[GenerationResult, ...]:
        """Generate locally in request order and preserve request IDs."""

        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate request IDs are not allowed within a batch")
        results: list[GenerationResult] = []
        for request in requests:
            prompt = self._render_chat(request)
            options: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "return_full_text": True,
            }
            if self.temperature is not None:
                options["temperature"] = self.temperature
            if self.top_p is not None:
                options["top_p"] = self.top_p
            raw = self._generate_text(prompt, **options)
            results.append(GenerationResult(request.request_id, self._extract_text(raw, prompt)))
        return tuple(results)
