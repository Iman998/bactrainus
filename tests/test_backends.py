from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from bactrainus.backends import HuggingFaceGenerator, OpenAICompatibleGenerator
from bactrainus.schemas import GenerationRequest


def request(identifier: str = "r1") -> GenerationRequest:
    return GenerationRequest(identifier, "example", "System", "User")


class FakeCompletions:
    def __init__(self, content: str = "Answer") -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def test_openai_compatible_backend_preserves_ids_and_options() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAICompatibleGenerator(
        model="model",
        client=client,
        temperature=0.2,
        top_p=0.9,
        max_tokens=12,
        seed=7,
    )

    result = backend.generate((request(),))

    assert result[0].request_id == "r1"
    assert result[0].text == "Answer"
    assert completions.calls[0]["seed"] == 7
    assert completions.calls[0]["max_tokens"] == 12


def test_openai_backend_rejects_invalid_or_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="top_p"):
        OpenAICompatibleGenerator(model="m", client=object(), top_p=0.0)
    with pytest.raises(ValueError, match="duplicate"):
        OpenAICompatibleGenerator(model="m", client=SimpleNamespace()).generate(
            (request(), request())
        )

    empty = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(content="   ")))
    with pytest.raises(RuntimeError, match="empty text"):
        OpenAICompatibleGenerator(model="m", client=empty).generate((request(),))

    monkeypatch.delenv("MISSING_BACTRAINUS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_BACTRAINUS_KEY"):
        OpenAICompatibleGenerator(model="m", api_key_env="MISSING_BACTRAINUS_KEY")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": " "}, "model"),
        ({"model": "m", "api_key_env": " "}, "api_key_env"),
        ({"model": "m", "temperature": -1}, "temperature"),
        ({"model": "m", "max_tokens": 0}, "max_tokens"),
    ],
)
def test_openai_constructor_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleGenerator(client=object(), **kwargs)  # type: ignore[arg-type]


def test_openai_client_is_created_only_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    module = ModuleType("openai")

    def constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace()

    module.OpenAI = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("SAFE_TEST_KEY", "secret-value")

    OpenAICompatibleGenerator(
        model="model", api_key_env="SAFE_TEST_KEY", base_url="https://example.invalid/v1"
    )

    assert captured == {
        "api_key": "secret-value",
        "base_url": "https://example.invalid/v1",
    }


def test_openai_backend_rejects_missing_choice() -> None:
    completions = SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with pytest.raises(RuntimeError, match="no choice"):
        OpenAICompatibleGenerator(model="m", client=client).generate((request(),))


def test_huggingface_backend_extracts_only_continuation() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def generate(prompt: str, **kwargs: object) -> object:
        calls.append((prompt, kwargs))
        return [{"generated_text": prompt + "Paris"}]

    backend = HuggingFaceGenerator(generate, max_new_tokens=9)
    result = backend.generate((request(),))

    assert result[0].text == "Paris"
    assert calls[0][1]["max_new_tokens"] == 9
    assert calls[0][1]["do_sample"] is False


def test_huggingface_sampling_and_contract_validation() -> None:
    with pytest.raises(ValueError, match="do_sample"):
        HuggingFaceGenerator(lambda _: "x", temperature=0.1)
    with pytest.raises(ValueError, match="top_p"):
        HuggingFaceGenerator(lambda _: "x", do_sample=True, top_p=2.0)

    backend = HuggingFaceGenerator(lambda prompt, **_: prompt)
    with pytest.raises(RuntimeError, match="empty"):
        backend.generate((request(),))

    duplicate_backend = HuggingFaceGenerator(lambda prompt, **_: prompt + "x")
    with pytest.raises(ValueError, match="duplicate"):
        duplicate_backend.generate((request(), request()))


@pytest.mark.parametrize(
    "raw, expected",
    [("answer", "answer"), ([{"generated_text": "promptanswer"}], "answer")],
)
def test_huggingface_result_shapes(raw: object, expected: str) -> None:
    assert HuggingFaceGenerator._extract_text(raw, "prompt") == expected


def test_huggingface_rejects_unsupported_result_shape() -> None:
    with pytest.raises(RuntimeError, match="unsupported"):
        HuggingFaceGenerator._extract_text([], "prompt")
    with pytest.raises(RuntimeError, match="generated_text"):
        HuggingFaceGenerator._extract_text([{"other": "value"}], "prompt")
    with pytest.raises(RuntimeError, match="unsupported result item"):
        HuggingFaceGenerator._extract_text([42], "prompt")


def test_huggingface_from_pretrained_uses_native_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("transformers")

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages: object, *, tokenize: bool, add_generation_prompt: bool
        ) -> str:
            assert messages
            assert not tokenize
            assert add_generation_prompt
            return "native-prompt"

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> Tokenizer:
            return Tokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> object:
            return object()

    def pipeline(*_: object, **__: object) -> object:
        return lambda prompt, **kwargs: [{"generated_text": prompt + "response"}]

    module.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    module.AutoModelForCausalLM = AutoModel  # type: ignore[attr-defined]
    module.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)

    backend = HuggingFaceGenerator.from_pretrained("local/model")

    assert backend.generate((request(),))[0].text == "response"
