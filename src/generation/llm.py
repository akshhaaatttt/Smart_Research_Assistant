from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

from src.generation.prompt_builder import BuiltPrompt
from src.utils.config import AppConfig
from src.utils.exceptions import GenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResult:
    text: str
    backend: str
    model: str
    usage: dict = field(default_factory=dict)


class BaseLLM(abc.ABC):
    backend: str = "base"

    def __init__(
        self, model: str, temperature: float, max_tokens: int, context_window: int = 2048
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Max prompt tokens the model accepts; used to budget context.
        self.context_window = context_window

    @abc.abstractmethod
    def generate(self, prompt: BuiltPrompt) -> LLMResult:
        """Produce an answer for the given built prompt."""

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can actually run (deps/keys present)."""


class OpenAILLM(BaseLLM):
    backend = "openai"

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        base_url: Optional[str] = None,
        context_window: int = 6000,
    ) -> None:
        super().__init__(model, temperature, max_tokens, context_window)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        if api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key, base_url=base_url or None)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not initialise OpenAI client: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: BuiltPrompt) -> LLMResult:
        if not self.is_available:
            raise GenerationError("OpenAI backend is not configured (missing API key).")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=prompt.as_chat_messages(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise GenerationError(f"OpenAI generation failed: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResult(text=text, backend=self.backend, model=self.model, usage=usage)


class LocalLLM(BaseLLM):
    """Offline HuggingFace seq2seq model (FLAN-T5 by default)."""

    backend = "local"

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        max_input_tokens: int = 512,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.3,
    ) -> None:
        super().__init__(model, temperature, max_tokens, context_window=max_input_tokens)
        self.max_input_tokens = max_input_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                pipeline,
            )

            logger.info("Loading local LLM '%s' (first run downloads weights)...", self.model)
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            # Cap input length so truncation cuts to the encoder window rather than
            # the tokenizer's unbounded default.
            tokenizer.model_max_length = self.max_input_tokens
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model)
            self._pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )
        except Exception as exc:
            raise GenerationError(f"Failed to load local LLM '{self.model}': {exc}") from exc

    @property
    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except Exception:  # pragma: no cover
            return False

    def generate(self, prompt: BuiltPrompt) -> LLMResult:
        self._ensure_loaded()
        text_in = prompt.as_flat_text()
        try:
            # Beam search with n-gram blocking + repetition penalty prevents the
            # copy/repeat degeneration small models fall into on dense text. Input is
            # hard-truncated to the encoder window.
            gen_kwargs = dict(
                max_new_tokens=self.max_tokens,
                num_beams=self.num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                truncation=True,
                early_stopping=True,
            )
            if self.temperature > 0.05:
                gen_kwargs.update(do_sample=True, temperature=self.temperature)
            output = self._pipeline(text_in, **gen_kwargs)
            answer = output[0]["generated_text"].strip()
        except Exception as exc:
            raise GenerationError(f"Local generation failed: {exc}") from exc

        # Approximate token usage for the dashboards.
        tok = self._pipeline.tokenizer
        usage = {
            "prompt_tokens": len(tok.encode(text_in)),
            "completion_tokens": len(tok.encode(answer)),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return LLMResult(text=answer, backend=self.backend, model=self.model, usage=usage)


def build_llm(config: Optional[AppConfig] = None) -> BaseLLM:
    """Construct the LLM backend selected by config (with ``auto`` resolution)."""
    config = config or AppConfig.load()
    gen = config.section("generation")
    backend = (gen.get("backend", "auto") or "auto").lower()
    temperature = float(gen.get("temperature", 0.2))
    max_tokens = int(gen.get("max_tokens", 512))

    openai_cfg = gen.get("openai", {})
    api_key = (openai_cfg.get("api_key") or "").strip()

    if backend == "auto":
        backend = "openai" if api_key else "local"
        logger.info("Generation backend auto-resolved to '%s'", backend)

    if backend == "openai":
        return OpenAILLM(
            model=openai_cfg.get("model", "gpt-4o-mini"),
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=openai_cfg.get("base_url"),
            context_window=openai_cfg.get("max_context_tokens", 6000),
        )
    if backend == "local":
        local_cfg = gen.get("local", {})
        return LocalLLM(
            model=local_cfg.get("model_name", "google/flan-t5-base"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_input_tokens=local_cfg.get("max_input_tokens", 512),
            num_beams=local_cfg.get("num_beams", 4),
            no_repeat_ngram_size=local_cfg.get("no_repeat_ngram_size", 3),
            repetition_penalty=local_cfg.get("repetition_penalty", 1.3),
        )
    raise GenerationError(f"Unknown generation backend '{backend}'.")
