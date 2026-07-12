"""Tests for prompt construction and answer generation."""

import pytest

from src.generation.generator import AnswerGenerator
from src.generation.llm import BaseLLM, LLMResult, LocalLLM, OpenAILLM, build_llm
from src.generation.prompt_builder import BuiltPrompt, PromptBuilder
from src.utils.config import AppConfig
from src.utils.schemas import Chunk, RetrievedChunk


def _rc(text, i, **meta):
    return RetrievedChunk(
        chunk=Chunk(content=text, source="hr_policy.txt", doc_id="d", chunk_index=i, metadata=meta),
        score=0.9 - 0.1 * i,
        rank=i,
    )


CTX = [
    _rc("Employees get 26 weeks of paid maternity leave.", 0, filename="hr_policy.txt", page=1),
    _rc("Annual leave is 24 days per year.", 1, filename="hr_policy.txt", page=2),
]


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #
def test_prompt_includes_grounding_rules_and_context():
    p = PromptBuilder().build("What is the maternity leave?", CTX)
    assert "ONLY the provided context" in p.system
    assert "[1]" in p.user and "[2]" in p.user
    assert "26 weeks" in p.user
    assert "page 1" in p.user  # source label with page


def test_prompt_with_no_context_signals_none():
    p = PromptBuilder().build("Anything?", [])
    assert "(none retrieved)" in p.user
    assert p.used_contexts == []


def test_prompt_includes_conversation_history():
    history = [("What is the maternity leave policy?", "Employees get 26 weeks.")]
    p = PromptBuilder().build("How much notice is required?", CTX, history=history)
    assert "Conversation so far:" in p.user
    assert "What is the maternity leave policy?" in p.user
    assert "How much notice is required?" in p.user  # current question still present
    # Grounding is preserved: context passages still included.
    assert "26 weeks" in p.user


def test_history_omitted_when_absent():
    p = PromptBuilder().build("A question?", CTX)
    assert "Conversation so far:" not in p.user


def test_history_does_not_crash_small_window():
    long_hist = [("q" * 500, "a" * 500) for _ in range(6)]
    p = PromptBuilder(max_context_tokens=512).build("follow up?", CTX, history=long_hist)
    assert "Question: follow up?" in p.user  # still well-formed


def test_prompt_budget_scales_with_model_window():
    """A small model window yields a proportionally smaller context budget so the
    prompt never overflows a tiny local model (the 1102>512 truncation bug)."""
    cfg = AppConfig.load()
    small = PromptBuilder(cfg, max_context_tokens=512)
    large = PromptBuilder(cfg, max_context_tokens=6000)
    assert small.context_char_budget < large.context_char_budget
    # 512-token window minus reserved tokens, times chars/token, stays modest.
    assert small.context_char_budget <= (512 - 200) * 4 + 10


def test_context_budget_truncates():
    cfg = AppConfig.load()
    pb = PromptBuilder(cfg)
    pb.context_char_budget = 60  # force truncation
    big = [_rc("x" * 100, i, filename="f") for i in range(5)]
    p = pb.build("q", big)
    assert len(p.used_contexts) < 5  # not all fit


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def test_auto_backend_without_key_is_local(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = AppConfig.load().with_overrides({"generation": {"backend": "auto"}})
    llm = build_llm(cfg)
    assert isinstance(llm, LocalLLM)


def test_auto_backend_with_key_is_openai():
    cfg = AppConfig.load().with_overrides(
        {"generation": {"backend": "auto", "openai": {"api_key": "sk-test123"}}}
    )
    llm = build_llm(cfg)
    assert isinstance(llm, OpenAILLM)
    assert llm.is_available  # client constructed


def test_openai_without_key_reports_unavailable():
    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.2, max_tokens=64, api_key="")
    assert not llm.is_available


# --------------------------------------------------------------------------- #
# Generator orchestration (with an injected fake LLM - no download)
# --------------------------------------------------------------------------- #
class _FakeLLM(BaseLLM):
    backend = "fake"

    def __init__(self):
        super().__init__("fake-model", 0.0, 64)
        self.last_prompt = None

    @property
    def is_available(self):
        return True

    def generate(self, prompt: BuiltPrompt) -> LLMResult:
        self.last_prompt = prompt
        return LLMResult(text="Answer with citation [1].", backend=self.backend,
                         model=self.model, usage={"total_tokens": 42})


def test_generator_wires_prompt_and_sources():
    gen = AnswerGenerator(llm=_FakeLLM())
    out = gen.generate("How much maternity leave?", CTX)
    assert out.answer == "Answer with citation [1]."
    assert out.backend == "fake"
    assert out.usage["total_tokens"] == 42
    assert out.sources == ["hr_policy.txt, page 1", "hr_policy.txt, page 2"]
    assert out.latency_seconds >= 0.0
    # The fake received a fully-built grounded prompt.
    assert "26 weeks" in gen.llm.last_prompt.user


# --------------------------------------------------------------------------- #
# Real local generation (integration; model is cached after first run)
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_local_generation_is_grounded():
    gen = AnswerGenerator(llm=LocalLLM("google/flan-t5-base", 0.0, 64))
    out = gen.generate("How many weeks of maternity leave?", CTX)
    assert "26" in out.answer
    assert out.usage["total_tokens"] > 0
