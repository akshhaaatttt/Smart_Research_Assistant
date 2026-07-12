
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from src.generation.llm import BaseLLM, build_llm
from src.generation.prompt_builder import PromptBuilder, Turn
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.schemas import RetrievedChunk

logger = get_logger(__name__)


@dataclass
class GenerationOutput:
    answer: str
    used_contexts: List[RetrievedChunk]
    backend: str
    model: str
    usage: dict = field(default_factory=dict)
    latency_seconds: float = 0.0

    @property
    def sources(self) -> List[str]:
        seen: List[str] = []
        for rc in self.used_contexts:
            label = self._label(rc)
            if label not in seen:
                seen.append(label)
        return seen

    @staticmethod
    def _label(rc: RetrievedChunk) -> str:
        meta = rc.chunk.metadata
        name = meta.get("filename") or meta.get("title") or meta.get("url") or rc.source
        return f"{name}, page {meta['page']}" if "page" in meta else str(name)


class AnswerGenerator:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or AppConfig.load()
        self.llm = llm or build_llm(self.config)
        # Size the context budget to the backend's input window so the prompt fits
        # (small local models otherwise truncate and degenerate).
        self.prompt_builder = PromptBuilder(
            self.config, max_context_tokens=getattr(self.llm, "context_window", None)
        )

    def generate(
        self,
        query: str,
        contexts: List[RetrievedChunk],
        history: Optional[Sequence[Turn]] = None,
    ) -> GenerationOutput:
        """Generate a grounded answer for ``query`` given retrieved ``contexts`` and
        optional prior conversation ``history``."""
        prompt = self.prompt_builder.build(query, contexts, history=history)

        start = time.perf_counter()
        result = self.llm.generate(prompt)
        latency = time.perf_counter() - start

        logger.info(
            "Generated answer via %s/%s in %.2fs (%d context(s))",
            result.backend, result.model, latency, len(prompt.used_contexts),
        )
        return GenerationOutput(
            answer=result.text,
            used_contexts=prompt.used_contexts,
            backend=result.backend,
            model=result.model,
            usage=result.usage,
            latency_seconds=latency,
        )
