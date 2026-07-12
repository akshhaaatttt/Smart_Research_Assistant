from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.utils.config import AppConfig
from src.utils.schemas import RetrievedChunk

# A conversation turn is a (user question, assistant answer) pair.
Turn = Tuple[str, str]

_SYSTEM_GROUNDED = (
    "You are a precise research assistant. Answer the user's question using ONLY the "
    "provided context passages. Follow these rules strictly:\n"
    "1. Base every statement on the context. Do not use prior knowledge.\n"
    "2. Give the actual answer in a complete sentence, not just a reference.\n"
    "3. If the answer is not in the context, reply exactly: "
    "\"I don't have enough information in the provided documents to answer that.\"\n"
    "4. Be concise and factual. Do not speculate.\n"
    "5. You may optionally cite supporting passages with bracketed numbers like [1].\n"
    "6. Earlier conversation may be shown for context; use it only to interpret "
    "follow-up questions (e.g. what 'it' refers to), never as a source of facts."
)

# Rough chars-per-token ratio for budgeting without a tokenizer.
_CHARS_PER_TOKEN = 4


@dataclass
class BuiltPrompt:
    system: str
    user: str
    used_contexts: List[RetrievedChunk]

    def as_chat_messages(self) -> List[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def as_flat_text(self) -> str:
        """Single-string form for encoder-decoder models."""
        return f"{self.system}\n\n{self.user}"


class PromptBuilder:
    """Assemble grounded prompts from a query and retrieved context."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        max_context_tokens: Optional[int] = None,
    ) -> None:
        cfg = config or AppConfig.load()
        gen = cfg.section("generation")
        self.style: str = gen.get("system_prompt_style", "grounded")

        # Context budget must fit the model's input window or the backend silently
        # truncates the prompt and small models degenerate. Reserve room for the
        # system prompt and question, then budget the rest for context.
        window = max_context_tokens if max_context_tokens else 2048
        reserved_tokens = 220
        context_tokens = max(120, window - reserved_tokens)
        self.context_char_budget: int = context_tokens * _CHARS_PER_TOKEN
        self.max_history_turns: int = 4

    def build(
        self,
        query: str,
        contexts: List[RetrievedChunk],
        history: Optional[Sequence[Turn]] = None,
    ) -> BuiltPrompt:
        system = _SYSTEM_GROUNDED if self.style == "grounded" else _SYSTEM_GROUNDED

        # History competes with context for the window, so budget it first.
        history_text = self._format_history(history)
        remaining = max(120 * _CHARS_PER_TOKEN, self.context_char_budget - len(history_text))
        formatted, used = self._format_contexts(contexts, remaining)

        parts: List[str] = []
        if history_text:
            parts.append(history_text)
        parts.append(f"Context passages:\n{formatted if used else '(none retrieved)'}")
        parts.append(f"Question: {query}")
        if used:
            cue = "Answer the question in a complete sentence based on the passages above."
            if history_text:
                cue += " Use the conversation only to interpret the question."
            parts.append(f"{cue}\nAnswer:")
        else:
            parts.append("Answer:")

        return BuiltPrompt(system=system, user="\n\n".join(parts), used_contexts=used)

    def _format_history(self, history: Optional[Sequence[Turn]]) -> str:
        """Render recent turns as a compact, length-capped memory block."""
        if not history:
            return ""
        cap = max(200, self.context_char_budget // 3)
        lines: List[str] = []
        total = 0
        for question, answer in list(history)[-self.max_history_turns:]:
            answer = " ".join((answer or "").split())
            if len(answer) > 240:
                answer = answer[:240] + "..."
            entry = f"User: {question.strip()}\nAssistant: {answer}"
            if total + len(entry) > cap and lines:
                break
            lines.append(entry)
            total += len(entry)
        return "Conversation so far:\n" + "\n".join(lines) if lines else ""

    def _format_contexts(self, contexts: List[RetrievedChunk], budget: int):
        """Number and label contexts, truncating to the given char budget."""
        parts: List[str] = []
        used: List[RetrievedChunk] = []

        for i, rc in enumerate(contexts, start=1):
            label = self._source_label(rc)
            block = f"[{i}] (source: {label})\n{rc.content.strip()}"
            if len(block) > budget and used:
                break  # keep at least one context
            parts.append(block)
            used.append(rc)
            budget -= len(block)
            if budget <= 0:
                break
        return "\n\n".join(parts), used

    @staticmethod
    def _source_label(rc: RetrievedChunk) -> str:
        meta = rc.chunk.metadata
        name = meta.get("filename") or meta.get("title") or meta.get("url") or rc.source
        if "page" in meta:
            return f"{name}, page {meta['page']}"
        return str(name)
