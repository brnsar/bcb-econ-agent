"""Tool-using agent: Claude decides when to search Copom documents and when to read indicators."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from .config import Settings
from .tools import TOOL_DEFINITIONS, ToolExecutor

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
MAX_TOKENS = 16000

SYSTEM_PROMPT = """\
You are an economic research assistant specialized in Brazilian monetary policy. You answer \
questions using two tools: `search_copom_documents` (Copom statements and minutes) and \
`get_indicators` (monthly Selic, IPCA, USD/BRL and real interest rate).

How to work:
- Use `search_copom_documents` for reasoning, assessments and guidance ("why", "what did Copom say").
- Use `get_indicators` for any number. Every number in your answer must come from a tool result \
in this conversation, never from memory. If the data is not available, say so.
- Questions that mix both ("why did they cut, and what was inflation then?") need both tools. \
You can call several tools in parallel.
- Cite sources inline: "(Copom #281, minutes)" for documents and "(indicators, 2025-12)" for numbers.
- Answer in the same language as the question. Be concise: a short direct answer first, then the \
supporting details.
- When a question says "last", "current" or "latest", check `available_range` in the indicator \
results or the meeting dates in the documents instead of assuming.

Limits:
- Only answer questions about the Brazilian economy and monetary policy. For anything else, \
politely say what you can help with, without calling tools.
- Do not give personalized investment advice (what to buy, sell or invest in). You may explain \
what the data and Copom documents say.
"""


@dataclass
class ToolCall:
    name: str
    input: dict
    is_error: bool
    output: str


@dataclass
class AgentResult:
    question: str
    answer: str
    stop_reason: str
    iterations: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tools_used(self) -> set[str]:
        return {c.name for c in self.tool_calls}


class QuestionRejected(ValueError):
    """Guardrail G1: the question is rejected before any model call."""


def _text_of(content) -> str:
    return "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text").strip()


class EconAgent:
    def __init__(
        self,
        client,
        executor: ToolExecutor,
        settings: Settings,
        fallback_state_factory: Callable[[], contextlib.AbstractContextManager] | None = None,
        today: date | None = None,
    ) -> None:
        self.client = client
        self.executor = executor
        self.settings = settings
        self.fallback_state_factory = fallback_state_factory
        self.today = today

    def _system(self) -> list[dict]:
        today = (self.today or date.today()).isoformat()
        return [
            # Stable block first so it can be cached; the date goes in a separate block after it.
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Today's date: {today}."},
        ]

    def _create(self, messages: list) -> object:
        return self.client.beta.messages.create(
            model=self.settings.chat_model,
            max_tokens=MAX_TOKENS,
            system=self._system(),
            tools=TOOL_DEFINITIONS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
        )

    def ask(self, question: str) -> AgentResult:
        question = (question or "").strip()
        if not question:
            raise QuestionRejected("The question is empty.")
        if len(question) > MAX_QUESTION_CHARS:
            raise QuestionRejected(f"The question has more than {MAX_QUESTION_CHARS} characters.")

        state = self.fallback_state_factory() if self.fallback_state_factory else contextlib.nullcontext()
        with state:
            return self._loop(question)

    def _loop(self, question: str) -> AgentResult:
        messages: list = [{"role": "user", "content": question}]
        result = AgentResult(question=question, answer="", stop_reason="", iterations=0)

        for iteration in range(1, self.settings.max_iterations + 1):
            response = self._create(messages)
            result.iterations = iteration
            usage = getattr(response, "usage", None)
            if usage is not None:
                result.input_tokens += getattr(usage, "input_tokens", 0) or 0
                result.output_tokens += getattr(usage, "output_tokens", 0) or 0

            stop = response.stop_reason
            result.stop_reason = stop

            if stop == "refusal":
                result.answer = "I can't help with this request."
                return result
            if stop == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if stop != "tool_use" or not tool_uses:
                result.answer = _text_of(response.content)
                if stop == "max_tokens":
                    result.answer += "\n\n[Answer truncated: output limit reached.]"
                return result

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_uses:
                output = self.executor.execute(block.name, dict(block.input))
                log.info("tool %s(%s) -> error=%s", block.name, block.input, output.is_error)
                result.tool_calls.append(ToolCall(block.name, dict(block.input), output.is_error, output.content))
                result.sources.extend(output.sources)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output.content,
                        "is_error": output.is_error,
                    }
                )
            # All results go back in a single user message (keeps parallel tool calls working).
            messages.append({"role": "user", "content": tool_results})

        result.stop_reason = "max_iterations"
        result.answer = "I could not finish the analysis within the step limit. Try a more specific question."
        return result
