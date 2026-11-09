"""Tool definitions exposed to the model and their validated implementations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .embeddings import Embedder
from .indicators import INDICATORS, IndicatorStore
from .vector_store import VectorStore

MAX_TOP_K = 8
MAX_INDICATOR_ROWS = 120
_MONTH_RE = re.compile(r"^(19|20)\d{2}-(0[1-9]|1[0-2])$")

TOOL_DEFINITIONS = [
    {
        "name": "search_copom_documents",
        "description": (
            "Semantic search over Copom (Brazilian Central Bank monetary policy committee) statements and "
            "meeting minutes. Use it for questions about WHY Copom decided something, its assessment of "
            "inflation, activity, the labor market, fiscal policy or the external scenario, and its guidance. "
            "Returns text excerpts with the meeting number, date and document type. "
            "top_k must be between 1 and 8."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, phrased as a short topic or question. Portuguese works best.",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["statement", "minutes", "any"],
                    "description": "statement = short decision note; minutes = detailed reasoning; any = both.",
                },
                "top_k": {"type": "integer", "description": "Number of excerpts to return (1-8)."},
            },
            "required": ["query", "doc_type", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_indicators",
        "description": (
            "Monthly Brazilian macro indicators from the Central Bank SGS system. Use it for any number: "
            "Selic target, IPCA inflation, USD/BRL exchange rate or the real interest rate. "
            "Months use the YYYY-MM format; the range can cover at most 120 months. Available indicators: "
            + "; ".join(f"{k} = {v}" for k, v in INDICATORS.items())
            + "."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "start_month": {"type": "string", "description": "First month, YYYY-MM."},
                "end_month": {"type": "string", "description": "Last month, YYYY-MM (inclusive)."},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(INDICATORS)},
                    "description": "Indicators to return.",
                },
            },
            "required": ["start_month", "end_month", "indicators"],
            "additionalProperties": False,
        },
    },
]


class ToolInputError(ValueError):
    """Raised when the model sends an input that fails validation (guardrail G2)."""


@dataclass
class ToolOutput:
    content: str
    is_error: bool = False
    sources: list[dict] = field(default_factory=list)


def _month_index(month: str) -> int:
    year, mon = month.split("-")
    return int(year) * 12 + int(mon)


class ToolExecutor:
    def __init__(self, index: VectorStore, embedder: Embedder, indicators: IndicatorStore) -> None:
        self.index = index
        self.embedder = embedder
        self.indicators = indicators

    def execute(self, name: str, tool_input: dict) -> ToolOutput:
        try:
            if name == "search_copom_documents":
                return self._search(tool_input)
            if name == "get_indicators":
                return self._indicators(tool_input)
            raise ToolInputError(f"Unknown tool: {name}")
        except ToolInputError as exc:
            return ToolOutput(content=f"Error: {exc}", is_error=True)

    # --- search_copom_documents -------------------------------------------------
    def _search(self, tool_input: dict) -> ToolOutput:
        query = str(tool_input.get("query", "")).strip()
        doc_type = tool_input.get("doc_type", "any")
        top_k = tool_input.get("top_k", 5)
        if not query:
            raise ToolInputError("query must not be empty")
        if len(query) > 500:
            raise ToolInputError("query must be at most 500 characters")
        if doc_type not in {"statement", "minutes", "any"}:
            raise ToolInputError("doc_type must be statement, minutes or any")
        if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ToolInputError(f"top_k must be an integer between 1 and {MAX_TOP_K}")

        query_vector = self.embedder.embed([query])[0]
        where = None if doc_type == "any" else (lambda r: r["doc_type"] == doc_type)
        hits = self.index.search(query_vector, top_k=top_k, where=where)
        results = [
            {
                "source": f"Copom #{h.record['meeting']} {h.record['doc_type']} ({h.record['date']})",
                "meeting": h.record["meeting"],
                "doc_type": h.record["doc_type"],
                "date": h.record["date"],
                "title": h.record["title"],
                "score": round(h.score, 3),
                "text": h.record["text"],
            }
            for h in hits
        ]
        sources = [
            {"meeting": h.record["meeting"], "doc_type": h.record["doc_type"], "date": h.record["date"], "url": h.record["url"]}
            for h in hits
        ]
        return ToolOutput(content=json.dumps({"results": results}, ensure_ascii=False), sources=sources)

    # --- get_indicators ----------------------------------------------------------
    def _indicators(self, tool_input: dict) -> ToolOutput:
        start = str(tool_input.get("start_month", ""))
        end = str(tool_input.get("end_month", ""))
        indicators = tool_input.get("indicators") or []
        for label, month in (("start_month", start), ("end_month", end)):
            if not _MONTH_RE.match(month):
                raise ToolInputError(f"{label} must use the YYYY-MM format, got {month!r}")
        if _month_index(start) > _month_index(end):
            raise ToolInputError("start_month must not be after end_month")
        if _month_index(end) - _month_index(start) + 1 > MAX_INDICATOR_ROWS:
            raise ToolInputError(f"the range can cover at most {MAX_INDICATOR_ROWS} months")
        if not isinstance(indicators, list) or not indicators:
            raise ToolInputError("indicators must be a non-empty list")
        unknown = [i for i in indicators if i not in INDICATORS]
        if unknown:
            raise ToolInputError(f"unknown indicators {unknown}; allowed: {list(INDICATORS)}")

        rows = self.indicators.query(start, end, list(dict.fromkeys(indicators)))
        lo, hi = self.indicators.month_range()
        payload = {
            "rows": rows,
            "available_range": {"first_month": lo, "last_month": hi},
            "units": {k: INDICATORS[k] for k in indicators},
        }
        if not rows:
            payload["note"] = "No data for this range. Check available_range."
        sources = [{"table": "monthly_indicators", "start_month": start, "end_month": end, "indicators": indicators}]
        return ToolOutput(content=json.dumps(payload, ensure_ascii=False), sources=sources)
