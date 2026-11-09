"""Shared fixtures: a small offline knowledge base and a scripted fake Claude client."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from econ_agent.config import Settings
from econ_agent.embeddings import HashingEmbedder
from econ_agent.indicators import IndicatorStore, build_monthly_rows
from econ_agent.tools import ToolExecutor
from econ_agent.vector_store import VectorStore

DOCS = [
    (275, "statement", "2025-12-10", "Copom mantém a taxa Selic em 15,00% a.a.",
     "O Copom decidiu manter a taxa Selic em 15,00% a.a. As expectativas de inflação seguem desancoradas."),
    (277, "statement", "2026-03-18", "Copom reduz a taxa Selic para 14,75% a.a.",
     "Diante da moderação da atividade e da queda da inflação corrente, o Copom reduziu a taxa Selic para 14,75% a.a."),
    (282, "statement", "2026-11-04", "Copom mantém a taxa Selic em 13,75% a.a.",
     "O ambiente externo tornou-se mais adverso com os conflitos no Oriente Médio e a alta do petróleo."),
]


def _months(start: date, n: int) -> list[date]:
    out, y, m = [], start.year, start.month
    for _ in range(n):
        out.append(date(y, m, 1))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


@pytest.fixture
def index(embedder):
    store = VectorStore(embedder.name, embedder.dim)
    records = [
        {"id": f"{t}-{m}-0", "meeting": m, "doc_type": t, "date": d, "title": title, "url": f"https://example/{m}", "text": text}
        for m, t, d, title, text in DOCS
    ]
    store.add(embedder.embed([f"{r['title']}\n{r['text']}" for r in records]), records)
    return store


@pytest.fixture
def indicators(tmp_path):
    months = _months(date(2024, 1, 1), 30)  # 2024-01 .. 2026-06
    usd = [(date(d.year, d.month, day), 5.0 + i * 0.01 + day / 1000) for i, d in enumerate(months) for day in (2, 15, 28)]
    selic = [(date(d.year, d.month, 28), 15.0 if d.year == 2025 else 14.0) for d in months]
    ipca = [(d, 0.4) for d in months]
    store = IndicatorStore(tmp_path / "indicators.db")
    store.write(build_monthly_rows(usd, selic, ipca))
    return store


@pytest.fixture
def executor(index, embedder, indicators):
    return ToolExecutor(index, embedder, indicators)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, offline_embeddings=True, fallback_model="", max_iterations=4)


# --- Fake Claude client -----------------------------------------------------------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def response(content, stop_reason, input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeClient:
    """Returns scripted responses in order and records every request."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._scripted:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._scripted.pop(0)
