import json
from datetime import date

from econ_agent.indicators import build_monthly_rows


def test_monthly_rows_derive_ipca_12m_and_real_rate():
    months = [date(2025, m, 1) for m in range(1, 13)]
    rows = build_monthly_rows(
        usd=[(date(2025, 12, 1), 5.0), (date(2025, 12, 31), 6.0)],
        selic=[(date(2025, 12, 10), 15.0), (date(2025, 12, 31), 14.75)],
        ipca=[(d, 0.5) for d in months],
    )
    dec = rows[-1]
    assert dec["month"] == "2025-12"
    assert dec["usd_avg"] == 5.5 and dec["usd_close"] == 6.0
    assert dec["selic_target"] == 14.75  # last value of the month
    assert dec["ipca_12m"] == round((1.005**12 - 1) * 100, 2)
    assert dec["real_rate_12m"] == round((1.1475 / (1 + dec["ipca_12m"] / 100) - 1) * 100, 2)
    assert "ipca_12m" not in rows[-2]  # only 11 months available before December


def test_get_indicators_returns_rows_and_sources(executor):
    out = executor.execute("get_indicators", {"start_month": "2025-01", "end_month": "2025-03", "indicators": ["selic_target"]})
    payload = json.loads(out.content)
    assert not out.is_error
    assert [r["month"] for r in payload["rows"]] == ["2025-01", "2025-02", "2025-03"]
    assert payload["rows"][0]["selic_target"] == 15.0
    assert payload["available_range"] == {"first_month": "2024-01", "last_month": "2026-06"}
    assert out.sources[0]["table"] == "monthly_indicators"


def test_get_indicators_validation_errors_are_tool_errors(executor):
    cases = [
        {"start_month": "2025-13", "end_month": "2025-12", "indicators": ["selic_target"]},
        {"start_month": "2025-06", "end_month": "2025-01", "indicators": ["selic_target"]},
        {"start_month": "2000-01", "end_month": "2025-01", "indicators": ["selic_target"]},
        {"start_month": "2025-01", "end_month": "2025-02", "indicators": ["DROP TABLE"]},
        {"start_month": "2025-01", "end_month": "2025-02", "indicators": []},
    ]
    for tool_input in cases:
        out = executor.execute("get_indicators", tool_input)
        assert out.is_error, tool_input
        assert out.content.startswith("Error:")


def test_search_returns_citable_results(executor):
    out = executor.execute("search_copom_documents", {"query": "Oriente Médio petróleo", "doc_type": "any", "top_k": 2})
    payload = json.loads(out.content)
    assert payload["results"][0]["meeting"] == 282
    assert payload["results"][0]["source"].startswith("Copom #282")
    assert {s["meeting"] for s in out.sources} >= {282}


def test_search_validation_and_unknown_tool(executor):
    assert executor.execute("search_copom_documents", {"query": "", "doc_type": "any", "top_k": 3}).is_error
    assert executor.execute("search_copom_documents", {"query": "x", "doc_type": "any", "top_k": 50}).is_error
    assert executor.execute("search_copom_documents", {"query": "x", "doc_type": "blog", "top_k": 3}).is_error
    assert executor.execute("delete_everything", {}).is_error
