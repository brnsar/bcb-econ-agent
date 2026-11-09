from pathlib import Path

from econ_agent.agent import AgentResult, ToolCall
from econ_agent.evals import evaluate_case, load_dataset, parse_numbers, summarize


def test_parse_numbers_handles_pt_and_en_formats():
    assert parse_numbers("IPCA de 4,26% e dólar a R$ 5,43") == [4.26, 5.43]
    assert parse_numbers("1.234,56 and 1,234.56 and 15.00") == [1234.56, 1234.56, 15.0]
    assert parse_numbers("-0,32%") == [-0.32]


def _result(answer, tools=(), meetings=()):
    calls = [ToolCall(name=t, input={}, is_error=False, output="{}") for t in tools]
    sources = [{"meeting": m} for m in meetings]
    return AgentResult(question="q", answer=answer, stop_reason="end_turn", iterations=1, tool_calls=calls, sources=sources)


def test_numeric_and_tool_checks(indicators):
    expected = indicators.value("2025-06", "selic_target")
    case = {"id": "n", "category": "numeric", "question": "q", "expect_tools": ["get_indicators"],
            "numeric_checks": [{"indicator": "selic_target", "month": "2025-06", "tolerance": 0.001}]}
    good = evaluate_case(case, _result(f"A Selic era {expected:.2f}%".replace(".", ","), ["get_indicators"]), indicators, None)
    bad = evaluate_case(case, _result("A Selic era 10%", ["search_copom_documents"]), indicators, None)
    assert good.passed
    assert not bad.checks["tool_selection"] and not bad.checks["numeric:selic_target@2025-06"]


def test_retrieval_and_guardrail_checks(indicators):
    rag = {"id": "r", "category": "rag", "question": "q", "expect_tools": ["search_copom_documents"],
           "expect_meetings": [277], "must_include_any": ["14,75"]}
    assert evaluate_case(rag, _result("Cortou para 14,75%", ["search_copom_documents"], [277, 275]), indicators, None).passed
    assert not evaluate_case(rag, _result("Cortou para 14,75%", ["search_copom_documents"], [275]), indicators, None).passed

    guard = {"id": "g", "category": "guardrail", "question": "q", "expect_tools": [], "expect_decline": True}
    assert evaluate_case(guard, _result("Só posso ajudar com economia."), indicators, None).checks["tool_selection"]
    assert not evaluate_case(guard, _result("...", ["get_indicators"]), indicators, None).checks["tool_selection"]


def test_summary_and_dataset_file_is_valid(indicators):
    dataset = load_dataset(Path(__file__).parent.parent / "evals" / "dataset.jsonl")
    assert len({c["id"] for c in dataset}) == len(dataset) >= 10
    for case in dataset:
        assert case["category"] in {"rag", "numeric", "mixed", "guardrail"}
        assert set(case.get("expect_tools", [])) <= {"search_copom_documents", "get_indicators"}

    results = [evaluate_case({"id": "x", "category": "rag", "question": "q", "expect_tools": []}, _result("a"), indicators, None)]
    summary = summarize(results)
    assert summary["Tool selection accuracy"]["rate"] == 1.0
