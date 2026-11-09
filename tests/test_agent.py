import pytest

from conftest import FakeClient, response, text_block, tool_use_block
from econ_agent.agent import EconAgent, QuestionRejected
from econ_agent.tools import TOOL_DEFINITIONS


def test_agent_runs_parallel_tools_and_returns_final_answer(executor, settings):
    client = FakeClient([
        response([
            tool_use_block("t1", "search_copom_documents", {"query": "por que manteve", "doc_type": "statement", "top_k": 2}),
            tool_use_block("t2", "get_indicators", {"start_month": "2025-11", "end_month": "2025-11", "indicators": ["ipca_12m"]}),
        ], "tool_use"),
        response([text_block("O Copom manteve a Selic em 15% (Copom #275, statement).")], "end_turn"),
    ])
    result = EconAgent(client, executor, settings).ask("Por que o Copom manteve a Selic?")

    assert result.answer.startswith("O Copom manteve")
    assert result.tools_used == {"search_copom_documents", "get_indicators"}
    assert result.iterations == 2
    assert result.input_tokens == 200

    # Both tool results go back in ONE user message, matched by tool_use_id.
    second_request = client.requests[1]["messages"]
    tool_results = second_request[-1]["content"]
    assert [r["tool_use_id"] for r in tool_results] == ["t1", "t2"]
    assert second_request[-2]["role"] == "assistant"


def test_agent_request_shape(executor, settings):
    client = FakeClient([response([text_block("ok")], "end_turn")])
    EconAgent(client, executor, settings).ask("Oi")
    req = client.requests[0]
    assert req["model"] == settings.chat_model
    assert req["tools"] == TOOL_DEFINITIONS
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": settings.effort}
    assert "tool_choice" not in req  # forced tool choice is not supported on this model
    for tool in TOOL_DEFINITIONS:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False


def test_tool_errors_are_sent_back_to_the_model(executor, settings):
    client = FakeClient([
        response([tool_use_block("t1", "get_indicators", {"start_month": "2025", "end_month": "2025-02", "indicators": ["selic_target"]})], "tool_use"),
        response([text_block("Corrigido.")], "end_turn"),
    ])
    result = EconAgent(client, executor, settings).ask("Selic?")
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "YYYY-MM" in sent["content"]
    assert result.tool_calls[0].is_error


def test_iteration_limit_guardrail(executor, settings):
    loop_forever = [
        response([tool_use_block(f"t{i}", "search_copom_documents", {"query": "x", "doc_type": "any", "top_k": 1})], "tool_use")
        for i in range(settings.max_iterations)
    ]
    result = EconAgent(FakeClient(loop_forever), executor, settings).ask("?")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == settings.max_iterations


def test_refusal_is_handled(executor, settings):
    result = EconAgent(FakeClient([response([], "refusal")]), executor, settings).ask("algo")
    assert result.stop_reason == "refusal"
    assert result.answer


def test_question_guardrails(executor, settings):
    agent = EconAgent(FakeClient([]), executor, settings)
    with pytest.raises(QuestionRejected):
        agent.ask("   ")
    with pytest.raises(QuestionRejected):
        agent.ask("x" * 2001)


def test_fallback_state_wraps_the_conversation(executor, settings):
    entered = []

    class State:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *exc):
            return False

    client = FakeClient([response([text_block("ok")], "end_turn")])
    EconAgent(client, executor, settings, fallback_state_factory=State).ask("Oi")
    assert entered == [True]
