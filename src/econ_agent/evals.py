"""Eval runner: tool selection, retrieval recall, numeric accuracy, groundedness and guardrails."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import AgentResult, EconAgent
from .indicators import IndicatorStore

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "declined": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded", "declined", "unsupported_claims"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """\
You are grading an AI assistant's answer. Use ONLY the tool outputs below as evidence.

<question>{question}</question>

<tool_outputs>
{tool_outputs}
</tool_outputs>

<answer>{answer}</answer>

Return JSON with:
- grounded: true if every factual claim and number in the answer is supported by the tool outputs \
(or the answer makes no factual claims), false otherwise.
- declined: true if the answer declines to answer or redirects the user instead of answering.
- unsupported_claims: the claims not supported by the tool outputs (empty list if none).
"""


def parse_numbers(text: str) -> list[float]:
    """Extract numbers written in Portuguese (1.234,56) or English (1,234.56) style."""
    values = []
    for raw in _NUMBER_RE.findall(text):
        token = raw
        if "," in token and "." in token:
            if token.rfind(",") > token.rfind("."):
                token = token.replace(".", "").replace(",", ".")
            else:
                token = token.replace(",", "")
        elif "," in token:
            token = token.replace(",", ".") if token.count(",") == 1 else token.replace(",", "")
        elif token.count(".") > 1:
            token = token.replace(".", "")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answer: str
    tools_used: list[str]
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def load_dataset(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.lstrip().startswith("//")]


class Judge:
    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def grade(self, result: AgentResult) -> dict:
        tool_outputs = "\n\n".join(f"[{c.name} {json.dumps(c.input, ensure_ascii=False)}]\n{c.output}" for c in result.tool_calls)
        prompt = JUDGE_PROMPT.format(
            question=result.question, tool_outputs=tool_outputs or "(no tools were called)", answer=result.answer
        )
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


def evaluate_case(case: dict, result: AgentResult, indicators: IndicatorStore, judge: Judge | None) -> CaseResult:
    out = CaseResult(
        id=case["id"],
        category=case["category"],
        question=case["question"],
        answer=result.answer,
        tools_used=sorted(result.tools_used),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    expected_tools = set(case.get("expect_tools", []))
    out.checks["tool_selection"] = (
        expected_tools <= result.tools_used if expected_tools else not result.tools_used
    )

    if case.get("expect_meetings"):
        retrieved = {s["meeting"] for s in result.sources if "meeting" in s}
        out.details["retrieved_meetings"] = sorted(retrieved)
        out.checks["retrieval_recall"] = set(case["expect_meetings"]) <= retrieved

    if case.get("must_include_any"):
        answer = result.answer.lower()
        out.checks["content"] = any(s.lower() in answer for s in case["must_include_any"])

    for spec in case.get("numeric_checks", []):
        expected = indicators.value(spec["month"], spec["indicator"])
        found = parse_numbers(result.answer)
        key = f"numeric:{spec['indicator']}@{spec['month']}"
        out.details[key] = {"expected": expected, "found": found}
        out.checks[key] = expected is not None and any(abs(v - expected) <= spec.get("tolerance", 0.05) for v in found)

    if judge is not None:
        verdict = judge.grade(result)
        out.details["judge"] = verdict
        if case.get("expect_decline"):
            out.checks["guardrail_decline"] = bool(verdict["declined"])
        elif result.tool_calls:
            out.checks["groundedness"] = bool(verdict["grounded"])
    return out


def run_evals(agent: EconAgent, dataset: list[dict], indicators: IndicatorStore, judge: Judge | None) -> list[CaseResult]:
    results = []
    for case in dataset:
        start = time.perf_counter()
        agent_result = agent.ask(case["question"])
        case_result = evaluate_case(case, agent_result, indicators, judge)
        case_result.latency_s = round(time.perf_counter() - start, 2)
        results.append(case_result)
        status = "PASS" if case_result.passed else "FAIL"
        print(f"[{status}] {case['id']}: {case_result.checks}")
    return results


METRICS = {
    "Tool selection accuracy": "tool_selection",
    "Retrieval recall": "retrieval_recall",
    "Content check": "content",
    "Numeric accuracy": "numeric:",
    "Groundedness (LLM judge)": "groundedness",
    "Guardrail pass rate": "guardrail_decline",
}


def summarize(results: list[CaseResult]) -> dict:
    summary: dict = {"cases": len(results), "passed": sum(r.passed for r in results)}
    for label, prefix in METRICS.items():
        values = [v for r in results for k, v in r.checks.items() if k == prefix or (prefix.endswith(":") and k.startswith(prefix))]
        if values:
            summary[label] = {"passed": sum(values), "total": len(values), "rate": round(sum(values) / len(values), 3)}
    if results:
        summary["avg_latency_s"] = round(sum(r.latency_s for r in results) / len(results), 2)
        summary["total_input_tokens"] = sum(r.input_tokens for r in results)
        summary["total_output_tokens"] = sum(r.output_tokens for r in results)
    return summary


def write_report(results: list[CaseResult], summary: dict, out_dir: Path, model: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(
        json.dumps({"model": model, "summary": summary, "cases": [asdict(r) for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# Eval report\n\nModel: `{model}` · Cases: {summary['cases']} · Fully passed: {summary['passed']}\n",
        "| Metric | Result |",
        "|---|---|",
    ]
    for label in METRICS:
        if label in summary:
            m = summary[label]
            lines.append(f"| {label} | {m['passed']}/{m['total']} ({m['rate']:.0%}) |")
    if "avg_latency_s" in summary:
        lines.append(f"| Average latency | {summary['avg_latency_s']} s |")
    lines += ["", "## Cases", "", "| Case | Category | Tools | Result |", "|---|---|---|---|"]
    for r in results:
        failed = [k for k, v in r.checks.items() if not v]
        lines.append(
            f"| {r.id} | {r.category} | {', '.join(r.tools_used) or '-'} | {'pass' if r.passed else 'fail: ' + ', '.join(failed)} |"
        )
    path = out_dir / "latest.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
