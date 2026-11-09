"""Command-line interface: econ-agent ingest | ask | eval."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader so the project needs no extra dependency. Existing env vars win."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_agent(settings: Settings):
    from .agent import EconAgent
    from .embeddings import make_embedder
    from .indicators import IndicatorStore
    from .llm import make_client, new_fallback_state
    from .tools import ToolExecutor
    from .vector_store import VectorStore

    index = VectorStore.load(settings.index_dir)
    embedder = make_embedder(settings)
    if index.embedder_name != embedder.name:
        sys.exit(
            f"The index was built with '{index.embedder_name}' but the current embedder is '{embedder.name}'. "
            "Run `econ-agent ingest` again with the same settings."
        )
    indicators = IndicatorStore(settings.indicators_db)
    if not indicators.exists():
        sys.exit("No indicator table found. Run `econ-agent ingest` first.")
    client = make_client(settings)
    state_factory = new_fallback_state if settings.fallback_model else None
    agent = EconAgent(client, ToolExecutor(index, embedder, indicators), settings, fallback_state_factory=state_factory)
    return agent, client, indicators


def cmd_ingest(settings: Settings, args) -> None:
    from .embeddings import make_embedder
    from .ingest import ingest_documents, ingest_indicators

    if not args.skip_indicators:
        store = ingest_indicators(settings)
        lo, hi = store.month_range()
        print(f"Indicators: {lo} to {hi} -> {settings.indicators_db}")
    if not args.skip_documents:
        index = ingest_documents(settings, make_embedder(settings))
        print(f"Copom index: {len(index)} chunks ({index.embedder_name}) -> {settings.index_dir}")


def cmd_ask(settings: Settings, args) -> None:
    from .agent import QuestionRejected

    agent, _, _ = _build_agent(settings)
    try:
        result = agent.ask(args.question)
    except QuestionRejected as exc:
        sys.exit(str(exc))
    print(result.answer)
    if args.verbose:
        print("\n--- trace ---")
        for call in result.tool_calls:
            print(f"{call.name}({json.dumps(call.input, ensure_ascii=False)}) error={call.is_error}")
        print(f"iterations={result.iterations} stop={result.stop_reason} "
              f"tokens_in={result.input_tokens} tokens_out={result.output_tokens}")


def cmd_eval(settings: Settings, args) -> None:
    from .evals import Judge, load_dataset, run_evals, summarize, write_report

    agent, client, indicators = _build_agent(settings)
    dataset = load_dataset(Path(args.dataset))
    if args.only:
        dataset = [c for c in dataset if c["id"] in set(args.only)]
    judge = None if args.no_judge else Judge(client, settings.judge_model)
    results = run_evals(agent, dataset, indicators, judge)
    summary = summarize(results)
    path = write_report(results, summary, Path(args.out), settings.chat_model)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {path}")


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="econ-agent", description=__doc__)
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download Copom documents and SGS series, build the local index")
    p_ingest.add_argument("--skip-documents", action="store_true")
    p_ingest.add_argument("--skip-indicators", action="store_true")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-v", "--verbose", action="store_true", help="Show the tool-call trace")

    p_eval = sub.add_parser("eval", help="Run the eval dataset against the real model")
    p_eval.add_argument("--dataset", default="evals/dataset.jsonl")
    p_eval.add_argument("--out", default="evals/results")
    p_eval.add_argument("--only", nargs="*", help="Run only these case ids")
    p_eval.add_argument("--no-judge", action="store_true", help="Skip the LLM-judge checks")

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval}[args.command](settings, args)


if __name__ == "__main__":
    main()
