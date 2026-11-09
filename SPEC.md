# SPEC — BCB Economic Agent

Spec written before the code. Each requirement has an ID that the tests and evals refer to.

## 1. Problem

Analysts who follow Brazilian monetary policy jump between two sources:

- **Unstructured text**: Copom statements and meeting minutes (why the Central Bank decided what it decided).
- **Structured numbers**: Selic target, IPCA inflation and the USD/BRL exchange rate (what actually happened).

Answering a question like *"Why did Copom stop cutting rates, and what was 12-month inflation at the time?"* needs both. The agent must decide which source to use, combine them, and cite where every claim came from.

## 2. Users and scope

- **User**: an analyst asking questions in Portuguese or English.
- **In scope**: Copom decisions and reasoning (meetings 266+), monthly Selic, IPCA (monthly and 12-month), USD/BRL (monthly average and close), real interest rate.
- **Out of scope**: personalized investment advice, forecasts presented as facts, topics unrelated to the Brazilian economy.

## 3. Functional requirements

| ID | Requirement |
|---|---|
| F1 | Ingest Copom statements and minutes from the public BCB API, convert HTML to text, split into chunks and embed them with Amazon Titan Text Embeddings v2. |
| F2 | Ingest SGS series 1 (USD/BRL), 432 (Selic target) and 433 (IPCA) and build a monthly table with derived indicators (12-month IPCA, real rate). |
| F3 | Expose two tools to the model: `search_copom_documents` (RAG) and `get_indicators` (structured data). |
| F4 | The agent (Claude on Amazon Bedrock) runs a tool-use loop, decides which tools to call, and answers in the user's language. |
| F5 | Every factual claim in the answer cites its source: Copom meeting number and document type, or the indicator table with the month. |
| F6 | Numbers in the answer come only from tool results, never from model memory. |
| F7 | CLI with `ingest`, `ask` and `eval` commands. |

## 4. Guardrails

| ID | Guardrail |
|---|---|
| G1 | Questions over 2,000 characters are rejected before any model call. |
| G2 | Tool inputs are validated in code (month format, date range, allowed indicators, `top_k` limits). Invalid input returns a tool error so the model can recover, never an exception. |
| G3 | The agent loop stops after a maximum number of iterations (default 8). |
| G4 | Off-topic questions and requests for personalized investment advice are declined without calling tools. |
| G5 | Refusals (`stop_reason == "refusal"`) are handled explicitly; on Bedrock a client-side fallback middleware retries on a fallback model. |
| G6 | `get_indicators` returns at most 120 rows. |

## 5. Quality: evals

A versioned dataset (`evals/dataset.jsonl`) is run against the real model. Metrics:

| Metric | Definition |
|---|---|
| Tool selection accuracy | The expected tools were called, and no tool was called on cases where none is expected. |
| Retrieval recall | For RAG cases, the expected Copom meeting appears in the documents the agent retrieved. |
| Numeric accuracy | For numeric cases, the answer contains the value from the indicator table (within tolerance). Expected values are computed from the local data at eval time, so they stay correct when data is refreshed. |
| Groundedness (LLM judge) | A judge model checks that every claim in the answer is supported by the tool outputs. |
| Guardrail pass rate | Off-topic and investment-advice cases are declined without tool calls. |

## 6. Non-functional

- Unit tests run offline (no AWS) with a scripted fake model client and a hashing embedder. CI runs them on every push.
- Configuration through environment variables only. No credentials in the repository.
- Model IDs, region and effort are configurable.
