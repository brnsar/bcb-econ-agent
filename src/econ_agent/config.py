"""Settings loaded from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    aws_region: str = "us-east-1"
    chat_model: str = "anthropic.claude-opus-5-5"
    # Client-side refusal fallback (Bedrock has no server-side `fallbacks`). Empty disables it.
    fallback_model: str = "anthropic.claude-opus-5"
    judge_model: str = "anthropic.claude-opus-5-5"
    effort: str = "medium"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dim: int = 1024
    # Use a local hashing embedder instead of Titan (no AWS needed; lower retrieval quality).
    offline_embeddings: bool = False
    data_dir: Path = Path("data")
    first_meeting: int = 266
    indicators_start: str = "2015-01"
    max_iterations: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            aws_region=os.getenv("AWS_REGION", d.aws_region),
            chat_model=os.getenv("ECON_AGENT_CHAT_MODEL", d.chat_model),
            fallback_model=os.getenv("ECON_AGENT_FALLBACK_MODEL", d.fallback_model),
            judge_model=os.getenv("ECON_AGENT_JUDGE_MODEL", d.judge_model),
            effort=os.getenv("ECON_AGENT_EFFORT", d.effort),
            embedding_model=os.getenv("ECON_AGENT_EMBEDDING_MODEL", d.embedding_model),
            embedding_dim=int(os.getenv("ECON_AGENT_EMBEDDING_DIM", d.embedding_dim)),
            offline_embeddings=_bool(os.getenv("ECON_AGENT_OFFLINE_EMBEDDINGS"), d.offline_embeddings),
            data_dir=Path(os.getenv("ECON_AGENT_DATA_DIR", str(d.data_dir))),
            first_meeting=int(os.getenv("ECON_AGENT_FIRST_MEETING", d.first_meeting)),
            indicators_start=os.getenv("ECON_AGENT_INDICATORS_START", d.indicators_start),
            max_iterations=int(os.getenv("ECON_AGENT_MAX_ITERATIONS", d.max_iterations)),
        )

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def indicators_db(self) -> Path:
        return self.data_dir / "indicators.db"
