"""Minimal persistent vector store (numpy + JSON) with cosine similarity search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class SearchHit:
    score: float
    record: dict


class VectorStore:
    def __init__(self, embedder_name: str, dim: int) -> None:
        self.embedder_name = embedder_name
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.records: list[dict] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, vectors: np.ndarray, records: list[dict]) -> None:
        if vectors.shape != (len(records), self.dim):
            raise ValueError(f"expected vectors of shape ({len(records)}, {self.dim}), got {vectors.shape}")
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.records.extend(records)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        where: Callable[[dict], bool] | None = None,
    ) -> list[SearchHit]:
        if not self.records:
            return []
        scores = self.vectors @ query_vector.astype(np.float32).reshape(-1)
        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for idx in order:
            record = self.records[int(idx)]
            if where is not None and not where(record):
                continue
            hits.append(SearchHit(score=float(scores[idx]), record=record))
            if len(hits) >= top_k:
                break
        return hits

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "records.json").write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
        meta = {"embedder": self.embedder_name, "dim": self.dim, "count": len(self.records)}
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No index in {directory}. Run `econ-agent ingest` first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(meta["embedder"], meta["dim"])
        store.vectors = np.load(directory / "vectors.npy")
        store.records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        return store
