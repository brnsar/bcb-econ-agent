"""Text embedders: Amazon Titan on Bedrock, plus an offline hashing embedder for tests."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with L2-normalized rows."""
        ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", text)


class HashingEmbedder:
    """Deterministic bag-of-words + bigram hashing embedder. No network, no model.

    Good enough for tests and a rough offline mode; use Titan for real retrieval quality.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = [t for t in _tokens(text) if len(t) > 2]
            features = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for feature in features:
                bucket, sign = self._index(feature)
                matrix[row, bucket] += sign
        return _normalize_rows(matrix)


class TitanEmbedder:
    """Amazon Titan Text Embeddings v2 through the Bedrock runtime API."""

    def __init__(self, model_id: str, region: str, dim: int = 1024, client=None) -> None:
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self._model_id = model_id
        self.dim = dim
        self.name = f"{model_id}:{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
            response = self._client.invoke_model(
                modelId=self._model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            vectors.append(payload["embedding"])
        return _normalize_rows(np.asarray(vectors, dtype=np.float32))


def make_embedder(settings) -> Embedder:
    if settings.offline_embeddings:
        return HashingEmbedder()
    return TitanEmbedder(settings.embedding_model, settings.aws_region, settings.embedding_dim)
