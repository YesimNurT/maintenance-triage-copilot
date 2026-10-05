"""Vector index for maintenance cases: in-memory for evaluation and tests, Pinecone when deployed.

Both classes expose ``upsert(ids, vectors, metadata)`` and ``query(vector, k)``, so the
search and the agent do not know which one they talk to.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mtc.config import Settings


@dataclass(frozen=True)
class CaseHit:
    case_id: str
    label: str
    similarity: float


class CaseIndex(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None: ...

    def query(self, vector: np.ndarray, k: int) -> list[CaseHit]: ...


class LocalIndex:
    """Exact cosine search over a small matrix held in memory."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metadata: list[dict] = []
        self._unit = np.empty((0, 0), dtype=np.float32)

    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
        unit = _unit_rows(np.asarray(vectors, dtype=np.float32))
        self._unit = unit if not self.ids else np.vstack([self._unit, unit])
        self.ids += list(ids)
        self.metadata += list(metadata)

    def query(self, vector: np.ndarray, k: int) -> list[CaseHit]:
        if not self.ids:
            return []
        similarity = self._unit @ _unit_rows(np.asarray(vector, dtype=np.float32)[None, :])[0]
        order = np.argsort(-similarity, kind="stable")[:k]
        return [
            CaseHit(self.ids[i], str(self.metadata[i].get("label", "")), float(similarity[i]))
            for i in order
        ]


class PineconeIndex:
    """Thin wrapper around a Pinecone index handle (anything with ``upsert`` and ``query``)."""

    BATCH = 100

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
        records = [
            {"id": i, "values": [float(x) for x in v], "metadata": m}
            for i, v, m in zip(ids, np.asarray(vectors), metadata, strict=True)
        ]
        for start in range(0, len(records), self.BATCH):
            self.handle.upsert(vectors=records[start : start + self.BATCH])

    def query(self, vector: np.ndarray, k: int) -> list[CaseHit]:
        response = self.handle.query(
            vector=[float(x) for x in np.asarray(vector)], top_k=k, include_metadata=True
        )
        matches = response["matches"] if isinstance(response, dict) else response.matches
        return [
            CaseHit(_get(m, "id"), str((_get(m, "metadata") or {}).get("label", "")),
                    float(_get(m, "score")))
            for m in matches
        ]


def make_index(settings: Settings, dimension: int) -> CaseIndex:
    """Pinecone when an API key is configured, otherwise the local index."""
    if not settings.pinecone_api_key:
        return LocalIndex()
    from pinecone import Pinecone, ServerlessSpec  # imported lazily: optional dependency

    client = Pinecone(api_key=settings.pinecone_api_key)
    if not client.has_index(settings.pinecone_index_name):
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return PineconeIndex(client.Index(settings.pinecone_index_name))


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to length 1; an all-zero row stays zero and so matches nothing."""
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norm > 0, norm, 1.0)


def _get(match: Any, key: str) -> Any:
    return match[key] if isinstance(match, dict) else getattr(match, key)
