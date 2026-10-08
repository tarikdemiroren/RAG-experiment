from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from local_rag.chunking import Chunk


class IndexMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    id: str
    title: str
    source: str
    text: str
    distance: float
    keyword_score: float = 0.0
    keyword_coverage: float = 0.0
    score: float = 0.0


class VectorStore:
    def __init__(self, path: Path, collection: str):
        self._client = chromadb.PersistentClient(
            path=str(path.resolve()), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._name = collection

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        try:
            self._client.delete_collection(self._name)
        except NotFoundError:
            pass
        collection = self._client.create_collection(
            self._name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
        step = self._client.get_max_batch_size()
        for start in range(0, len(chunks), step):
            batch = chunks[start : start + step]
            collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings[start : start + step],
                documents=[c.text for c in batch],
                metadatas=[
                    {"title": c.title, "source": c.recipe_id, "index": c.index} for c in batch
                ],
            )

    def query(self, embedding: list[float], k: int) -> list[Hit]:
        return self._query(self._collection(), embedding, k)

    def query_all(self, embedding: list[float]) -> list[Hit]:
        """Every chunk, nearest first."""
        collection = self._collection()
        return self._query(collection, embedding, collection.count())

    def _collection(self):
        try:
            return self._client.get_collection(self._name, embedding_function=None)
        except NotFoundError:
            raise IndexMissingError("no index found; run `local-rag ingest` first") from None

    @staticmethod
    def _query(collection, embedding: list[float], k: int) -> list[Hit]:
        result = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(id=id_, title=meta["title"], source=meta["source"], text=doc, distance=dist)
            for id_, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def count(self) -> int:
        try:
            return self._client.get_collection(self._name, embedding_function=None).count()
        except NotFoundError:
            return 0
