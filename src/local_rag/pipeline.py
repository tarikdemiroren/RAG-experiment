from __future__ import annotations

from dataclasses import dataclass

from local_rag.chunking import chunk_recipe
from local_rag.config import Settings
from local_rag.embeddings import Embedder, SentenceTransformerEmbedder
from local_rag.loaders import load_recipes
from local_rag.retrieval import Retriever
from local_rag.store import VectorStore


class NothingToIngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestResult:
    recipes: int
    chunks: int
    skipped: list[tuple[str, str]]
    truncated: list[tuple[str, int]]


def ingest(settings: Settings, embedder: Embedder, store: VectorStore) -> IngestResult:
    """Rebuild the index from the resources folder."""
    report = load_recipes(settings.resources_dir)
    chunks = [
        chunk
        for recipe in report.recipes
        for chunk in chunk_recipe(recipe, settings.chunk_size, settings.chunk_overlap)
    ]
    if not chunks:
        raise NothingToIngestError(f"no documents with text under {settings.resources_dir}")
    tokens = [(c.id, embedder.count_tokens(c.text)) for c in chunks]
    store.rebuild(chunks, embedder.embed([c.text for c in chunks]))
    return IngestResult(
        recipes=len(report.recipes),
        chunks=len(chunks),
        skipped=report.skipped,
        truncated=[(id_, n) for id_, n in tokens if n > embedder.max_tokens],
    )


def make_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(settings.embedding_model)


def make_store(settings: Settings) -> VectorStore:
    return VectorStore(settings.db_dir, settings.collection)


def make_retriever(settings: Settings) -> Retriever:
    return Retriever(make_store(settings), make_embedder(settings))
