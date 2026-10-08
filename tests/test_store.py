from pathlib import Path

import pytest
from conftest import FakeEmbedder

from local_rag.chunking import Chunk
from local_rag.store import IndexMissingError, VectorStore


def _chunk(recipe_id, text, index=0):
    return Chunk(f"{recipe_id}#{index}", recipe_id, recipe_id.title(), index, text)


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "db", "recipes")


def test_query_returns_nearest_first(store):
    embedder = FakeEmbedder()
    chunks = [
        _chunk("hummus", "blend chickpeas lemon garlic tahini"),
        _chunk("pancake", "flour milk egg fry pancakes"),
        _chunk("salad", "tomato cucumber olives feta"),
    ]
    store.rebuild(chunks, embedder.embed([c.text for c in chunks]))

    hits = store.query(embedder.embed(["fry pancakes with milk"])[0], k=2)

    assert len(hits) == 2
    top = hits[0]
    assert top.id == "pancake#0"
    assert (top.title, top.source, top.text) == ("Pancake", "pancake", chunks[1].text)
    assert 0 <= top.distance < hits[1].distance


def test_identical_vector_has_zero_cosine_distance(store):
    store.rebuild([_chunk("a", "x")], [[1.0, 0.0]])
    assert store.query([2.0, 0.0], k=1)[0].distance == pytest.approx(0.0, abs=1e-6)


def test_rebuild_replaces_previous_index(store):
    store.rebuild([_chunk("old", "x"), _chunk("old", "y", 1)], [[1.0, 0.0], [0.0, 1.0]])
    store.rebuild([_chunk("new", "z")], [[1.0, 0.0]])
    assert store.count() == 1
    assert [h.id for h in store.query([1.0, 0.0], k=5)] == ["new#0"]


def test_index_persists_on_disk(tmp_path):
    VectorStore(tmp_path / "db", "recipes").rebuild([_chunk("a", "x")], [[1.0, 0.0]])
    assert VectorStore(tmp_path / "db", "recipes").count() == 1


def test_query_without_index_raises(store):
    assert store.count() == 0
    with pytest.raises(IndexMissingError, match="local-rag ingest"):
        store.query([1.0, 0.0], k=1)


def test_relative_path_follows_current_directory(tmp_path, monkeypatch):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    monkeypatch.chdir(tmp_path / "one")
    VectorStore(Path("db"), "recipes").rebuild([_chunk("a", "x")], [[1.0, 0.0]])
    monkeypatch.chdir(tmp_path / "two")
    assert VectorStore(Path("db"), "recipes").count() == 0
