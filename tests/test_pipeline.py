import pytest
from conftest import FakeEmbedder

from local_rag.config import Settings
from local_rag.pipeline import NothingToIngestError, ingest
from local_rag.store import VectorStore


def test_ingest(tmp_path, make_docx):
    make_docx(tmp_path / "res" / "Pancake" / "Recipe.docx", "flour milk egg")
    make_docx(tmp_path / "res" / "Pancake" / "Photo.docx")
    settings = Settings.from_env({"RAG_RESOURCES_DIR": str(tmp_path / "res")})
    store = VectorStore(tmp_path / "db", "recipes")

    result = ingest(settings, FakeEmbedder(max_tokens=3), store)

    assert (result.recipes, result.chunks) == (1, 1)
    assert result.skipped == [("Pancake/Photo.docx", "no text")]
    assert result.truncated == [("Pancake#0", 4)]
    assert store.count() == 1


def test_ingest_nothing(tmp_path):
    settings = Settings.from_env({"RAG_RESOURCES_DIR": str(tmp_path)})
    with pytest.raises(NothingToIngestError, match="no documents with text"):
        ingest(settings, FakeEmbedder(), VectorStore(tmp_path / "db", "recipes"))
