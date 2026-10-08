import math

import pytest

from local_rag.config import DEFAULTS


@pytest.fixture(scope="module")
def embedder():
    from sentence_transformers import SentenceTransformer

    from local_rag.embeddings import SentenceTransformerEmbedder

    name = DEFAULTS["RAG_EMBEDDING_MODEL"]
    try:
        SentenceTransformer(name, local_files_only=True)
    except OSError:
        pytest.skip("embedding model not downloaded")
    return SentenceTransformerEmbedder(name)


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_normalized_384_dims(embedder):
    (v,) = embedder.embed(["boiled eggs"])
    assert len(v) == 384
    assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-5)


def test_similar_texts_are_closer(embedder):
    q, eggs, car = embedder.embed(
        ["how to boil an egg", "Simmer the eggs for 8 minutes", "car tire"]
    )
    assert _cosine(q, eggs) > _cosine(q, car)


def test_token_limit(embedder):
    assert embedder.max_tokens == 256
    assert embedder.count_tokens("word " * 300) > 256
