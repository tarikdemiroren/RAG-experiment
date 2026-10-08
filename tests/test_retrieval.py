import math

import pytest
from conftest import FakeEmbedder

from local_rag.chunking import Chunk
from local_rag.retrieval import BM25, Retriever, is_relevant, reciprocal_rank_fusion, tokenize
from local_rag.store import Hit, VectorStore


def test_tokenize_lowercases_drops_stopwords_and_plurals():
    assert tokenize("How do I boil the Eggs?") == ["boil", "egg"]
    assert tokenize("Which recipes uses saffron?") == ["saffron"]
    assert tokenize("Tomatoes, berries, glass") == ["tomato", "berry", "glass"]
    assert tokenize("Turkish Çılbır 1/2 tsp") == ["turkish", "çılbır", "1", "2", "tsp"]


def test_bm25_prefers_rare_terms():
    bm25 = BM25([["egg", "flour"], ["egg", "saffron"], ["egg", "milk"]])
    scores = bm25.scores(["egg", "saffron"])
    assert scores[1] > scores[0] == scores[2] > 0


def test_bm25_term_frequency_saturates():
    docs = [["beef"] + ["x"] * 9, ["beef"] * 2 + ["x"] * 8, ["beef"] * 10, ["bun"] * 10]
    one, two, ten, none = BM25(docs).scores(["beef"])
    assert none == 0
    assert one < two < ten < 10 * one


def test_bm25_penalizes_long_documents():
    scores = BM25([["saffron", "rice"], ["saffron"] + ["filler"] * 20]).scores(["saffron"])
    assert scores[0] > scores[1] > 0


def test_bm25_unknown_terms_score_zero():
    assert BM25([["egg"]]).scores(["tire"]) == [0.0]


def test_bm25_idf_formula():
    bm25 = BM25([["a"], ["b"], ["b"]])
    assert bm25.idf["a"] == pytest.approx(math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert sorted(fused, key=fused.get, reverse=True) == ["a", "c", "b"]


def test_is_relevant():
    near = Hit("a", "A", "a", "x", distance=0.3)
    far = Hit("b", "B", "b", "x", distance=0.9, keyword_coverage=0.49)
    far_match = Hit("c", "C", "c", "x", distance=0.9, keyword_coverage=0.5)
    assert is_relevant(near, 0.6)
    assert not is_relevant(far, 0.6)
    assert is_relevant(far_match, 0.6)


@pytest.fixture
def retriever(tmp_path):
    texts = {
        "pancake": "Pancake\nflour milk egg fry pancakes",
        "hummus": "Hummus\nblend chickpeas lemon garlic",
        "pilaf": "Rice pilaf\nrice butter onion stock saffron",
        "salad": "Greek salad\ntomato cucumber olives feta lemon",
    }
    chunks = [Chunk(f"{k}#0", k, v.split("\n")[0], 0, v) for k, v in texts.items()]
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path / "db", "recipes")
    store.rebuild(chunks, embedder.embed([c.text for c in chunks]))
    return Retriever(store, embedder)


def test_search_scores_every_hit(retriever):
    hits = retriever.search("saffron rice", k=4)
    top = hits[0]
    assert top.id == "pilaf#0"
    assert top.keyword_coverage == 1.0
    assert top.keyword_score > 0
    assert top.score == pytest.approx(2 / 61)
    assert all(h.keyword_coverage == 0 for h in hits[1:])


def test_keyword_coverage_counts_share_of_query_terms(retriever):
    hits = {h.id: h for h in retriever.search("saffron lemon onion", k=4)}
    assert hits["pilaf#0"].keyword_coverage == pytest.approx(2 / 3)
    assert hits["hummus#0"].keyword_coverage == pytest.approx(1 / 3)
    assert hits["pancake#0"].keyword_coverage == 0


def test_keyword_mode_returns_only_matching_chunks(retriever):
    assert [h.id for h in retriever.search("lemon", k=4, mode="keyword")] == [
        "hummus#0",
        "salad#0",
    ]


def test_vector_mode_ranks_by_distance(retriever):
    hits = retriever.search("lemon", k=4, mode="vector")
    assert len(hits) == 4
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_max_distance_filters_before_truncating(retriever):
    hits = retriever.search("saffron", k=4, max_distance=0.0)
    assert [h.id for h in hits] == ["pilaf#0"]


def test_half_of_the_keywords_is_enough(retriever):
    assert [h.id for h in retriever.search("saffron tea", k=4, max_distance=0.0)] == ["pilaf#0"]
    assert retriever.search("saffron tea coffee", k=4, max_distance=0.0) == []


def test_no_relevant_hits(retriever):
    assert retriever.search("car tire", k=4, max_distance=0.0) == []
