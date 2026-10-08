from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from typing import Literal

from local_rag.embeddings import Embedder
from local_rag.store import Hit, VectorStore

Mode = Literal["hybrid", "vector", "keyword"]
MODES: tuple[Mode, ...] = ("hybrid", "vector", "keyword")
RRF_K = 60
MIN_KEYWORD_COVERAGE = 0.5

# Includes question words ("use", "make", "recipe") that would otherwise be required keywords.
STOPWORDS = frozenset(
    "a all an and any are as at be by can contain could do does for from get give had has have "
    "how i in include is it me my need of on or should some the there to use used using what "
    "which with would you your make cook recipe dish".split()
)
_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    words = (word for word in _WORD.findall(text.lower()) if word not in STOPWORDS)
    return [term for term in map(_stem, words) if term not in STOPWORDS]


def _stem(word: str) -> str:
    # Crude plural folding: eggs -> egg, tomatoes -> tomato, berries -> berry.
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_length = sum(self.lengths) / len(documents) if documents else 0.0
        n = len(documents)
        doc_freq = Counter(term for doc in documents for term in set(doc))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in doc_freq.items()}

    def scores(self, query: list[str]) -> list[float]:
        terms = sorted(t for t in set(query) if t in self.idf)
        results = []
        for counts, length in zip(self.term_counts, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            results.append(
                sum(
                    self.idf[t] * counts[t] * (self.k1 + 1) / (counts[t] + norm)
                    for t in terms
                    if t in counts
                )
            )
        return results


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, 1):
            scores[id_] = scores.get(id_, 0.0) + 1 / (k + rank)
    return scores


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def search(
        self, query: str, k: int, mode: Mode = "hybrid", max_distance: float | None = None
    ) -> list[Hit]:
        """Top ``k`` chunks; with ``max_distance``, only those that pass :func:`is_relevant`."""
        hits = self.store.query_all(self.embedder.embed([query])[0])
        terms = set(tokenize(query))
        documents = [tokenize(h.text) for h in hits]
        keyword = BM25(documents).scores(list(terms))
        hits = [
            replace(h, keyword_score=score, keyword_coverage=_coverage(terms, doc))
            for h, score, doc in zip(hits, keyword, documents, strict=True)
        ]

        by_vector = [h.id for h in hits]
        by_keyword = [
            h.id for h in sorted(hits, key=lambda h: -h.keyword_score) if h.keyword_score > 0
        ]
        rankings = {
            "hybrid": [by_vector, by_keyword],
            "vector": [by_vector],
            "keyword": [by_keyword],
        }[mode]
        fused = reciprocal_rank_fusion(rankings)
        ranked = sorted(
            (replace(h, score=fused[h.id]) for h in hits if h.id in fused),
            key=lambda h: -h.score,
        )
        if max_distance is not None:
            ranked = [h for h in ranked if is_relevant(h, max_distance)]
        return ranked[:k]


def is_relevant(hit: Hit, max_distance: float) -> bool:
    """Close in meaning, or contains at least half of the query's keywords."""
    return hit.distance <= max_distance or hit.keyword_coverage >= MIN_KEYWORD_COVERAGE


def _coverage(terms: set[str], document: list[str]) -> float:
    return len(terms.intersection(document)) / len(terms) if terms else 0.0
