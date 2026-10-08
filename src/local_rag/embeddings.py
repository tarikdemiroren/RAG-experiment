from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    @property
    def max_tokens(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    """Loads the model on first use."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        try:
            # Use the cached copy without contacting Hugging Face; download only on first run.
            return SentenceTransformer(self.model_name, local_files_only=True)
        except OSError:
            return SentenceTransformer(self.model_name)

    @property
    def max_tokens(self) -> int:
        return self._model.max_seq_length

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text, verbose=False)["input_ids"])
