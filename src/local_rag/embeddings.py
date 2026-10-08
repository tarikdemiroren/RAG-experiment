from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    max_tokens: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        try:
            # Use the cached copy without contacting Hugging Face; download only on first run.
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            self._model = SentenceTransformer(model_name)
        self.max_tokens: int = self._model.max_seq_length

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text, verbose=False)["input_ids"])
