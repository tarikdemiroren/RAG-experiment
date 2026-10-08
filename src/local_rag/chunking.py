from __future__ import annotations

import re
from dataclasses import dataclass

from local_rag.loaders import Recipe

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    recipe_id: str
    title: str
    index: int
    text: str


def chunk_recipe(recipe: Recipe, size: int, overlap: int) -> list[Chunk]:
    """Split a recipe into chunks, each starting with the recipe title."""
    return [
        Chunk(
            id=f"{recipe.id}#{i}",
            recipe_id=recipe.id,
            title=recipe.title,
            index=i,
            text=f"{recipe.title}\n{body}",
        )
        for i, body in enumerate(split_text(recipe.text, size, overlap))
    ]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack lines into chunks of at most ``size`` chars, repeating up to ``overlap`` chars of
    trailing lines at the start of the next chunk. Lines longer than ``size`` are split into
    sentences, and sentences longer than ``size`` are cut."""
    units = [unit for line in text.splitlines() for unit in _split_long(line.strip(), size)]
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _length([*current, unit]) > size:
            chunks.append("\n".join(current))
            current = _tail(current, overlap)
            while current and _length([*current, unit]) > size:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_long(line: str, size: int) -> list[str]:
    if len(line) <= size:
        return [line] if line else []
    pieces = []
    for sentence in _SENTENCE_END.split(line):
        pieces.extend(sentence[i : i + size] for i in range(0, len(sentence), size))
    return pieces


def _length(lines: list[str]) -> int:
    return sum(map(len, lines)) + len(lines) - 1


def _tail(lines: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    for line in reversed(lines):
        if _length([line, *tail]) > overlap:
            break
        tail.insert(0, line)
    return tail
