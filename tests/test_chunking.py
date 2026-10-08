from local_rag.chunking import chunk_recipe, split_text
from local_rag.loaders import Recipe


def test_short_text_is_one_chunk():
    assert split_text("a\nb\nc", size=100, overlap=10) == ["a\nb\nc"]


def test_empty_text_has_no_chunks():
    assert split_text("\n\n", size=100, overlap=10) == []


def test_chunks_respect_size():
    lines = [f"line {i:02d} " + "x" * 30 for i in range(20)]
    chunks = split_text("\n".join(lines), size=120, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)
    assert "\n".join(chunks) == "\n".join(lines)


def test_overlap_repeats_trailing_lines():
    lines = ["aaaa", "bbbb", "cccc", "dddd", "eeee"]
    chunks = split_text("\n".join(lines), size=14, overlap=5)
    assert chunks == ["aaaa\nbbbb\ncccc", "cccc\ndddd\neeee"]


def test_overlap_is_dropped_when_it_would_not_fit():
    chunks = split_text("aaaa\nbbbbbbbbbbbbb", size=14, overlap=5)
    assert chunks == ["aaaa", "bbbbbbbbbbbbb"]


def test_long_line_is_split_by_sentences():
    line = "First sentence here. Second sentence here. Third one."
    chunks = split_text(line, size=25, overlap=0)
    assert chunks == ["First sentence here.", "Second sentence here.", "Third one."]


def test_sentence_longer_than_size_is_cut():
    chunks = split_text("x" * 50, size=20, overlap=0)
    assert chunks == ["x" * 20, "x" * 20, "x" * 10]


def test_chunk_recipe_prefixes_title_and_ids():
    recipe = Recipe(id="recipes/Pancake", title="Pancake", text="aaaa\nbbbb", files=())
    chunks = chunk_recipe(recipe, size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].id == "recipes/Pancake#0"
    assert chunks[0].text == "Pancake\naaaa\nbbbb"
    assert (chunks[0].recipe_id, chunks[0].title, chunks[0].index) == (
        "recipes/Pancake",
        "Pancake",
        0,
    )


def test_every_chunk_gets_the_title():
    recipe = Recipe(id="r", title="Croissants", text="\n".join(["x" * 40] * 6), files=())
    chunks = chunk_recipe(recipe, size=100, overlap=0)
    assert [c.id for c in chunks] == ["r#0", "r#1", "r#2"]
    assert all(c.text.startswith("Croissants\n") for c in chunks)
