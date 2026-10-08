from local_rag import loaders
from local_rag.loaders import clean_line, clean_title, load_recipes, read_docx


def test_each_folder_is_one_recipe(tmp_path, make_docx):
    make_docx(tmp_path / "recipes" / "Pancake" / "Ingredients.docx", "1 cup flour", "1 egg")
    make_docx(tmp_path / "recipes" / "Pancake" / "Recipe.docx", "Mix and fry.")
    make_docx(tmp_path / "recipes" / "Hummus" / "Untitled.docx", "Blend chickpeas.")

    report = load_recipes(tmp_path)

    assert [(r.id, r.title) for r in report.recipes] == [
        ("recipes/Hummus", "Hummus"),
        ("recipes/Pancake", "Pancake"),
    ]
    pancake = report.recipes[1]
    assert pancake.text == "1 cup flour\n1 egg\nMix and fry."
    assert pancake.files == ("recipes/Pancake/Ingredients.docx", "recipes/Pancake/Recipe.docx")
    assert report.skipped == []


def test_empty_folders_and_other_files_are_ignored(tmp_path, make_docx):
    (tmp_path / "Untitled folder").mkdir()
    (tmp_path / "Pancake").mkdir()
    (tmp_path / "Pancake" / "photo.png").write_bytes(b"png")
    report = load_recipes(tmp_path)
    assert report.recipes == []
    assert report.skipped == []


def test_image_only_document_is_skipped(tmp_path, make_docx):
    make_docx(tmp_path / "Pancake" / "photo.docx")
    make_docx(tmp_path / "Pancake" / "steps.docx", "Fry.")
    report = load_recipes(tmp_path)
    assert report.recipes[0].files == ("Pancake/steps.docx",)
    assert report.skipped == [("Pancake/photo.docx", "no text")]


def test_hidden_and_lock_files_are_ignored(tmp_path, make_docx):
    make_docx(tmp_path / "Pancake" / "~$Recipe.docx", "lock")
    make_docx(tmp_path / "Pancake" / ".Recipe.docx", "hidden")
    make_docx(tmp_path / ".trash" / "Old" / "Recipe.docx", "old")
    assert load_recipes(tmp_path).recipes == []


def test_corrupt_document_is_skipped(tmp_path, make_docx):
    (tmp_path / "Pancake").mkdir()
    (tmp_path / "Pancake" / "broken.docx").write_bytes(b"not a zip")
    report = load_recipes(tmp_path)
    assert report.recipes == []
    assert report.skipped == [("Pancake/broken.docx", "unreadable (PackageNotFoundError)")]


def test_symlinks_are_not_followed(tmp_path, make_docx):
    outside = make_docx(tmp_path / "outside" / "secret.docx", "secret")
    root = tmp_path / "resources"
    (root / "Pancake").mkdir(parents=True)
    (root / "Pancake" / "link.docx").symlink_to(outside)
    (root / "Linked").symlink_to(outside.parent, target_is_directory=True)

    report = load_recipes(root)

    assert report.recipes == []
    assert report.skipped == [("Pancake/link.docx", "symlink")]


def test_large_files_are_skipped(tmp_path, make_docx, monkeypatch):
    make_docx(tmp_path / "Pancake" / "Recipe.docx", "Fry.")
    monkeypatch.setattr(loaders, "MAX_FILE_BYTES", 100)
    assert load_recipes(tmp_path).skipped == [("Pancake/Recipe.docx", "too large")]


def test_read_docx_includes_tables(tmp_path, make_docx):
    path = make_docx(tmp_path / "a.docx", "Ingredients", table=[["Flour", "2 cups"]])
    assert read_docx(path) == "Ingredients\nFlour | 2 cups"


def test_clean_line():
    assert clean_line("  Preheat   a grill.In a bowl, mix.  ") == "Preheat a grill. In a bowl, mix."
    assert clean_line("Cook 5 mins.Serve") == "Cook 5 mins. Serve"
    assert clean_line("Add 1.5 cups") == "Add 1.5 cups"
    assert clean_line("e.g. salt") == "e.g. salt"


def test_clean_title():
    assert clean_title("Pizza Crust ") == "Pizza Crust"
    assert clean_title("Turkish   Manti") == "Turkish Manti"
