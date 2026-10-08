import pytest

from local_rag import uploads
from local_rag.loaders import load_recipes
from local_rag.uploads import UploadError, safe_title, save_recipe


@pytest.fixture
def docx_bytes(tmp_path, make_docx):
    def _make(*paragraphs):
        return make_docx(tmp_path / "src" / "doc.docx", *paragraphs).read_bytes()

    return _make


def test_saves_files_under_uploads(tmp_path, docx_bytes):
    resources = tmp_path / "resources"
    folder = save_recipe(
        resources, "Lentil Soup", [("a.docx", docx_bytes("lentils")), ("b.docx", docx_bytes())]
    )
    assert folder == (resources / "uploads" / "Lentil Soup").resolve()
    assert sorted(p.name for p in folder.iterdir()) == ["1.docx", "2.docx"]
    report = load_recipes(resources)
    assert [(r.id, r.title, r.text) for r in report.recipes] == [
        ("uploads/Lentil Soup", "Lentil Soup", "lentils")
    ]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("  Turkish   Çılbır ", "Turkish Çılbır"),
        ("../../etc/passwd", "etcpasswd"),
        ("a/b\\c", "abc"),
        ("Mac & Cheese (easy), v2.", "Mac & Cheese (easy), v2"),
        ("x" * 100, "x" * 80),
    ],
)
def test_safe_title(title, expected):
    assert safe_title(title) == expected


@pytest.mark.parametrize("title", ["", "   ", "..", "/", "***"])
def test_safe_title_rejects_empty_names(title):
    with pytest.raises(UploadError, match="recipe name"):
        safe_title(title)


def test_uploaded_filenames_are_never_used(tmp_path, docx_bytes):
    folder = save_recipe(tmp_path, "Soup", [("../../evil.docx", docx_bytes("x"))])
    assert [p.name for p in folder.iterdir()] == ["1.docx"]
    assert not (tmp_path / "evil.docx").exists()


def test_existing_recipe_is_not_overwritten(tmp_path, docx_bytes):
    save_recipe(tmp_path, "Soup", [("a.docx", docx_bytes("first"))])
    with pytest.raises(UploadError, match="already exists"):
        save_recipe(tmp_path, "Soup", [("a.docx", docx_bytes("second"))])


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ([], "at least one"),
        ([("notes.txt", b"hello")], "only .docx"),
        ([("fake.docx", b"not a zip")], "not a valid .docx"),
    ],
)
def test_invalid_uploads(tmp_path, files, message):
    with pytest.raises(UploadError, match=message):
        save_recipe(tmp_path, "Soup", files)
    assert not (tmp_path / "uploads" / "Soup").exists()


def test_documents_without_text_are_rejected(tmp_path, docx_bytes):
    with pytest.raises(UploadError, match="no text"):
        save_recipe(tmp_path, "Soup", [("photo.docx", docx_bytes())])


def test_large_file_is_rejected(tmp_path, docx_bytes, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_FILE_BYTES", 100)
    with pytest.raises(UploadError, match="larger than"):
        save_recipe(tmp_path, "Soup", [("a.docx", docx_bytes("x"))])


def test_zip_bomb_is_rejected(tmp_path, docx_bytes, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_UNPACKED_BYTES", 1000)
    with pytest.raises(UploadError, match="unpacks to more than"):
        save_recipe(tmp_path, "Soup", [("a.docx", docx_bytes("x"))])


def test_failed_write_leaves_nothing_behind(tmp_path, docx_bytes, monkeypatch):
    def fail(self, target):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.rename", fail)
    with pytest.raises(OSError, match="disk full"):
        save_recipe(tmp_path, "Soup", [("a.docx", docx_bytes("x"))])
    assert list((tmp_path / "uploads").iterdir()) == []
