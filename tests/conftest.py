import os
from types import SimpleNamespace

import pytest


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith("RAG_"):
            del os.environ[key]
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.environ.clear()
    os.environ.update(saved)


class FakeOllamaClient:
    def __init__(self, models=(), error=None):
        self.models = models
        self.error = error

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.models])


@pytest.fixture
def make_docx():
    from docx import Document

    def _make(path, *paragraphs):
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = Document()
        for text in paragraphs:
            doc.add_paragraph(text)
        doc.save(path)
        return path

    return _make
