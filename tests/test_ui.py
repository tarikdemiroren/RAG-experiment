import os
from pathlib import Path

import pytest
import streamlit as st
from conftest import FakeEmbedder, FakeOllamaClient
from streamlit.testing.v1 import AppTest

from local_rag import pipeline
from local_rag.config import Settings

UI = str(Path(pipeline.__file__).with_name("ui.py"))


@pytest.fixture
def app(isolated_env, make_docx, monkeypatch):
    st.cache_resource.clear()
    embedder = FakeEmbedder()
    monkeypatch.setattr("local_rag.pipeline.make_embedder", lambda settings: embedder)
    client = FakeOllamaClient(reply="Fry them [1].")
    monkeypatch.setattr("ollama.Client", lambda host: client)
    make_docx(isolated_env / "resources" / "Pancake" / "Recipe.docx", "Fry pancakes in butter.")
    make_docx(isolated_env / "resources" / "Hummus" / "Recipe.docx", "Blend chickpeas.")
    at = AppTest.from_file(UI, default_timeout=30)
    at.client = client
    yield at
    st.cache_resource.clear()


def _ingest():
    settings = Settings.from_env()
    pipeline.ingest(settings, pipeline.make_embedder(settings), pipeline.make_store(settings))


def _ask(at, question):
    at.chat_input[0].set_value(question).run()
    return at


def test_renders_with_empty_index(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "Recipe assistant"
    assert app.sidebar.caption[-1].value == "0 chunks indexed"


def test_rebuild_index_button(app):
    app.run()
    next(b for b in app.sidebar.button if b.label == "Rebuild index").click().run()
    assert app.sidebar.success[0].value == "Indexed 2 chunks from 2 recipes."
    assert app.sidebar.caption[-1].value == "2 chunks indexed"


def test_question_before_ingest(app):
    app.run()
    _ask(app, "pancakes?")
    assert "No index yet" in app.chat_message[1].markdown[0].value


def test_answer_with_cited_sources(app):
    _ingest()
    app.run()
    _ask(app, "how do I fry pancakes")
    user, assistant = app.chat_message
    assert user.markdown[0].value == "how do I fry pancakes"
    assert assistant.markdown[0].value == "Fry them [1]."
    assert [e.label for e in assistant.expander] == ["[1] Pancake"]
    assert assistant.expander[0].text[0].value == "Pancake\nFry pancakes in butter."


def test_debug_toggle_shows_retrieved_chunks(app):
    _ingest()
    app.run()
    app.sidebar.toggle[0].set_value(True).run()
    _ask(app, "how do I fry pancakes")
    labels = [e.label for e in app.chat_message[1].expander]
    assert labels[-1] == "Retrieved chunks"


def test_no_relevant_chunks(app):
    os.environ["RAG_MAX_DISTANCE"] = "0.0"
    _ingest()
    app.run()
    _ask(app, "capital of France")
    assert app.chat_message[1].markdown[0].value == "No relevant recipes found."
    assert app.client.chats == []


def test_model_output_is_sanitized(app):
    app.client.reply = "See ![x](http://evil.test/leak) [1]."
    _ingest()
    app.run()
    _ask(app, "how do I fry pancakes")
    assert "evil.test" not in app.chat_message[1].markdown[0].value


def test_history_and_clear(app):
    _ingest()
    app.run()
    _ask(app, "how do I fry pancakes")
    _ask(app, "blend chickpeas")
    assert len(app.chat_message) == 4
    next(b for b in app.sidebar.button if b.label == "Clear chat").click().run()
    assert len(app.chat_message) == 0


def test_config_error(app):
    os.environ["RAG_OLLAMA_HOST"] = "http://203.0.113.7:11434"
    app.run()
    assert "not local" in app.error[0].value
