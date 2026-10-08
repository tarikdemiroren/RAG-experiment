import os

import pytest
from conftest import FakeOllamaClient

from local_rag import cli


@pytest.fixture
def fake_ollama(monkeypatch):
    hosts = []
    state = {"models": ["llama3.2:3b"], "error": None}

    def factory(host):
        hosts.append(host)
        return FakeOllamaClient(state["models"], state["error"])

    monkeypatch.setattr("ollama.Client", factory)
    state["hosts"] = hosts
    return state


def test_check_all_ok(isolated_env, make_docx, fake_ollama, capsys):
    make_docx(isolated_env / "resources" / "Pancake" / "Recipe.docx", "mix")
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "[  ok] resources" in out
    assert "[  ok] llm model" in out


def test_check_reports_failures(isolated_env, fake_ollama, capsys):
    fake_ollama["models"] = []
    assert cli.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] resources" in out
    assert "[FAIL] llm model" in out


def test_dotenv_in_cwd_is_loaded(isolated_env, fake_ollama):
    (isolated_env / ".env").write_text("RAG_OLLAMA_HOST=http://localhost:9999\n")
    cli.main(["check"])
    assert fake_ollama["hosts"] == ["http://localhost:9999"]


def test_dotenv_does_not_override_real_env(isolated_env, fake_ollama):
    (isolated_env / ".env").write_text("RAG_OLLAMA_HOST=http://localhost:9999\n")
    os.environ["RAG_OLLAMA_HOST"] = "http://localhost:7777"
    cli.main(["check"])
    assert fake_ollama["hosts"] == ["http://localhost:7777"]


def test_dotenv_in_parent_dir_is_ignored(isolated_env, fake_ollama, monkeypatch):
    (isolated_env / ".env").write_text("RAG_OLLAMA_HOST=http://localhost:9999\n")
    child = isolated_env / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    cli.main(["check"])
    assert fake_ollama["hosts"] == ["http://127.0.0.1:11434"]


def test_config_error_exits_2(isolated_env, fake_ollama, capsys):
    os.environ["RAG_OLLAMA_HOST"] = "http://203.0.113.7:11434"
    assert cli.main(["check"]) == 2
    assert "config error" in capsys.readouterr().err
    assert fake_ollama["hosts"] == []


def test_env_defaults_applied(isolated_env, fake_ollama):
    os.environ.pop("ANONYMIZED_TELEMETRY", None)
    cli.main(["check"])
    assert os.environ["ANONYMIZED_TELEMETRY"] == "False"


def test_command_required(isolated_env):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


@pytest.fixture
def pipeline(isolated_env, make_docx, monkeypatch, fake_ollama):
    from conftest import FakeEmbedder

    embedder = FakeEmbedder()
    monkeypatch.setattr(cli, "make_embedder", lambda settings: embedder)
    recipes = isolated_env / "resources" / "recipes"
    make_docx(recipes / "Pancake" / "Ingredients.docx", "flour milk egg")
    make_docx(recipes / "Pancake" / "Recipe.docx", "Fry pancakes in butter.")
    make_docx(recipes / "Pancake" / "Photo.docx")
    make_docx(recipes / "Hummus" / "Recipe.docx", "Blend chickpeas with lemon and garlic.")
    (recipes / "Empty").mkdir()
    return embedder


def test_ingest(pipeline, capsys):
    assert cli.main(["ingest"]) == 0
    out = capsys.readouterr().out
    assert "skipped recipes/Pancake/Photo.docx: no text" in out
    assert "indexed 2 chunks from 2 recipes" in out


def test_ingest_warns_about_truncation(pipeline, capsys):
    pipeline.max_tokens = 3
    cli.main(["ingest"])
    assert "warning: recipes/Hummus#0 has 7 tokens; only the first 3 are embedded" in (
        capsys.readouterr().out
    )


def test_ingest_without_documents(isolated_env, capsys):
    assert cli.main(["ingest"]) == 1
    assert "no documents with text" in capsys.readouterr().err


def test_search(pipeline, capsys):
    cli.main(["ingest"])
    capsys.readouterr()
    assert cli.main(["search", "fry pancakes"]) == 0
    out = capsys.readouterr().out
    assert out.index("[1] Pancake") < out.index("[2] Hummus")


def test_ask_before_ingest(pipeline, capsys):
    assert cli.main(["ask", "pancakes?"]) == 1
    assert "run `local-rag ingest` first" in capsys.readouterr().err


def test_ask_answers_with_sources(pipeline, capsys, monkeypatch):
    client = FakeOllamaClient(reply="Fry them [1].")
    monkeypatch.setattr("ollama.Client", lambda host: client)
    os.environ["RAG_MAX_DISTANCE"] = "0.9"
    cli.main(["ingest"])
    capsys.readouterr()

    assert cli.main(["ask", "how do I fry pancakes"]) == 0

    out = capsys.readouterr().out
    assert out == "Fry them [1]. \n\nSources:\n[1] Pancake\n"
    assert "Fry pancakes in butter." in client.chats[0]["messages"][1]["content"]


def test_ask_without_relevant_chunks_skips_llm(pipeline, capsys, monkeypatch):
    client = FakeOllamaClient()
    monkeypatch.setattr("ollama.Client", lambda host: client)
    os.environ["RAG_MAX_DISTANCE"] = "0.0"
    cli.main(["ingest"])
    capsys.readouterr()

    assert cli.main(["ask", "capital of France", "--show-context"]) == 0

    assert capsys.readouterr().out == "No relevant recipes found.\n"
    assert client.chats == []


def test_ask_lists_only_cited_sources(pipeline, capsys, monkeypatch):
    client = FakeOllamaClient(reply="Blend them [2].")
    monkeypatch.setattr("ollama.Client", lambda host: client)
    os.environ["RAG_MAX_DISTANCE"] = "2.0"
    cli.main(["ingest"])
    capsys.readouterr()

    cli.main(["ask", "how do I fry pancakes"])

    out = capsys.readouterr().out
    assert out.endswith("Sources:\n[2] Hummus\n")
    assert "Pancake" not in out
    assert "distance" not in out


def test_ask_without_citations_prints_no_sources(pipeline, capsys, monkeypatch):
    client = FakeOllamaClient(reply="I don't know.")
    monkeypatch.setattr("ollama.Client", lambda host: client)
    os.environ["RAG_MAX_DISTANCE"] = "2.0"
    cli.main(["ingest"])
    capsys.readouterr()

    cli.main(["ask", "calories?"])

    assert capsys.readouterr().out == "I don't know. \n"


def test_ask_show_context_prints_scores(pipeline, capsys, monkeypatch):
    monkeypatch.setattr("ollama.Client", lambda host: FakeOllamaClient(reply="Fry [1]."))
    cli.main(["ingest"])
    capsys.readouterr()

    cli.main(["ask", "fry pancakes", "--show-context"])

    out = capsys.readouterr().out
    assert out.startswith("--- [1] Pancake  id=recipes/Pancake#0\n    distance=")
    assert "(100% of terms)" in out


def test_ask_keyword_match_passes_distance_gate(pipeline, capsys, monkeypatch):
    client = FakeOllamaClient(reply="Use garlic [1].")
    monkeypatch.setattr("ollama.Client", lambda host: client)
    os.environ["RAG_MAX_DISTANCE"] = "0.0"
    cli.main(["ingest"])
    capsys.readouterr()

    cli.main(["ask", "garlic"])

    assert capsys.readouterr().out.endswith("Sources:\n[1] Hummus\n")


@pytest.mark.parametrize(("mode", "count"), [("vector", 2), ("keyword", 1), ("hybrid", 2)])
def test_search_modes(pipeline, capsys, mode, count):
    cli.main(["ingest"])
    capsys.readouterr()
    cli.main(["search", "lemon", "--mode", mode])
    out = capsys.readouterr().out
    assert out.startswith("--- [1] Hummus")
    assert out.count("--- [") == count
