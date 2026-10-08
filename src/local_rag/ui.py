from __future__ import annotations

import ollama
import streamlit as st

from local_rag import pipeline
from local_rag.answer import NO_MATCH, cited, safe_markdown, stream_answer
from local_rag.config import ConfigError, Settings, load_settings
from local_rag.pipeline import NothingToIngestError
from local_rag.retrieval import MODES, Retriever
from local_rag.store import Hit, IndexMissingError
from local_rag.uploads import UploadError, save_recipe


@st.cache_resource
def _retriever(settings: Settings) -> Retriever:
    return pipeline.make_retriever(settings)


def main() -> None:
    st.set_page_config(page_title="Local RAG", page_icon="🍳")
    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(f"Config error: {exc}")
        st.stop()
    retriever = _retriever(settings)

    with st.sidebar:
        _sidebar(settings, retriever)

    st.title("Recipe assistant")
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(safe_markdown(message["content"]))
            _details(message)

    if question := st.chat_input("Ask about your recipes"):
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(safe_markdown(question))
        with st.chat_message("assistant"):
            messages.append(_answer(settings, retriever, question))


def _answer(settings: Settings, retriever: Retriever, question: str) -> dict:
    try:
        with st.spinner("Searching…"):
            hits = retriever.search(
                question, settings.top_k, st.session_state.mode, max_distance=settings.max_distance
            )
    except IndexMissingError:
        text = "No index yet. Use **Rebuild index** in the sidebar."
        st.markdown(text)
        return {"role": "assistant", "content": text}
    if not hits:
        st.markdown(NO_MATCH)
        return {"role": "assistant", "content": NO_MATCH}

    placeholder = st.empty()
    text = ""
    try:
        client = ollama.Client(host=settings.ollama_host)
        for token in stream_answer(client, settings.llm_model, question, hits):
            text += token
            placeholder.markdown(safe_markdown(text))
    except Exception as exc:
        st.error(f"Could not get an answer from Ollama: {exc}")
    message = {"role": "assistant", "content": text, "hits": hits, "sources": cited(text, hits)}
    _details(message)
    return message


def _details(message: dict) -> None:
    for n, hit in message.get("sources", []):
        with st.expander(f"[{n}] {hit.title}"):
            st.text(hit.text)
    if st.session_state.debug and message.get("hits"):
        with st.expander("Retrieved chunks"):
            for i, hit in enumerate(message["hits"], 1):
                _debug_hit(i, hit)


def _debug_hit(i: int, hit: Hit) -> None:
    st.markdown(f"**[{i}] {hit.title}** · `{hit.id}`")
    st.caption(
        f"distance {hit.distance:.3f} · keyword {hit.keyword_score:.2f} "
        f"({hit.keyword_coverage:.0%} of terms) · fused {hit.score:.4f}"
    )
    st.text(hit.text)


def _sidebar(settings: Settings, retriever: Retriever) -> None:
    st.radio("Retrieval", MODES, key="mode", horizontal=True, format_func=str.capitalize)
    st.toggle("Show retrieved chunks", key="debug")
    if st.button("Clear chat"):
        st.session_state.messages = []

    st.divider()
    st.subheader("Index")
    if st.button("Rebuild index"):
        _rebuild(settings, retriever)

    with st.form("upload", clear_on_submit=True):
        st.markdown("**Add a recipe**")
        title = st.text_input("Recipe name")
        files = st.file_uploader("Word documents", type=["docx"], accept_multiple_files=True)
        if st.form_submit_button("Add and re-index"):
            try:
                folder = save_recipe(
                    settings.resources_dir, title, [(f.name, f.getvalue()) for f in files or []]
                )
            except UploadError as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved {folder.name}.")
                _rebuild(settings, retriever)

    st.caption(f"{retriever.store.count()} chunks indexed")


def _rebuild(settings: Settings, retriever: Retriever) -> None:
    with st.spinner("Indexing…"):
        try:
            result = pipeline.ingest(settings, retriever.embedder, retriever.store)
        except NothingToIngestError as exc:
            st.error(str(exc))
            return
    st.success(f"Indexed {result.chunks} chunks from {result.recipes} recipes.")
    if result.skipped:
        st.caption(f"Skipped {len(result.skipped)} files without usable text.")
    for chunk_id, tokens in result.truncated:
        st.warning(f"{chunk_id} has {tokens} tokens and is cut off when embedded.")


main()
