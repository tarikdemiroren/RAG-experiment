from conftest import FakeOllamaClient

from local_rag.answer import SYSTEM_PROMPT, build_messages, cited, relevant, stream_answer
from local_rag.store import Hit


def _hit(title, distance, text=None):
    return Hit(f"{title}#0", title, title, text or f"{title}\nsteps", distance)


def test_relevant_filters_by_distance():
    hits = [_hit("A", 0.2), _hit("B", 0.6), _hit("C", 0.61)]
    assert [h.title for h in relevant(hits, 0.6)] == ["A", "B"]


def test_build_messages_numbers_context():
    messages = build_messages("How long?", [_hit("Eggs", 0.3), _hit("Pancake", 0.5)])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert user == (
        "<context>\n[1] Eggs\nsteps\n\n[2] Pancake\nsteps\n</context>\n\nQuestion: How long?"
    )


def test_stream_answer_yields_tokens_and_sends_prompt():
    client = FakeOllamaClient(reply="Boil for 8 minutes [1].")
    hits = [_hit("Eggs", 0.3)]
    text = "".join(stream_answer(client, "llama3.2:3b", "How long?", hits))
    assert text.strip() == "Boil for 8 minutes [1]."
    (call,) = client.chats
    assert call["model"] == "llama3.2:3b"
    assert call["stream"] is True
    assert call["messages"] == build_messages("How long?", hits)


def test_cited_returns_referenced_hits_in_order():
    hits = [_hit("A", 0.1), _hit("B", 0.2), _hit("C", 0.3)]
    answer = "Use C [3], then A [1]. Also A again [1]."
    assert [(n, h.title) for n, h in cited(answer, hits)] == [(1, "A"), (3, "C")]


def test_cited_accepts_grouped_citations():
    hits = [_hit("A", 0.1), _hit("B", 0.2)]
    assert [n for n, _ in cited("Both [1, 2].", hits)] == [1, 2]
    assert [n for n, _ in cited("Both [1][2].", hits)] == [1, 2]


def test_cited_ignores_unknown_numbers_and_plain_text():
    hits = [_hit("A", 0.1)]
    assert cited("See [0], [2] and 1 cup [flour].", hits) == []
