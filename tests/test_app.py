"""Smoke tests for the Streamlit app using Streamlit's native AppTest harness.

These run the actual app script in-process (no browser) and assert it renders without
exceptions and that the core user flow - load sample docs, then ask a question -
produces a grounded, attributed answer through the real pipeline.
"""

import pytest

from streamlit.testing.v1 import AppTest

APP = "app/streamlit_app.py"


def _find_button(at, label_substr):
    for b in at.button:
        if label_substr.lower() in (b.label or "").lower():
            return b
    raise AssertionError(f"Button containing {label_substr!r} not found")


def test_app_renders_without_exception():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert len(at.tabs) == 5
    labels = [s.label for s in at.sidebar.selectbox]
    assert "Embedding model" in labels and "Vector database" in labels
    slider_labels = [s.label for s in at.sidebar.slider]
    for expected in ["Chunk size (chars)", "Chunk overlap (chars)", "Top-K retrieved",
                     "LLM temperature"]:
        assert expected in slider_labels


@pytest.mark.integration
def test_load_samples_then_ask():
    at = AppTest.from_file(APP, default_timeout=300).run()
    # Reset any index carried over from a previous session for a clean run.
    _find_button(at, "Reset index").click().run()
    # Load the bundled sample documents and index them.
    _find_button(at, "Load sample documents").click().run()
    assert not at.exception

    # After indexing, ask a grounded question via the chat input.
    assert at.chat_input, "expected a chat input on the Chat tab"
    at.chat_input[0].set_value(
        "How many weeks of paid maternity leave do employees get?"
    ).run()
    assert not at.exception

    # The generated answer should be grounded (mentions 26).
    markdowns = " ".join(m.value for m in at.markdown)
    assert "26" in markdowns
