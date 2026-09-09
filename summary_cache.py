"""Reuse successful AI text; briefly back off when generation is unavailable."""

from collections.abc import Callable

import streamlit as st


class _SummaryUnavailable(Exception):
    """Prevent unsuccessful responses from entering the persistent cache."""


@st.cache_data(persist="disk", show_spinner=False)
def _successful_response(
    prompt: str,
    system_msg: str | None,
    max_tokens: int,
    _generate: Callable[[], tuple[str | None, str]],
) -> tuple[str, str]:
    text, status = _generate()
    if status != "ok" or not text or not text.strip():
        raise _SummaryUnavailable(status if status != "ok" else "error: empty response")
    return text, status


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_response(
    prompt: str,
    system_msg: str | None,
    max_tokens: int,
    _generate: Callable[[], tuple[str | None, str]],
) -> tuple[str | None, str]:
    """Cache by full input, excluding credentials captured by the callback.

    Successful text is saved on local disk without a daily expiry. Errors only
    live in this five-minute memory cache, so a later visit can retry them.
    """
    try:
        return _successful_response(prompt, system_msg, max_tokens, _generate)
    except _SummaryUnavailable as error:
        return None, str(error)
