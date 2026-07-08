"""Shared helpers for talking to a local Ollama server."""

import time

import requests

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def check_ollama(host):
    try:
        requests.get(f"{host}/api/tags", timeout=3).raise_for_status()
    except requests.RequestException:
        return False
    return True


def retry(fn, attempts=3, base_delay=0.5):
    """Runs fn() with exponential backoff on transient request failures.
    Re-raises the last exception if every attempt fails."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_exc


def embed_texts(texts, model, host, batch_size, timeout=300):
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        print(f"  batch {batch_num}/{total_batches} ({len(batch)} texts) ...")

        def do_request():
            resp = requests.post(
                f"{host}/api/embed",
                json={"model": model, "input": batch},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]

        embeddings.extend(retry(do_request))
    return embeddings


def embed_query(question, model, host, timeout=300):
    """Embeds a single query string with the 'search_query:' task prefix
    nomic-embed-text expects (the counterpart to 'search_document:' used at
    ingestion time -- see log_ingest.py / code_ingest.py)."""
    return embed_texts([f"search_query: {question}"], model, host, batch_size=1, timeout=timeout)[0]


def chat_with_tools(messages, tools, model, host, timeout=300):
    """Calls Ollama's /api/chat with tool definitions attached. Returns the
    raw assistant message dict, which may contain a 'tool_calls' list if the
    model decided to invoke one or more tools instead of answering directly."""

    def do_request():
        resp = requests.post(
            f"{host}/api/chat",
            json={"model": model, "messages": messages, "tools": tools, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]

    return retry(do_request)
