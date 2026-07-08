"""Shared helpers for talking to a local Ollama server."""

import requests

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def check_ollama(host):
    try:
        requests.get(f"{host}/api/tags", timeout=3).raise_for_status()
    except requests.RequestException:
        return False
    return True


def embed_texts(texts, model, host, batch_size):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{host}/api/embed",
            json={"model": model, "input": batch},
        )
        resp.raise_for_status()
        embeddings.extend(resp.json()["embeddings"])
    return embeddings
