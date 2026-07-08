"""
Query flow (Phase 3): embed a question with Ollama -> similarity search the
log and/or code Chroma collections -> for logs, read raw text straight from
metadata; for code, re-read the current file from disk at the stored line
range -> optionally synthesize an answer with qwen2.5:7b-instruct, citing
file/line for code.

Usage:
    python src/query.py "why do PAN verifications fail?"
    python src/query.py "where is useState used?" --target code --repo-path "C:\\path\\to\\project"
    python src/query.py "loan approvals today" --target logs --status fail
    python src/query.py "some question" --no-llm      (skip the LLM, just show retrieved matches)
"""

import argparse
import os
import sys

import chromadb
import requests

from ollama_utils import DEFAULT_OLLAMA_HOST, check_ollama, embed_texts

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about production logs "
    "and a codebase using only the context provided below. For code, cite "
    "the exact file path and line numbers you used. If the context doesn't "
    "contain the answer, say so instead of guessing."
)


def embed_query(question, host):
    return embed_texts([question], EMBED_MODEL, host, batch_size=1)[0]


def query_collection(collection, query_embedding, top_k, where=None):
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where or None,
        include=["metadatas", "distances"],
    )
    return list(zip(result["ids"][0], result["metadatas"][0], result["distances"][0]))


def read_code_snippet(repo_path, meta):
    abs_path = os.path.join(repo_path, meta["file"])
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError as e:
        return f"<could not read {abs_path}: {e}>"
    return "\n".join(lines[meta["start_line"] - 1 : meta["end_line"]])


def build_log_context(results):
    lines = []
    for _id, meta, dist in results:
        lines.append(
            f"- [{meta['level']}] [{meta['status']}] {meta['component']} "
            f"(seen {meta['count']}x, {meta['first_seen']} to {meta['last_seen']}): {meta['raw']}"
        )
    return "\n".join(lines)


def build_code_context(results, repo_path):
    blocks = []
    for _id, meta, dist in results:
        snippet = read_code_snippet(repo_path, meta) if repo_path else "<--repo-path not given, code not re-read>"
        blocks.append(
            f"### {meta['name']} ({meta['kind']}, {meta['language']}) - "
            f"{meta['file']}:{meta['start_line']}-{meta['end_line']}\n```\n{snippet}\n```"
        )
    return "\n\n".join(blocks)


def ask_llm(question, log_context, code_context, host):
    sections = [f"QUESTION:\n{question}"]
    if log_context:
        sections.append(f"LOG CONTEXT:\n{log_context}")
    if code_context:
        sections.append(f"CODE CONTEXT:\n{code_context}")

    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(sections)},
            ],
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Query the log/code Chroma collections via Ollama.")
    parser.add_argument("question")
    parser.add_argument("--persist-dir", default="chroma_store")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--target", choices=["logs", "code", "both"], default="both")
    parser.add_argument("--top-k-logs", type=int, default=5)
    parser.add_argument("--top-k-code", type=int, default=3)
    parser.add_argument("--status", choices=["fail", "success", "other"], default=None, help="Filter logs by status")
    parser.add_argument("--level", default=None, help="Filter logs by log level, e.g. ERROR")
    parser.add_argument("--repo-path", default=None, help="Root of the indexed codebase, needed to re-read code snippets from disk")
    parser.add_argument("--no-llm", action="store_true", help="Skip qwen2.5 synthesis, just print retrieved matches")
    args = parser.parse_args()

    if not check_ollama(args.ollama_host):
        print(f"ERROR: could not reach Ollama at {args.ollama_host}.", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=args.persist_dir)
    query_embedding = embed_query(args.question, args.ollama_host)

    log_context = ""
    code_context = ""

    if args.target in ("logs", "both"):
        try:
            logs_collection = client.get_collection("prod_logs")
            where = {k: v for k, v in (("status", args.status), ("level", args.level)) if v}
            log_results = query_collection(logs_collection, query_embedding, args.top_k_logs, where)
            log_context = build_log_context(log_results)
        except Exception as e:
            print(f"WARNING: could not query prod_logs: {e}", file=sys.stderr)

    if args.target in ("code", "both"):
        try:
            code_collection = client.get_collection("my_repo")
            code_results = query_collection(code_collection, query_embedding, args.top_k_code)
            code_context = build_code_context(code_results, args.repo_path)
        except Exception as e:
            print(f"WARNING: could not query my_repo: {e}", file=sys.stderr)

    if log_context:
        print("=== LOG MATCHES ===")
        print(log_context)
        print()
    if code_context:
        print("=== CODE MATCHES ===")
        print(code_context)
        print()

    if not args.no_llm:
        print(f"=== ANSWER ({CHAT_MODEL}) ===")
        print(ask_llm(args.question, log_context, code_context, args.ollama_host))


if __name__ == "__main__":
    main()
