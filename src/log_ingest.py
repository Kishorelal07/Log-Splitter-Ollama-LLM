"""
Log RAG indexing (Phase 1): parse -> mask PII -> dedupe by template -> classify
status -> embed with a local Ollama model -> store in a Chroma collection.

Usage:
    ollama pull nomic-embed-text        (one-time, run once Ollama is installed)
    python src/log_ingest.py --log-file sample_logs/production.log
"""

import argparse
import hashlib
import re
import sys

import chromadb

from ollama_utils import DEFAULT_OLLAMA_HOST, check_ollama, embed_texts

LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>\w+)\s+"
    r"(?P<component>\S+)\s+-\s+"
    r"(?P<message>.*)$"
)

FAIL_KEYWORDS = ("fail", "error", "denied", "rejected", "locked", "lost")
SUCCESS_KEYWORDS = ("success", "succeeded", "completed", "approved")

EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 32


def mask_sensitive(text):
    """Mask PAN/Aadhaar/account numbers and collapse variable IDs into
    placeholders. This both protects PII before it leaves the process and
    produces a stable template string used for deduplication."""
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "<PAN>", text)
    text = re.sub(
        r"(?i)\b(aadhaar)(\s+for)?\s+\d{9,14}\b",
        lambda m: f"{m.group(1)}{m.group(2) or ''} <AADHAAR>",
        text,
    )
    text = re.sub(
        r"(?i)\b(account)\s+\d{6,18}\b",
        lambda m: f"{m.group(1)} <ACCOUNT>",
        text,
    )
    text = re.sub(r"\bLN\d+\b", "<LOAN_ID>", text)
    text = re.sub(
        r"(?i)\b(user_id)\s+\d+\b",
        lambda m: f"{m.group(1)} <USER_ID>",
        text,
    )
    text = re.sub(
        r"(?i)\b(amount)\s+\d+\b",
        lambda m: f"{m.group(1)} <AMOUNT>",
        text,
    )
    return text


def classify_status(level, message):
    lowered = message.lower()
    if level in ("ERROR", "FATAL") or any(k in lowered for k in FAIL_KEYWORDS):
        return "fail"
    if any(k in lowered for k in SUCCESS_KEYWORDS):
        return "success"
    return "other"


def parse_log_file(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = LOG_LINE_RE.match(line)
            if not m:
                print(f"WARNING: skipping unparsable line {line_no}: {line!r}", file=sys.stderr)
                continue
            entries.append(m.groupdict())
    return entries


def dedupe_entries(entries):
    """Collapse entries that share (component, level, masked message) into a
    single template with a running count and first/last seen timestamps."""
    templates = {}
    order = []
    for e in entries:
        masked_message = mask_sensitive(e["message"])
        status = classify_status(e["level"], e["message"])
        key = (e["component"], e["level"], masked_message)

        if key not in templates:
            templates[key] = {
                "component": e["component"],
                "level": e["level"],
                "status": status,
                "masked_message": masked_message,
                "count": 0,
                "first_seen": e["timestamp"],
                "last_seen": e["timestamp"],
            }
            order.append(key)

        t = templates[key]
        t["count"] += 1
        t["first_seen"] = min(t["first_seen"], e["timestamp"])
        t["last_seen"] = max(t["last_seen"], e["timestamp"])

    return [templates[k] for k in order]


def build_embed_text(template):
    return f"[{template['level']}] [{template['status']}] {template['component']}: {template['masked_message']}"


def template_id(template):
    key = f"{template['component']}|{template['level']}|{template['masked_message']}"
    return "log_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Index a log file into Chroma via local Ollama embeddings.")
    parser.add_argument("--log-file", default="sample_logs/production.log")
    parser.add_argument("--collection", default="prod_logs")
    parser.add_argument("--persist-dir", default="chroma_store")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection first")
    args = parser.parse_args()

    if not check_ollama(args.ollama_host):
        print(
            f"ERROR: could not reach Ollama at {args.ollama_host}. "
            f"Is it running? (start the Ollama app, or run `ollama serve`)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Parsing {args.log_file} ...")
    entries = parse_log_file(args.log_file)
    print(f"  {len(entries)} lines parsed")

    templates = dedupe_entries(entries)
    print(f"  {len(templates)} unique templates after dedup")

    texts = [build_embed_text(t) for t in templates]

    print(f"Embedding {len(texts)} templates with Ollama ({EMBED_MODEL}) ...")
    embed_inputs = [f"search_document: {t}" for t in texts]
    embeddings = embed_texts(embed_inputs, EMBED_MODEL, args.ollama_host, EMBED_BATCH_SIZE)

    client = chromadb.PersistentClient(path=args.persist_dir)
    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=args.collection)

    ids = [template_id(t) for t in templates]
    metadatas = [
        {
            "level": t["level"],
            "status": t["status"],
            "component": t["component"],
            "count": t["count"],
            "first_seen": t["first_seen"],
            "last_seen": t["last_seen"],
            "raw": text,
        }
        for t, text in zip(templates, texts)
    ]

    print(f"Storing {len(ids)} entries in Chroma collection '{args.collection}' ({args.persist_dir}) ...")
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

    print("Done.")
    print(f"Total log lines: {len(entries)} -> {len(templates)} unique templates stored.")


if __name__ == "__main__":
    main()
