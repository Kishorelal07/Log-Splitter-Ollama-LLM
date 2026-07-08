"""
Code RAG indexing (Phase 2): walk a codebase -> AST-chunk each source file into
functions/methods/classes (tree-sitter for Java/JS/JSX, Python's `ast` for .py)
-> embed each chunk's source text with Ollama -> store address only (file path
+ line range) in a Chroma collection. No code text is stored in Chroma -- query
time re-reads the current file from disk, so answers always reflect the latest
code, not a stale indexed copy.

Usage:
    python src/code_ingest.py --repo-path "C:\\path\\to\\project"
"""

import argparse
import ast
import hashlib
import os
import sys

import chromadb
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

from ollama_utils import DEFAULT_OLLAMA_HOST, check_ollama, embed_texts

EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 32

EXCLUDED_DIRS = {
    "node_modules", "target", "dist", "build",
    ".git", "__pycache__", ".venv", "venv", ".idea", ".vscode",
}

EXT_LANGUAGE = {
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
}

JAVA_PARSER = Parser(Language(tsjava.language()))
JS_PARSER = Parser(Language(tsjs.language()))

JAVA_TYPE_NODES = ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration")
JAVA_METHOD_NODES = ("method_declaration", "constructor_declaration")


def iter_source_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in EXT_LANGUAGE:
                yield os.path.join(dirpath, filename), EXT_LANGUAGE[ext]


def _name_of(node, source_bytes):
    n = node.child_by_field_name("name")
    if n is not None:
        return source_bytes[n.start_byte:n.end_byte].decode("utf-8", "replace")
    return None


def _line_range(node):
    return node.start_point[0] + 1, node.end_point[0] + 1


def extract_java_chunks(source_bytes):
    chunks = []

    def walk(node, class_stack):
        if node.type in JAVA_TYPE_NODES:
            stack = class_stack + [_name_of(node, source_bytes) or "?"]
            before = len(chunks)
            for child in node.children:
                walk(child, stack)
            if len(chunks) == before:
                start, end = _line_range(node)
                chunks.append({"name": ".".join(stack), "kind": "class", "start_line": start, "end_line": end})
            return

        if node.type in JAVA_METHOD_NODES:
            name = _name_of(node, source_bytes) or "?"
            kind = "constructor" if node.type == "constructor_declaration" else "method"
            start, end = _line_range(node)
            chunks.append({"name": ".".join(class_stack + [name]), "kind": kind, "start_line": start, "end_line": end})

        for child in node.children:
            walk(child, class_stack)

    walk(JAVA_PARSER.parse(source_bytes).root_node, [])
    return chunks


def extract_js_chunks(source_bytes):
    chunks = []

    def walk(node, class_stack):
        if node.type == "class_declaration":
            stack = class_stack + [_name_of(node, source_bytes) or "?"]
            before = len(chunks)
            for child in node.children:
                walk(child, stack)
            if len(chunks) == before:
                start, end = _line_range(node)
                chunks.append({"name": ".".join(stack), "kind": "class", "start_line": start, "end_line": end})
            return

        if node.type == "function_declaration":
            name = _name_of(node, source_bytes) or "?"
            start, end = _line_range(node)
            chunks.append({"name": ".".join(class_stack + [name]), "kind": "function", "start_line": start, "end_line": end})
        elif node.type == "method_definition":
            name = _name_of(node, source_bytes) or "?"
            start, end = _line_range(node)
            chunks.append({"name": ".".join(class_stack + [name]), "kind": "method", "start_line": start, "end_line": end})
        elif node.type == "variable_declarator":
            value = node.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function", "function_expression"):
                name = _name_of(node, source_bytes) or "?"
                start, end = _line_range(node)
                chunks.append({"name": ".".join(class_stack + [name]), "kind": "function", "start_line": start, "end_line": end})

        for child in node.children:
            walk(child, class_stack)

    walk(JS_PARSER.parse(source_bytes).root_node, [])
    return chunks


def extract_python_chunks(source_text):
    chunks = []

    def walk(node, class_stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                before = len(chunks)
                walk(child, class_stack + [child.name])
                if len(chunks) == before:
                    chunks.append({
                        "name": ".".join(class_stack + [child.name]),
                        "kind": "class",
                        "start_line": child.lineno,
                        "end_line": child.end_lineno,
                    })
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append({
                    "name": ".".join(class_stack + [child.name]),
                    "kind": "method" if class_stack else "function",
                    "start_line": child.lineno,
                    "end_line": child.end_lineno,
                })
                walk(child, class_stack + [child.name])
            else:
                walk(child, class_stack)

    walk(ast.parse(source_text), [])
    return chunks


def extract_chunks(path, language):
    with open(path, "rb") as f:
        source_bytes = f.read()
    try:
        if language == "java":
            return extract_java_chunks(source_bytes)
        if language == "javascript":
            return extract_js_chunks(source_bytes)
        if language == "python":
            return extract_python_chunks(source_bytes.decode("utf-8", "replace"))
    except (SyntaxError, UnicodeDecodeError, RecursionError) as e:
        print(f"WARNING: could not parse {path}: {e}", file=sys.stderr)
    return []


def chunk_id(rel_path, chunk):
    key = f"{rel_path}|{chunk['start_line']}|{chunk['end_line']}|{chunk['name']}"
    return "code_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def build_embed_text(rel_path, language, chunk, source_lines):
    snippet = "\n".join(source_lines[chunk["start_line"] - 1 : chunk["end_line"]])
    return f"[{language}] [{chunk['kind']}] {chunk['name']} in {rel_path}:\n{snippet}"


def main():
    parser = argparse.ArgumentParser(description="Index a codebase into Chroma via local Ollama embeddings.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--collection", default="my_repo")
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

    repo_path = os.path.abspath(args.repo_path)
    print(f"Walking {repo_path} ...")

    all_chunks = []  # list of (rel_path, language, chunk)
    file_count = 0
    for abs_path, language in iter_source_files(repo_path):
        file_count += 1
        rel_path = os.path.relpath(abs_path, repo_path)
        for chunk in extract_chunks(abs_path, language):
            all_chunks.append((rel_path, language, chunk))

    print(f"  {file_count} source files scanned, {len(all_chunks)} function/method/class chunks found")

    file_lines_cache = {}
    texts = []
    for rel_path, language, chunk in all_chunks:
        if rel_path not in file_lines_cache:
            with open(os.path.join(repo_path, rel_path), "r", encoding="utf-8", errors="replace") as f:
                file_lines_cache[rel_path] = f.read().splitlines()
        texts.append(build_embed_text(rel_path, language, chunk, file_lines_cache[rel_path]))

    print(f"Embedding {len(texts)} chunks with Ollama ({EMBED_MODEL}) ...")
    embeddings = embed_texts(texts, EMBED_MODEL, args.ollama_host, EMBED_BATCH_SIZE)

    client = chromadb.PersistentClient(path=args.persist_dir)
    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=args.collection)

    ids = [chunk_id(rel_path, chunk) for rel_path, _, chunk in all_chunks]
    metadatas = [
        {
            "file": rel_path,
            "language": language,
            "kind": chunk["kind"],
            "name": chunk["name"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
        }
        for rel_path, language, chunk in all_chunks
    ]

    print(f"Storing {len(ids)} entries in Chroma collection '{args.collection}' ({args.persist_dir}) ...")
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print("Done.")
    print(f"{file_count} files scanned -> {len(all_chunks)} chunks stored (no code text stored; re-read from disk at query time).")


if __name__ == "__main__":
    main()
