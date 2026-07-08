"""
Print the contents of a Chroma collection for quick inspection.

Usage:
    python src/inspect_store.py
    python src/inspect_store.py --collection prod_logs --limit 5
"""

import argparse

import chromadb


def main():
    parser = argparse.ArgumentParser(description="Inspect a Chroma collection.")
    parser.add_argument("--persist-dir", default="chroma_store")
    parser.add_argument("--collection", default="prod_logs")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)

    print("Collections:", [c.name for c in client.list_collections()])

    collection = client.get_collection(args.collection)
    print(f"\n'{args.collection}' has {collection.count()} entries\n")

    result = collection.get(limit=args.limit, include=["metadatas", "documents"])
    for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        print(f"id: {id_}")
        print(f"  doc:  {doc}")
        print(f"  meta: {meta}")
        print()


if __name__ == "__main__":
    main()
