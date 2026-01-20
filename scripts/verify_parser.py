#!/usr/bin/env python3
import argparse
import json
import os
import sys
import random

# Usage examples:
# python scripts/verify_parser.py file src/main.cpp
# python scripts/verify_parser.py db --n 50 --out db_dump.json

# Add project root to sys.path to allow module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from indexing.parser import CodeParser
    from storage.vector_store import VectorStore
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running this script correctly from the project root or scripts directory.")
    sys.exit(1)


def _node_to_dict(node):
    """Convert CodeNode object to dictionary (includes preview for debugging)."""
    return {
        "source": "live_parsing",
        "type": node.type,
        "name": node.name,
        "parent_name": node.parent_name,
        "file_path": os.path.basename(node.file_path),
        "location": f"L{node.start_line + 1}-L{node.end_line + 1}",
        "signature": node.signature,
        "docstring_preview": (node.docstring[:50] + "...") if node.docstring else None,
        "content_preview": node.content[:150].replace("\n", "\\n") + "..." if len(node.content) > 150 else node.content,
    }


def inspect_file(file_path, output_file=None):
    """Parses a specific file and shows the results."""
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"Error: File not found: {abs_path}", file=sys.stderr)
        return

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return

    print(f"🔍 Parsing File: {abs_path}")
    print(f"   Size: {len(code)} bytes")
    print("-" * 60)

    try:
        code_parser = CodeParser()
        nodes = code_parser.parse_file(abs_path, code)
    except Exception as e:
        print(f"❌ Parser failed: {e}", file=sys.stderr)
        return

    payload = [_node_to_dict(n) for n in nodes]
    _output_results(payload, output_file)


def inspect_db_samples(n_samples, output_file=None):
    """Fetches random samples from the Vector DB."""
    print(f"🔍 Inspecting Vector Database (Random {n_samples} samples)...")
    
    try:
        store = VectorStore()
        # Fetch all data (ids, metadatas, documents)
        all_data = store.collection.get(include=['metadatas', 'documents'])
        total_docs = len(all_data['ids'])
    except Exception as e:
        print(f"❌ Failed to connect to DB: {e}")
        return

    if total_docs == 0:
        print("❌ Database is empty. Run indexing first.")
        return

    print(f"📚 Total indexed documents: {total_docs}")
    
    # Random sampling
    sample_indices = random.sample(range(total_docs), min(n_samples, total_docs))
    payload = []

    for idx in sample_indices:
        doc = all_data['documents'][idx]
        meta = all_data['metadatas'][idx]
        
        # Clean up metadata for display
        item = {
            "source": "database",
            "id": all_data['ids'][idx],
            "type": meta.get('type', 'unknown'),
            "name": meta.get('name', 'unknown'),
            "parent_name": meta.get('parent_name'),
            "file_path": os.path.basename(meta.get('file_path', 'unknown')),
            "location": f"L{meta.get('start_line', '?')}-L{meta.get('end_line', '?')}",
            "signature": meta.get('signature'),
            "content_preview": doc[:150].replace("\n", "\\n") + "..." if len(doc) > 150 else doc,
            "full_path": meta.get('file_path', 'unknown') # For reference
        }
        payload.append(item)

    _output_results(payload, output_file)


def _output_results(payload, output_file):
    """Helper to print or save results."""
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {output_file}")
    else:
        print(f"✅ Retrieved {len(payload)} items:\n")
        for i, item in enumerate(payload):
            print(f"[{i+1}] {item['type']} : {item['name']}")
            print(f"    File: {item['file_path']}")
            print(f"    Loc : {item['location']}")
            if item.get('parent_name'):
                print(f"    Parent: {item['parent_name']}")
            if item.get('signature'):
                print(f"    Sig : {item['signature']}")
            print(f"    Preview: {item['content_preview']}")
            print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Parser & Inspect Database: Check parsing logic or inspect DB contents."
    )
    
    # Mode selection
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")
    
    # Mode 1: File Parsing (Existing functionality)
    file_parser = subparsers.add_parser("file", help="Parse a specific source file")
    file_parser.add_argument("path", help="Path to the source file")
    file_parser.add_argument("--out", help="Output JSON file path")

    # Mode 2: DB Inspection (New functionality)
    db_parser = subparsers.add_parser("db", help="Inspect random samples from the Vector DB")
    db_parser.add_argument("--n", type=int, default=20, help="Number of samples to retrieve")
    db_parser.add_argument("--out", help="Output JSON file path")

    args = parser.parse_args()

    if args.mode == "file":
        inspect_file(args.path, args.out)
    elif args.mode == "db":
        inspect_db_samples(args.n, args.out)

if __name__ == "__main__":
    main()