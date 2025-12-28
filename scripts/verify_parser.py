#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexing.parser import CodeParser


def _node_to_dict(node):
    return {
        "type": node.type,
        "name": node.name,
        "file_path": node.file_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "language": node.language,
        "parent_name": node.parent_name,
        "signature": node.signature,
        "return_type": node.return_type,
        "arguments": node.arguments,
        "imports": node.imports,
        "docstring": node.docstring,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse a file and dump CodeNode records."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the source file to parse",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        return 2

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()

    code_parser = CodeParser()
    nodes = code_parser.parse_file(file_path, code)
    payload = [_node_to_dict(n) for n in nodes]

    if args.format == "json":
        output_text = json.dumps(payload, ensure_ascii=True, indent=2)
    else:
        output_text = "\n".join(json.dumps(item, ensure_ascii=True) for item in payload)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
