#!/usr/bin/env python3
"""Visualize the Code RAG Agent's LangGraph workflow.

No API keys required — works entirely from graph definition.

Usage:
    python scripts/visualize_graph.py --mermaid        # Print Mermaid to stdout (default)
    python scripts/visualize_graph.py --png graph.png  # Save PNG (uses Mermaid API)
    python scripts/visualize_graph.py --ascii           # ASCII art (requires grandalf)
"""

import argparse
import sys
import os

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def get_graph():
    """Compile the LangGraph workflow and return its DrawableGraph."""
    from agent.graph import define_graph

    app = define_graph()
    return app.get_graph()


def print_mermaid(graph):
    """Print Mermaid diagram to stdout."""
    mermaid = graph.draw_mermaid()
    print(mermaid)
    print("\n# Paste the above into https://mermaid.live/ to render")


def save_png(graph, output_path):
    """Save graph as PNG using Mermaid rendering API."""
    try:
        png_data = graph.draw_mermaid_png()
    except Exception as e:
        print(f"Error: Failed to render PNG: {e}", file=sys.stderr)
        print("This requires network access to the Mermaid rendering API.", file=sys.stderr)
        sys.exit(1)

    with open(output_path, "wb") as f:
        f.write(png_data)
    print(f"Graph saved to {output_path}")


def print_ascii(graph):
    """Print ASCII representation of the graph."""
    try:
        ascii_art = graph.draw_ascii()
    except ImportError:
        print(
            "Error: ASCII rendering requires the 'grandalf' package.",
            file=sys.stderr,
        )
        print("Install with: pip install grandalf", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(ascii_art)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the Code RAG Agent graph topology"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--mermaid",
        action="store_true",
        default=True,
        help="Print Mermaid diagram to stdout (default)",
    )
    group.add_argument(
        "--png",
        metavar="PATH",
        help="Save graph as PNG image",
    )
    group.add_argument(
        "--ascii",
        action="store_true",
        help="Print ASCII art representation (requires grandalf)",
    )

    args = parser.parse_args()
    graph = get_graph()

    if args.png:
        save_png(graph, args.png)
    elif args.ascii:
        print_ascii(graph)
    else:
        print_mermaid(graph)


if __name__ == "__main__":
    main()
