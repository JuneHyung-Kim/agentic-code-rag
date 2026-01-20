import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from indexing.indexer import CodeIndexer
from storage.vector_store import get_vector_store
from retrieval.search_engine import get_search_engine
from agent.core import CodeAgent

def index_project(project_path: str):
    if not os.path.exists(project_path):
        print(f"Error: Path {project_path} does not exist.")
        return
    indexer = CodeIndexer(project_path)
    indexer.index_project()

def search_code(query: str, n_results: int = 5, alpha: float = 0.7):
    """
    Run hybrid search (semantic + keyword) against the indexed codebase.
    """
    engine = get_search_engine()
    results = engine.hybrid_search(query, n_results=n_results, alpha=alpha)
    
    print(f"\nSearch results for: '{query}'\n")

    if not results:
        print("No results found.")
        return
    
    for i, res in enumerate(results):
        meta = res["metadata"]
        content = res["content"]
        print(f"--- Result {i+1} (Score: {res['score']:.2f}) ---")
        print(f"File: {meta.get('file_path')}:{meta.get('start_line')}-{meta.get('end_line')}")
        print(f"Type: {meta.get('type')}, Name: {meta.get('name')}")
        snippet = content[:400] + "..." if len(content) > 400 else content
        print(snippet)
        print("\n")

def start_chat():
    try:
        agent = CodeAgent()
        print("AI Agent initialized. Type 'exit' or 'quit' to stop.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            try:
                response = agent.chat(user_input)
                print(f"\nAgent: {response}")
            except Exception as e:
                print(f"Error during chat: {e}")
                
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Failed to start agent: {e}")

def reset_database():
    """Reset the vector database completely."""
    print("⚠️  WARNING: This will delete ALL indexed data.")
    confirm = input("Are you sure you want to reset the database? [y/N]: ")
    
    if confirm.lower() == 'y':
        try:
            store = get_vector_store()
            store.reset_collection()
            print("✅ Database reset successfully.")
        except Exception as e:
            print(f"❌ Failed to reset database: {e}")
            print("   (Tip: Try deleting the './db' folder manually if this fails.)")
    else:
        print("Operation cancelled.")

def main():
    parser = argparse.ArgumentParser(description="OS Devel Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a project")
    index_parser.add_argument("path", help="Path to the project to index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the indexed code (hybrid)")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument(
        "--n-results",
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for vector score (1.0=vector only, 0.0=keyword only)",
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session with the AI Agent")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset (Clear) the vector database")

    args = parser.parse_args()

    if args.command == "index":
        index_project(args.path)
    elif args.command == "search":
        search_code(args.query, n_results=args.n_results, alpha=args.alpha)
    elif args.command == "chat":
        start_chat()
    elif args.command == "reset":
        reset_database()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
