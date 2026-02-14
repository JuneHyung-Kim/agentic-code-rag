import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from indexing.indexer import CodeIndexer
from indexing.storage.vector_store import get_vector_store
from indexing.storage.graph_store import get_graph_store
from retrieval.search_engine import get_search_engine
from agent.core import CodeAgent
from profiling.builder import ProfileBuilder
from profiling.profile_store import save_profile, reset_profile_cache

def index_project(project_path: str):
    if not os.path.exists(project_path):
        print(f"Error: Path {project_path} does not exist.")
        return
    indexer = CodeIndexer(project_path)
    indexer.index_project()


def init_project(project_path: str, use_llm: bool = True):
    """Index a project and generate a codebase profile."""
    project_path = os.path.abspath(project_path)
    if not os.path.exists(project_path):
        print(f"Error: Path {project_path} does not exist.")
        return

    # Phase 1: Index
    print(f"Indexing {project_path}...")
    indexer = CodeIndexer(project_path)
    indexer.index_project()

    # Phase 2: Extract profile
    print("Building codebase profile...")
    builder = ProfileBuilder(get_vector_store(), get_graph_store(), project_path)
    profile = builder.build()

    # Phase 3: Optional LLM synthesis
    if use_llm:
        print("Generating AI synthesis (two-step)...")
        from profiling.synthesizer import synthesize_profile
        synthesize_profile(profile)

    # Phase 4: Persist
    save_profile(profile)
    reset_profile_cache()
    print(f"Profile saved to ./db/codebase_profile.json and ./db/codebase_profile.md")
    print(f"  Files: {profile.total_files}, Symbols: {profile.total_symbols}")
    if profile.language_stats:
        langs = ", ".join(sorted(profile.language_stats.keys()))
        print(f"  Languages: {langs}")

def search_code(query: str, n_results: int = 5, alpha: float = 0.7, project_root: str = None):
    """
    Run hybrid search (semantic + keyword) against the indexed codebase.
    """
    engine = get_search_engine()
    results = engine.hybrid_search(
        query,
        n_results=n_results,
        alpha=alpha,
        project_root=project_root
    )

    search_scope = f" (filtered by project: {project_root})" if project_root else " (across all projects)"
    print(f"\nSearch results for: '{query}'{search_scope}\n")

    if not results:
        print("No results found.")
        return

    for i, res in enumerate(results):
        meta = res["metadata"]
        content = res["content"]
        print(f"--- Result {i+1} (Score: {res['score']:.2f}) ---")

        # Display relative_path if available, otherwise file_path
        display_path = meta.get('relative_path', meta.get('file_path'))
        project = meta.get('project_root', 'unknown')
        print(f"Project: {project}")
        print(f"File: {display_path}:{meta.get('start_line')}-{meta.get('end_line')}")
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
    import shutil

    print("⚠️  WARNING: This will delete ALL indexed data, registry, and database files.")
    print("   This includes all projects and ChromaDB metadata.")
    confirm = input("Are you sure you want to reset the database? [y/N]: ")

    if confirm.lower() == 'y':
        try:
            db_path = "./db"

            if os.path.exists(db_path):
                # Complete removal of db directory
                shutil.rmtree(db_path)
                print("✅ Database directory completely removed.")

                # Recreate empty db directory
                os.makedirs(db_path, exist_ok=True)
                print("✅ Clean database directory created.")
            else:
                print("ℹ️  No database directory found.")

        except Exception as e:
            print(f"❌ Failed to reset database: {e}")
            print("   You may need to manually delete the './db' folder.")
            print("   Command: rm -rf ./db")
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
    search_parser.add_argument(
        "--project",
        "-p",
        type=str,
        default=None,
        help="Filter results by project root path (optional)",
    )

    # Init command
    init_parser = subparsers.add_parser("init", help="Index a project and generate a codebase profile")
    init_parser.add_argument("path", help="Path to the project to index and profile")
    init_parser.add_argument("--llm", action="store_true", help="Enable LLM-based two-step synthesis")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session with the AI Agent")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset (Clear) the vector database")

    args = parser.parse_args()

    if args.command == "init":
        init_project(args.path, use_llm=args.llm)
    elif args.command == "index":
        index_project(args.path)
    elif args.command == "search":
        search_code(
            args.query,
            n_results=args.n_results,
            alpha=args.alpha,
            project_root=args.project
        )
    elif args.command == "chat":
        start_chat()
    elif args.command == "reset":
        reset_database()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
