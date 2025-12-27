import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from indexing.indexer import CodeIndexer
from indexing.vector_store import VectorStore
from agent.core import CodeAgent

def index_project(project_path: str):
    if not os.path.exists(project_path):
        print(f"Error: Path {project_path} does not exist.")
        return
    indexer = CodeIndexer(project_path)
    indexer.index_project()

def search_code(query: str):
    store = VectorStore()
    results = store.query(query)
    
    print(f"\nSearch results for: '{query}'\n")
    
    # ChromaDB query results structure:
    # {'ids': [...], 'distances': [...], 'metadatas': [...], 'documents': [...]}
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found.")
        return

    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        print(f"--- Result {i+1} ({meta['file_path']}:{meta['start_line']}) ---")
        print(f"Type: {meta['type']}, Name: {meta['name']}")
        print(doc[:200] + "..." if len(doc) > 200 else doc)
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

def main():
    parser = argparse.ArgumentParser(description="OS Devel Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a project")
    index_parser.add_argument("path", help="Path to the project to index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the indexed code")
    search_parser.add_argument("query", help="Query string")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session with the AI Agent")

    args = parser.parse_args()

    if args.command == "index":
        index_project(args.path)
    elif args.command == "search":
        search_code(args.query)
    elif args.command == "chat":
        start_chat()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
