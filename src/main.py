import sys
import os
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from indexing.indexer import CodeIndexer
from agent.core import CodeAgent

def main():
    print("="*50)
    print("Agentic Code RAG - Starting Up")
    print("="*50)

    # 1. Indexing Phase
    project_path = config.project_root
    print(f"Target Project Path: {project_path}")
    
    if not os.path.exists(project_path):
        print(f"Error: Path {project_path} does not exist.")
        # If relative path fails, try to resolve it relative to CWD
        abs_path = os.path.abspath(project_path)
        print(f"Resolved Absolute Path: {abs_path}")
        if not os.path.exists(abs_path):
            print("❌ Critical Error: Target directory not found. Please check PROJECT_ROOT in .env")
            return
    
    print("\n[1/2] Indexing Codebase...")
    start_time = time.time()
    try:
        indexer = CodeIndexer(project_path)
        indexer.index_project()
        print(f"[OK] Indexing complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"[X] Indexing failed: {e}")
        return

    # 2. Agent Initialization
    print("\n[2/2] Initializing Agent...")
    try:
        agent = CodeAgent()
        print("[OK] Agent ready.")
    except Exception as e:
        print(f"[X] Agent initialization failed: {e}")
        return

    # 3. Interactive Loop
    print("\n" + "="*50)
    print("Interactive Session Started")
    print("Type 'exit' or 'quit' to end session.")
    print("="*50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            print("Agent is thinking...")
            response = agent.chat(user_input)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\nUser interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()