import sys
import os
import time

# Add src to path
# We need to add the 'src' directory to sys.path so that imports from 'src' work correctly.
# Assuming this file is in the project root, 'src' is a subdirectory.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import config
from indexing.indexer import CodeIndexer
from indexing.storage.vector_store import get_vector_store
from indexing.storage.graph_store import get_graph_store
from profiling.builder import ProfileBuilder
from profiling.synthesizer import synthesize_profile
from profiling.profile_store import save_profile, reset_profile_cache
from agent.model import APICallCancelled
from agent.core import CodeAgent

def main():
    print("="*50)
    print("Agentic Code RAG - Starting Up")
    print("="*50)

    # Resolve project path
    project_path = os.path.abspath(config.project_root)
    print(f"Target Project Path: {project_path}")

    if not os.path.exists(project_path):
        print(f"Error: Path {project_path} does not exist.")
        print("Please check PROJECT_ROOT in .env")
        return

    # --- Phase 1: Indexing ---
    print("\n[1/4] Indexing Codebase...")
    start_time = time.time()
    try:
        indexer = CodeIndexer(project_path)
        indexer.index_project()
        print(f"  Indexing complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"  Indexing failed: {e}")
        return

    # --- Phase 2: Build codebase profile (deterministic, no LLM) ---
    print("\n[2/4] Building codebase profile...")
    start_time = time.time()
    try:
        vector_store = get_vector_store()
        graph_store = get_graph_store()
        builder = ProfileBuilder(vector_store, graph_store, project_path)
        profile = builder.build()
        print(f"  Profile built in {time.time() - start_time:.2f}s")
        print(f"  Files: {profile.total_files}, Symbols: {profile.total_symbols}")
    except Exception as e:
        print(f"  Profile build failed: {e}")
        return

    # --- Phase 3: AI synthesis (first LLM call — confirmation may trigger) ---
    use_llm = os.environ.get("CODE_RAG_LLM_INIT", "").lower() in ("1", "true", "yes")
    if use_llm:
        print("\n[3/4] Generating AI synthesis (two-step)...")
        try:
            synthesize_profile(profile)
        except APICallCancelled:
            print("  Skipped AI synthesis (user declined API call).")
        except Exception as e:
            print(f"  AI synthesis failed: {e}")
            print("  Continuing without AI synthesis.")
    else:
        print("\n[3/4] Skipping AI synthesis (set CODE_RAG_LLM_INIT=1 to enable)")

    # --- Phase 4: Save profile ---
    print("\n[4/4] Saving profile...")
    try:
        save_profile(profile)
        reset_profile_cache()
        print("  Profile saved to ./db/")
    except Exception as e:
        print(f"  Profile save failed: {e}")

    # --- Agent Initialization ---
    print("\nInitializing Agent...")
    try:
        agent = CodeAgent()
        print("Agent ready.")
    except Exception as e:
        print(f"Agent initialization failed: {e}")
        return

    # --- Interactive Loop ---
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
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
