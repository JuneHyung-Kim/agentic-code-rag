import os
import sys
import random
import asyncio
import argparse
import json
from typing import List, Dict, Any, Optional

# python scripts/evaluate_search.py --mode generate --n 10
# python scripts/evaluate_search.py --mode evaluate --k 5

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from src.indexing.vector_store import VectorStore
    from src.config import config
    from src.utils.logger import logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Ensure tmp directory exists
DATASET_DIR = os.path.join(project_root, "tmp")
DATASET_FILE = os.path.join(DATASET_DIR, "test_dataset.json")

class SearchEvaluator:
    def __init__(self):
        self.store = VectorStore()
        
    async def generate_synthetic_query(self, document: str, metadata: Dict) -> str:
        """Generates a search query using the configured LLM."""
        code_context = document[:1500] 
        file_path = metadata.get('file_path', 'unknown')
        node_name = metadata.get('name', 'unknown')
        
        prompt = f"""
        Act as a Senior C/C++ Developer. Below is a code snippet from our codebase.
        
        [File]: {file_path}
        [Symbol]: {node_name}
        [Code Snippet]:
        ```c
        {code_context}
        ```
        
        Task: Write a single, specific search query that a developer would type to find this code.
        Rules:
        1. Do not use the exact function/class name if possible (simulate natural language search).
        2. Focus on the functionality, logic, or role of the code.
        3. Output ONLY the query string, no quotes.
        """

        try:
            if config.chat_provider == "openai":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=config.openai_api_key)
                response = await client.chat.completions.create(
                    model=config.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60
                )
                return response.choices[0].message.content.strip()

            elif config.chat_provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=config.gemini_api_key)
                model = genai.GenerativeModel(config.chat_model)
                response = await model.generate_content_async(prompt)
                return response.text.strip()

            elif config.chat_provider == "ollama":
                import ollama
                response = ollama.chat(
                    model=config.chat_model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return response['message']['content'].strip()
            
            else:
                return f"functionality of {node_name}"

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"code related to {node_name}"

    def ask_user_permission(self, sample_size: int, provider: str) -> bool:
        """
        Asks the user for permission before making costly API calls.
        """
        print("\n" + "!"*50)
        print(f"⚠️  APPROVAL REQUIRED")
        print("!"*50)
        print(f"You are about to generate {sample_size} synthetic queries.")
        print(f"Provider : {provider}")
        if provider in ["openai", "gemini"]:
            print(f"Cost     : This will incur API usage costs.")
        else:
            print(f"Cost     : Local execution (Ollama). Free, but consumes system resources.")
        
        try:
            response = input("\nDo you want to proceed? [y/N]: ").strip().lower()
        except EOFError:
            return False
        return response == 'y'

    def get_current_id_by_metadata(self, file_path: str, name: str) -> Optional[str]:
        """
        Dynamically resolves the current ID of a document using its metadata (File path + Name).
        Crucial for re-evaluation after re-indexing.
        """
        try:
            # Query by metadata
            result = self.store.collection.get(
                where={"$and": [{"file_path": file_path}, {"name": name}]}
            )
            if result['ids'] and len(result['ids']) > 0:
                return result['ids'][0]
            return None
        except Exception as e:
            logger.warning(f"Failed to resolve ID for {name} in {file_path}: {e}")
            return None

    async def generate_dataset(self, n_samples: int):
        """Generates a new test dataset and saves it to JSON."""
        print(f"\n🚀 Generating New Dataset (N={n_samples})...")
        
        # 1. Fetch all data
        all_data = self.store.collection.get(include=['metadatas', 'documents'])
        total_docs = len(all_data['ids'])
        
        if total_docs == 0:
            print("❌ Index is empty.")
            return

        # 2. User Permission Check
        actual_sample_size = min(n_samples, total_docs)
        if not self.ask_user_permission(actual_sample_size, config.chat_provider):
            print("❌ Operation cancelled by user.")
            return

        # 3. Random Sample
        indices = random.sample(range(total_docs), actual_sample_size)
        dataset = []
        
        print(f"\nGenerating queries using {config.chat_provider}...")
        
        for i, idx in enumerate(indices):
            doc_text = all_data['documents'][idx]
            doc_meta = all_data['metadatas'][idx]
            
            query = await self.generate_synthetic_query(doc_text, doc_meta)
            
            # Save metadata keys, NOT the ID.
            entry = {
                "file_path": doc_meta.get('file_path'),
                "name": doc_meta.get('name'),
                "type": doc_meta.get('type'),
                "query": query,
                "snippet_preview": doc_text[:100]
            }
            dataset.append(entry)
            print(f"   [{i+1}/{len(indices)}] Generated: {query}")
            
            # Add delay to avoid rate limits (5 seconds)
            await asyncio.sleep(5)

        # 4. Save to File
        os.makedirs(DATASET_DIR, exist_ok=True)
        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Dataset saved to {DATASET_FILE} ({len(dataset)} items)")
        print("   You can now run evaluation without LLM costs using: --mode evaluate")

    def run_evaluation(self, top_k: int):
        """Loads the dataset and evaluates search performance."""
        if not os.path.exists(DATASET_FILE):
            print(f"❌ Dataset file not found: {DATASET_FILE}")
            print("   Run with '--mode generate' first.")
            return

        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"\n🚀 Starting Evaluation (N={len(dataset)}, Top-K={top_k})")
        print(f"   Using Golden Dataset: {DATASET_FILE}")
        
        results = []
        
        for i, item in enumerate(dataset):
            target_file = item['file_path']
            target_name = item['name']
            query = item['query']
            
            # 1. Resolve Current ID (Handle re-indexing)
            target_id = self.get_current_id_by_metadata(target_file, target_name)
            
            if not target_id:
                print(f"   ⚠️  Target skipped (Not found in current DB): {target_name}")
                continue
                
            # 2. Perform Search
            search_res = self.store.query(query, n_results=top_k)
            retrieved_ids = search_res['ids'][0] if search_res['ids'] else []
            
            # 3. Check Rank
            hit = False
            rank = -1
            if target_id in retrieved_ids:
                hit = True
                rank = retrieved_ids.index(target_id) + 1
                
            results.append({"hit": hit, "rank": rank})
            
            status = f"✅ HIT (Rank {rank})" if hit else "❌ MISS"
            print(f"   [{i+1}/{len(dataset)}] {status} | Q: {query}")

        # 4. Report
        total = len(results)
        if total == 0:
            print("No valid targets found in DB.")
            return

        hits = sum(1 for r in results if r['hit'])
        mrr = sum(1/r['rank'] for r in results if r['hit']) / total
        
        print("\n" + "="*60)
        print("📊 Evaluation Summary (Golden Dataset)")
        print("="*60)
        print(f"Total Evaluated : {total}")
        print(f"Hit Rate        : {hits/total:.2%} (Recall@{top_k})")
        print(f"MRR             : {mrr:.4f}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Code Search Quality")
    parser.add_argument("--mode", type=str, required=True, choices=['generate', 'evaluate'], help="Mode: generate new dataset or evaluate existing one")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--k", type=int, default=5, help="Top-K results to consider")
    
    args = parser.parse_args()
    
    evaluator = SearchEvaluator()
    
    if args.mode == 'generate':
        # Async run for generation
        asyncio.run(evaluator.generate_dataset(args.n))
    else:
        # Sync run for evaluation
        evaluator.run_evaluation(args.k)