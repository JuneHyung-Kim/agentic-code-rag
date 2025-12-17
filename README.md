# OS Devel Agent

An AI Agent designed to understand and assist with Operating System development using Code RAG (Retrieval-Augmented Generation).

## 🌟 Features

- **Code Indexing**: Parses C, C++, and Python code using Tree-sitter to understand code structure.
- **Semantic Search**: Uses vector embeddings (OpenAI or Gemini) to find relevant code snippets.
- **AI Chat Agent**: Interactive chat interface to ask questions about the codebase.
- **Multi-Provider Support**: Supports both OpenAI (GPT-4o) and Google Gemini (Flash/Pro).

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JuneHyung-Kim/os-devel-agent.git
   cd os-devel-agent
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   Copy `.env.example` to `.env` and set your API keys.
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env`:
   ```ini
   # Choose your provider: openai or gemini
   MODEL_PROVIDER=gemini
   
   # Set your API Key
   GEMINI_API_KEY=your_api_key_here
   # OPENAI_API_KEY=your_api_key_here
   ```

## 🚀 Usage

### 1. Index a Project
Before searching or chatting, you need to index the target codebase.

```bash
python src/main.py index /path/to/your/project
```

### 2. Search Code
Perform a semantic search on the indexed codebase.

```bash
python src/main.py search "How is memory management implemented?"
```

### 3. Chat with Agent
Start an interactive session with the AI Agent.

```bash
python src/main.py chat
```

## 📁 Project Structure

```
src/
├── agent/              # AI Agent Logic
│   ├── core.py         # Main Agent class (OpenAI/Gemini integration)
│   └── __init__.py
├── indexing/           # Code Indexing Pipeline
│   ├── indexer.py      # File traversal and indexing orchestration
│   ├── parser.py       # Tree-sitter code parser
│   └── vector_store.py # ChromaDB vector storage
├── tools/              # Agent Tools
│   ├── search_tool.py  # Tool for agent to search code
│   └── filesystem.py   # File system operations
├── config.py           # Configuration management
└── main.py             # CLI Entry point
```

## 📝 License

MIT

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from pathlib import Path
from code_rag.indexing.builder import IndexBuilder
from code_rag.indexing.embedder import CodeEmbedder
from code_rag.retrieval.vector_store import ChromaVectorStore
from code_rag.retrieval.retrievers import HybridRetriever
from code_rag.context.builder import ContextBuilder, create_llm_prompt

# 1. Build index from repository
builder = IndexBuilder(repo_path=Path("/path/to/repo"))
chunks = builder.build()

# 2. Setup vector store and retriever
vector_store = ChromaVectorStore()
vector_store.add_chunks(chunks)

embedder = CodeEmbedder()
retriever = HybridRetriever(vector_store, chunks)

# 3. Search
query = "how to implement authentication?"
query_embedding = embedder.embed_query(query)
results = retriever.retrieve(query, query_embedding, top_k=5)

# 4. Build LLM context
context_builder = ContextBuilder()
context = context_builder.build(results, query=query)

# 5. Create LLM prompt
prompt = create_llm_prompt(context, query)

# 6. Send to LLM
response = llm.complete(prompt)  # Your LLM API call
```

## 🔄 Three Main Pipelines

### 1. Indexing Pipeline
```python
Source Code → Parse (Tree-sitter) → Chunks → Embeddings → Vector Store
```

- **Parser**: Extracts semantic chunks from C/C++/Python code
- **Embedder**: Generates embeddings using CodeBERT
- **Storage**: Persists chunks and vectors in ChromaDB

### 2. Retrieval Pipeline
```python
User Query → Embed Query → Search Vector Store → Re-rank → Top Results
```

- **Semantic Search**: Vector similarity based on embeddings
- **Keyword Search**: Symbol name and text matching
- **Hybrid**: Combines both strategies with configurable weights

### 3. Context Building
```python
Search Results → Format → LLM Prompt
```

- Formats chunks with metadata
- Respects token limits
- Supports Markdown and plain text formats

## ⚙️ Configuration

### Experiment with Indexing

```python
from code_rag.core.models import IndexConfig

config = IndexConfig(
    languages=["c", "cpp", "python"],
    max_chunk_size=5000,
    embedder_model="microsoft/codebert-base",  # Try: "microsoft/graphcodebert-base"
    batch_size=32,
    vector_db="chromadb",
    persist_dir="./vector_db",
)

builder = IndexBuilder(repo_path, config=config)
```

### Experiment with Retrieval

```python
from code_rag.core.models import RetrievalConfig

config = RetrievalConfig(
    strategy="hybrid",      # Try: "semantic", "keyword"
    top_k=10,
    semantic_weight=0.7,    # Adjust these weights
    keyword_weight=0.3,
    use_rerank=False,       # Enable for better quality
)
```

## 🧪 Easy to Experiment

The system is designed for rapid experimentation:

```python
# Try different embedding models
embedders = [
    CodeEmbedder("microsoft/codebert-base"),
    CodeEmbedder("microsoft/graphcodebert-base"),
]

# Try different retrieval strategies
retrievers = [
    SemanticRetriever(vector_store),
    KeywordRetriever(chunks),
    HybridRetriever(vector_store, chunks, 0.5, 0.5),
]

# Try different context formats
formats = ["markdown", "plain"]

# Easy to measure which combination works best
for embedder in embedders:
    for retriever in retrievers:
        for format in formats:
            results = evaluate(embedder, retriever, format)
```

## 🔮 Roadmap

- [ ] **Re-ranking**: Add cross-encoder based re-ranking for better quality
- [ ] **Caching**: Avoid re-embedding identical queries
- [ ] **Call Graph**: Index function call relationships
- [ ] **Type Information**: Extract and index type signatures
- [ ] **CLI Tool**: Command-line interface for indexing and searching
- [ ] **Web UI**: Browser-based search interface
- [ ] **More Languages**: Java, Go, Rust, etc.
- [ ] **FAISS Backend**: Faster vector search for large indexes

## 🤝 Design Philosophy

1. **Simple**: ~600 lines of core code, each file ~50-150 lines
2. **Focused**: Indexing, retrieval, context building - that's it
3. **Extensible**: Easy to swap components (embedders, retrievers, vector DBs)
4. **Practical**: Built for LLM integration, not just academic exploration
5. **Experimental**: Designed to support A/B testing different strategies

## 📖 Example Output

```
Query: how to implement authentication?

─────────────────────────────────────────────────────────────────
[function] verify_token (score: 0.89)
Location: src/auth.cpp:45-67
Language: cpp

bool verify_token(const string& token) {
    // Validate JWT token signature
    auto decoded = jwt::decode<jwt::traits::nlohmann_json>(token);
    ...
}
─────────────────────────────────────────────────────────────────

[function] generate_token (score: 0.87)
Location: src/auth.cpp:10-40
Language: cpp

string generate_token(const User& user) {
    auto token = jwt::create<jwt::traits::nlohmann_json>()
        .set_subject(user.id)
        ...
}
```

## 📝 License

MIT

---

**Build better AI systems with code understanding.** ✨
