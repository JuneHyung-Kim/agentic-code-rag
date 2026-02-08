# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Code RAG Agent** - A semantic code search and understanding system for large codebases. Uses tree-sitter for AST parsing, ChromaDB for vector storage, NetworkX for call graphs, and hybrid search (semantic + BM25) to enable natural language queries over C, C++, and Python code.

**Primary Use Case**: Multi-agent system component serving as a "Code Knowledge Expert" for navigating large codebases.

## Development Commands

```bash
# Install
pip install -e .           # Basic install
pip install -e ".[dev]"    # With dev dependencies

# Configure
cp .env.example .env       # Edit with API keys

# Run
python src/cli.py index /path/to/codebase
python src/cli.py search "query" --alpha 0.7 --project /filter/path
python src/cli.py chat
python src/cli.py reset

# Code quality
black src/
mypy src/
flake8 src/
pytest tests/
pytest tests/test_parser.py -v

# Development utilities
python scripts/verify_parser.py /path/to/file.c
python scripts/evaluate_search.py
```

## Architecture

### LangGraph Agent Flow

The agent uses LangGraph for stateful multi-step reasoning:

```
User Query → Planner → Executor → Refinery ─┬─→ Synthesizer → Response
                ↑                           │
                └───── CONTINUE ────────────┘
```

**Nodes** (`agent/nodes.py`):
- `plan_node`: Creates research plan from query and existing findings
- `execute_node`: Runs tools (search, file listing, related code) for current step
- `refine_node`: Decides CONTINUE (more research needed) or FINISH
- `synthesize_node`: Generates final answer from accumulated findings

**State** (`agent/state.py`): `AgentState` TypedDict with `input`, `chat_history`, `plan`, `current_step`, `findings`, `response`, `loop_decision`

### Multi-Strategy Indexing

Indexing uses the Strategy pattern with three parallel strategies:

```
CodeIndexer
    ├── VectorStrategy  → VectorStore (ChromaDB)
    ├── KeywordStrategy → KeywordStore (BM25)
    └── GraphStrategy   → GraphStore (NetworkX)
```

All strategies receive the same `CodeNode` list with pre-assigned IDs, ensuring consistent cross-store references.

**Storage persistence**:
- Vector data: `./db/` (ChromaDB)
- Graph data: `./db/graph_store.pkl`
- Keyword index: `./db/keyword_store.pkl`
- File registry: `./db/index_registry.json`

### Key Singletons

Access via factory functions to ensure single instances:
- `get_vector_store()` - ChromaDB connection
- `get_keyword_store()` - BM25 index
- `get_graph_store()` - NetworkX DiGraph
- `get_search_engine()` - Hybrid search orchestrator
- `get_model()` - LLM instance (`agent/model.py`)
- `tool_registry` - Tool registry singleton (`agent/tools.py`)

Reset functions available for testing: `reset_model()` (`agent/model.py`), `tool_registry.reset()` (`agent/tools.py`)

### Module Map

| Module | Purpose |
|--------|---------|
| `cli.py` | CLI entry point (index, search, chat, reset) |
| `agent/graph.py` | LangGraph workflow definition |
| `agent/nodes.py` | Node implementations (plan, execute, refine, synthesize) |
| `agent/model.py` | LLM singleton factory (gemini/ollama) |
| `agent/prompts/` | Prompt package: YAML templates, loader, Pydantic output schemas |
| `agent/tools.py` | ToolRegistry: tool metadata, singleton management, dispatch |
| `indexing/indexer.py` | Orchestrates multi-strategy indexing |
| `indexing/strategies/` | Vector, Keyword, Graph indexing strategies |
| `indexing/storage/` | VectorStore, KeywordStore, GraphStore |
| `retrieval/search_engine.py` | Hybrid search with score fusion |
| `tools/` | Agent tools: SearchTool, FileSystemTools, RelatedCodeTool |

### Data Flow

**Indexing**: File → Parser → CodeNode[] → ID assignment → Strategy.index() → Stores

**Search**: Query → VectorStore.search(2k) → KeywordStore.search() → Score fusion (alpha blend) → Top-n results

**Agent**: Query → Planner (LLM) → Plan steps → Executor (tools) → Refinery (LLM decision) → Loop or Synthesize

## Provider Configuration

Embedding and chat providers can be mixed independently:

```bash
# All local (Ollama)
EMBEDDING_PROVIDER=ollama
CHAT_PROVIDER=ollama

# Cloud
EMBEDDING_PROVIDER=gemini
CHAT_PROVIDER=gemini

# Hybrid
EMBEDDING_PROVIDER=default  # ChromaDB's Sentence Transformers
CHAT_PROVIDER=ollama
```

**Chat providers**: `gemini` and `ollama` (see `agent/model.py:get_model()`)
**Embedding providers**: `gemini`, `ollama`, `default` (ChromaDB's Sentence Transformers)

## Incremental Indexing

File registry tracks SHA1 hashes per project. On re-index:
1. Detect added/modified/deleted files
2. Delete old entries from all stores (vector, keyword, graph)
3. Index new/modified files through all strategies
4. Persist stores to disk

**Force full reindex**: Delete `./db/` or run `python src/cli.py reset`

**Schema changes**: Increment `SCHEMA_VERSION` in `indexing/file_registry.py`, then reset and reindex.

## Adding Features

### New Language Parser
1. Create `src/indexing/parsers/{lang}_parser.py`
2. Implement tree-sitter traversal returning `List[CodeNode]`
3. Register extension mapping in `src/indexing/parser.py`
4. Add `tree-sitter-{lang}` to `pyproject.toml`

### New Indexing Strategy
1. Create `src/indexing/strategies/{name}_strategy.py`
2. Extend `BaseStrategy` with `index()` and `delete()` methods
3. Add to `CodeIndexer.strategies` list in `indexing/indexer.py`

### New Agent Tool
1. Create tool class in `src/tools/`
2. Register in `ToolRegistry._register_builtins()` (`agent/tools.py`) with handler
3. For advanced routing, replace heuristics with LLM-based tool selection

## Debugging

```bash
# Check indexed files
cat ./db/index_registry.json | python -m json.tool

# Test parser on single file
python scripts/verify_parser.py /path/to/file.c

# Search with different alpha values
python src/cli.py search "query" --alpha 1.0   # Vector only
python src/cli.py search "query" --alpha 0.0   # Keyword only
python src/cli.py search "query" --alpha 0.7   # Hybrid (default)
```
