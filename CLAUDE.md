# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Code RAG Agent** - a specialized AI system for semantic code search and understanding of large codebases. It uses tree-sitter for AST parsing, ChromaDB for vector storage, and hybrid search (semantic + BM25) to enable natural language queries over C, C++, and Python code.

**Primary Use Case**: Multi-agent system component that serves as a "Code Knowledge Expert" for understanding and navigating large codebases.

## Development Commands

### Installation & Setup
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys and provider settings
```

### Running the Application
```bash
# Index a codebase
python src/main.py index /path/to/codebase

# Search indexed code
python src/main.py search "your natural language query"

# Interactive chat mode
python src/main.py chat

# Reset database
python src/main.py reset
```

### Code Quality & Testing
```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
flake8 src/

# Run tests
pytest tests/

# Run specific test
pytest tests/test_parser.py
pytest tests/test_integration.py
```

### Development Scripts
```bash
# Verify parser functionality (useful during development)
python scripts/verify_parser.py /path/to/code/file.c

# Evaluate search quality
python scripts/evaluate_search.py
```

## Architecture

### High-Level Component Flow

```
CLI (main.py)
    ↓
┌─────────────────────────────────────────┐
│ Application Layer                       │
│ - CodeAgent (agent/core.py)            │
│ - CodeIndexer (indexing/indexer.py)    │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Business Logic Layer                    │
│ - SearchEngine (retrieval/)             │
│ - CodeParser (indexing/parser.py)       │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Data Access Layer (Singleton Pattern)   │
│ - VectorStore (storage/vector_store.py) │
│ - KeywordStore (storage/keyword_store.py)│
│ - FileRegistry (indexing/file_registry.py)│
└─────────────────────────────────────────┘
```

### Key Architectural Patterns

**Singleton Pattern**: VectorStore and SearchEngine use singletons to ensure single ChromaDB connection and maintain consistent BM25 index across requests. Access via `get_vector_store()` and `get_search_engine()` factory functions.

**Strategy Pattern**: Language-specific parsing via `PythonParser` and `CppParser`. Each parser handles language-specific AST extraction with tree-sitter.

**DAO Pattern**: `VectorStore` encapsulates all ChromaDB operations, abstracting embedding function selection and vector operations.

**Incremental Indexing**: `FileRegistry` tracks file state (SHA1 hash, mtime, size) to enable delta-based reindexing. Only changed files are reprocessed unless schema version or root path changes.

### Module Responsibilities

- **`agent/core.py`**: AI orchestration with OpenAI, Gemini, or Ollama. Manages tool calling and conversation history.
- **`indexing/indexer.py`**: Coordinates indexing pipeline, file discovery, and incremental update logic.
- **`indexing/parser.py`**: Language dispatcher that routes to specialized parsers based on file extension.
- **`indexing/parsers/python_parser.py`**: Extracts functions, classes, methods with docstrings and type hints using tree-sitter.
- **`indexing/parsers/cpp_parser.py`**: Extracts functions, structs, classes, enums, typedefs, macros from C/C++ code.
- **`indexing/file_registry.py`**: JSON-based registry for incremental indexing change tracking.
- **`retrieval/search_engine.py`**: Hybrid search combining vector similarity (ChromaDB) and keyword matching (BM25) with weighted fusion.
- **`storage/vector_store.py`**: ChromaDB wrapper with multi-provider embedding support (OpenAI, Gemini, Ollama, default).
- **`storage/keyword_store.py`**: BM25-based keyword index for lexical search.
- **`tools/search_tool.py`**: Agent tool interface for semantic code search.
- **`config.py`**: Centralized configuration with provider validation and .env loading.

### Data Structures

**CodeNode**: Core data structure representing parsed code elements.
```python
@dataclass
class CodeNode:
    type: str           # 'function', 'class', 'struct', 'method', etc.
    name: str           # Identifier name
    file_path: str      # Relative path from project root
    start_line: int     # 0-indexed
    end_line: int       # 0-indexed
    content: str        # Raw source code
    language: str       # 'python', 'c', 'cpp'
    docstring: str      # Documentation
    arguments: List[str]
    return_type: str
    signature: str
    parent_name: str    # Class name for methods
    imports: List[str]  # File-level imports/includes
```

**Hybrid Search Algorithm**:
1. Vector search via ChromaDB (semantic similarity, retrieves 2k candidates)
2. BM25 keyword search (lexical matching)
3. Score normalization: vector = 1.0/(1.0+distance), bm25 = score/max_score
4. Weighted fusion: final = (alpha × vector) + ((1-alpha) × bm25), where alpha=0.7 default
5. Sort by final score and return top n results

### Provider Configuration

The system supports flexible provider mixing:

**Embedding Providers** (EMBEDDING_PROVIDER):
- `openai`: OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)
- `gemini`: Google Generative AI embeddings
- `ollama`: Local embeddings (mxbai-embed-large, nomic-embed-text)
- `default`: ChromaDB's default embedding (Sentence Transformers)

**Chat Providers** (CHAT_PROVIDER):
- `openai`: GPT models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
- `gemini`: Gemini models (gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash-exp)
- `ollama`: Local LLMs (llama3.2, qwen2.5, deepseek-r1)

Configuration validated in `config.py` on startup.

## Important Implementation Details

### File Filtering
Indexing automatically skips:
- Hidden directories (starting with `.`)
- Build directories (`build/`, `dist/`)
- Virtual environments (`venv/`, `.venv/`)
- Cache directories (`__pycache__/`)

Supported extensions: `.py`, `.c`, `.cpp`, `.h`, `.hpp`, `.cc`, `.cxx`

### Database Location
ChromaDB stores in `./db/` by default. To reset: `rm -rf db/` or `python src/main.py reset`.

File registry JSON at `./index_registry.json` tracks indexed state.

### Tree-sitter Parsers
C/C++ parser includes defensive filters for:
- Operator overloading (skips `operator[]`, `operator+`, etc.)
- Template syntax edge cases
- Macro definitions (extracts as code nodes)

Python parser extracts:
- Top-level functions with full signatures
- Classes with methods and decorators
- Docstrings and type annotations
- Import statements

### Performance Considerations
- Singleton pattern prevents redundant ChromaDB initialization
- Incremental indexing only processes changed files (SHA1-based diffing)
- BM25 index maintained in-memory for fast keyword matching
- Vector search limited to 2k candidates before BM25 fusion

## Adding New Features

### Adding a New Language Parser
1. Create parser in `src/indexing/parsers/new_language_parser.py`
2. Implement tree-sitter AST traversal and CodeNode extraction
3. Register in `src/indexing/parser.py` language mapping
4. Add tree-sitter language binding to dependencies in `pyproject.toml`

### Adding a New Agent Tool
1. Create tool definition in `src/tools/`
2. Implement tool interface with proper schema
3. Register tool in `src/agent/core.py` tool list
4. Update provider-specific tool calling logic if needed

### Adding a New Provider
1. Extend `config.py` with provider validation
2. Add provider-specific initialization in `agent/core.py` or `storage/vector_store.py`
3. Add required dependencies to `pyproject.toml`
4. Update `.env.example` with provider configuration

## Common Workflows

### Debugging Indexing Issues
1. Check file registry: `cat index_registry.json` to see indexed files
2. Verify parser output: `python scripts/verify_parser.py /path/to/file.c`
3. Enable verbose logging in indexer
4. Check ChromaDB collection: inspect `db/` directory

### Testing Search Quality
1. Index a test codebase
2. Run `python scripts/evaluate_search.py` for systematic evaluation
3. Adjust hybrid search alpha parameter in `retrieval/search_engine.py`
4. Compare vector-only (alpha=1.0) vs hybrid (alpha=0.7) vs keyword-only (alpha=0.0)

### Updating to New Schema
When changing CodeNode structure or metadata:
1. Increment `SCHEMA_VERSION` in `indexing/file_registry.py`
2. Clear database: `python src/main.py reset`
3. Reindex: `python src/main.py index /path/to/codebase`

## Configuration Examples

**All Local (Free, Private)**:
```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=mxbai-embed-large
CHAT_PROVIDER=ollama
CHAT_MODEL=qwen2.5:14b
OLLAMA_BASE_URL=http://localhost:11434
```

**Cloud Performance**:
```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
CHAT_PROVIDER=openai
CHAT_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

**Hybrid Cost-Effective**:
```bash
EMBEDDING_PROVIDER=default
CHAT_PROVIDER=gemini
CHAT_MODEL=gemini-1.5-flash
GEMINI_API_KEY=AIz...
```

## Documentation

- **USER_GUIDE.md**: Installation, configuration, and usage instructions
- **DEVELOPER_GUIDE.md**: High-level architecture and extension points
- **INDEXING_AND_PARSING.md**: Deep dive into parsing pipeline and AST extraction
- **API_REFERENCE.md**: Programmatic API documentation
- **ROADMAP.md**: Future features and improvements
