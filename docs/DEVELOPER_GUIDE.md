# Developer Guide

This guide provides a high-level view of the OS Devel Agent codebase, architecture, and workflows.
For deep, implementation-level details of the indexing and parsing pipeline, see `docs/INDEXING_AND_PARSING.md`.

## Table of Contents

- Introduction
- Architecture Overview
- Project Structure
- Core Components (Summary)
- Data Flow (Summary)
- Understanding the Codebase
- Extending the Agent
- Development Setup
- Testing
- Contributing

---

## Introduction

This guide is for developers who want to understand, modify, or extend the OS Devel Agent.
The focus here is on the overall architecture and entry points rather than internal indexing details.

### Design Philosophy

1. Modularity: Each component has a single, well-defined responsibility.
2. Extensibility: Easy to add new languages, providers, or tools.
3. Simplicity: Minimal dependencies, clear interfaces.
4. Performance: Efficient indexing and search for large codebases.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                         (main.py)                           │
│                    CLI: index, search, chat                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────┐                    ┌──────────────┐
             │             │                    │              │
        ┌────▼────┐   ┌────▼────┐         ┌────▼────┐   ┌─────▼────┐
        │ Indexer │   │  Agent  │         │  Search │   │  Vector  │
        │         │   │  (Core) │◄────────┤  Tool   │◄──┤  Store   │
        └────┬────┘   └────┬────┘         └─────────┘   └──────────┘
             │             │
        ┌────▼────┐   ┌────▼────────┐
        │ Parser  │   │  AI Provider │
        │(Tree-   │   │ (OpenAI/    │
        │ sitter) │   │  Gemini/    │
        └─────────┘   │  Ollama)    │
                      └─────────────┘
```

---

## Project Structure

```
project-root/
│
├── src/
│   ├── main.py                   # CLI entry point
│   ├── config.py                 # Configuration management
│   ├── agent/                    # AI agent logic
│   ├── indexing/                 # Indexing and parsing pipeline
│   ├── tools/                    # Agent tools
│   └── utils/                    # Shared utilities
│
├── docs/
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   ├── INDEXING_AND_PARSING.md
│   └── API_REFERENCE.md
│
├── db/                           # ChromaDB persistent data (runtime)
├── README.md
└── ROADMAP.md
```

---

## Core Components (Summary)

- `src/main.py`: CLI entry point and command routing.
- `src/agent/core.py`: AI chat orchestration and tool calling.
- `src/indexing/indexer.py`: Orchestrates indexing and persistence.
- `src/indexing/parser.py`: Language dispatch and parser initialization.
- `src/indexing/vector_store.py`: Embedding generation and similarity search.
- `src/tools/search_tool.py`: Tool interface for semantic code search.
- `src/config.py`: Central configuration and provider validation.

For indexing and parsing internals, see `docs/INDEXING_AND_PARSING.md`.

---

## Data Flow (Summary)

### Indexing
`main.py index` -> `CodeIndexer` -> `CodeParser` -> `VectorStore`

### Search
`main.py search` -> `VectorStore.query` -> formatted results

### Chat
`main.py chat` -> `CodeAgent` -> tool calls -> `SearchTool` -> `VectorStore`

---

## Understanding the Codebase

### Entry Point: `src/main.py`

CLI commands:
- `index <path>`: Run the indexing pipeline on a codebase.
- `search <query>`: Query indexed content via semantic search.
- `chat`: Start an interactive chat session.

---

## Extending the Agent

Common extension points:
- Add a new language parser in `src/indexing/parsers/` and register it in `src/indexing/parser.py`.
- Add new tools in `src/tools/` and wire them in `src/agent/core.py`.
- Add new providers by extending `src/config.py` and provider-specific code.

See `docs/INDEXING_AND_PARSING.md` for language parsing details.

---

## Development Setup

```bash
pip install -e ".[dev]"
```

---

## Testing

Manual smoke tests:

```bash
python src/main.py index /path/to/project
python src/main.py search "query"
python src/main.py chat
```

---

## Contributing

Workflow:
1. Create a feature branch.
2. Implement changes and test.
3. Format code with `black src/`.
4. Commit and open a PR.
