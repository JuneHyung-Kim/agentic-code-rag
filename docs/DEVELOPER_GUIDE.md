# Developer Guide

## 📖 Table of Contents

- [Introduction](#introduction)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Understanding the Codebase](#understanding-the-codebase)
- [Extending the Agent](#extending-the-agent)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Introduction

This guide is designed for developers who want to understand, modify, or extend the OS Devel Agent codebase. It provides a deep dive into the architecture, design decisions, and implementation details.

### Design Philosophy

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Extensibility**: Easy to add new languages, providers, or tools
3. **Simplicity**: Minimal dependencies, clear interfaces
4. **Performance**: Efficient indexing and search for large codebases

---

## Architecture Overview

### High-Level Architecture

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

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **main.py** | CLI entry point, command routing |
| **Indexer** | File traversal, orchestrating indexing pipeline |
| **Parser** | Code parsing using Tree-sitter |
| **VectorStore** | Embedding generation and similarity search |
| **Agent Core** | AI chat logic, tool calling orchestration |
| **SearchTool** | Code search interface for the agent |
| **Config** | Configuration management |

---

## Project Structure

```
project-root/
│
├── src/                          # Source code
│   ├── main.py                   # CLI entry point
│   ├── config.py                 # Configuration management
│   │
│   ├── agent/                    # AI Agent logic
│   │   ├── __init__.py
│   │   └── core.py               # CodeAgent class (OpenAI/Gemini/Ollama)
│   │
│   ├── indexing/                 # Code indexing pipeline
│   │   ├── __init__.py
│   │   ├── indexer.py            # CodeIndexer (file traversal)
│   │   ├── parser.py             # CodeParser (Tree-sitter)
│   │   └── vector_store.py       # VectorStore (ChromaDB)
│   │
│   ├── tools/                    # Agent tools
│   │   ├── __init__.py
│   │   └── search_tool.py        # SearchTool for code search
│   │
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       ├── logger.py             # Logging configuration
│       └── ollama_embedding.py   # Custom Ollama embeddings
│
├── docs/                         # Documentation
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   └── API_REFERENCE.md
│
├── db/                           # ChromaDB vector database (created at runtime)
│   └── chroma.sqlite3
│
├── .env.example                  # Environment variable template
├── pyproject.toml               # Project metadata and dependencies
└── README.md                     # Project overview
```

---

## Core Components

### 1. Configuration (`src/config.py`)

**Purpose**: Centralized configuration management

**Key Features:**
- Loads environment variables from `.env`
- Validates API keys based on provider
- Single source of truth for all settings
- Support for split providers (chat vs embedding)

**Code Structure:**
```python
class AgentConfig:
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Configuration
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "default")
        self.chat_provider = os.getenv("CHAT_PROVIDER", "gemini")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def validate_chat_config(self) -> None:
        # Validation logic for chat provider

    def validate_embedding_config(self) -> None:
        # Validation logic for embedding provider
```

**Design Decision**: Singleton pattern via module-level `config` instance

### 2. Code Parser (`src/indexing/parser.py`)

**Purpose**: Extract structured information from source code using Tree-sitter

**Supported Languages:**
- Python (`.py`)
- C (`.c`, `.h`)
- C++ (`.cpp`, `.hpp`, `.cc`, `.cxx`)

**What Gets Extracted:**
- Function definitions
- Class definitions
- Struct definitions

**How It Works:**

```python
# 1. Initialize parsers for each language
self.parsers['python'] = Parser(Language(tree_sitter_python.language()))
self.parsers['c'] = Parser(Language(tree_sitter_c.language()))
self.parsers['cpp'] = Parser(Language(tree_sitter_cpp.language()))

# 2. Parse code into AST
tree = self.parsers[language].parse(bytes(code, "utf8"))

# 3. Query AST for specific patterns
query = Query(language, "(function_definition) @function")
captures = cursor.captures(root_node)

# 4. Extract metadata
for node, name in captures:
    definition = {
        'type': name,  # function, class, struct
        'name': extract_name(node),
        'start_line': node.start_point[0],
        'end_line': node.end_point[0],
        'content': code[node.start_byte:node.end_byte]
    }
```

**Tree-sitter Queries:**

| Language | Query Pattern |
|----------|--------------|
| Python | `(function_definition) @function`, `(class_definition) @class` |
| C | `(function_definition) @function`, `(struct_specifier) @struct` |
| C++ | `(function_definition) @function`, `(class_specifier) @class`, `(struct_specifier) @struct` |

**Design Decisions:**
- **Why Tree-sitter?** Fast, incremental parsing; supports many languages
- **Why extract definitions only?** Reduces noise, focuses on important code structures
- **Content storage**: Full function/class body stored for context

### 3. Code Indexer (`src/indexing/indexer.py`)

**Purpose**: Orchestrate the indexing pipeline

**Process Flow:**

```
1. Walk directory tree
   ↓
2. Filter files by extension (.py, .c, .cpp, etc.)
   ↓
3. Skip hidden/build directories
   ↓
4. For each file:
   a. Read file content
   b. Parse with CodeParser
   c. Extract definitions
   d. Generate unique IDs
   e. Store in VectorStore
   ↓
5. Report progress
```

**Skipped Directories:**
- Hidden directories (`.git`, `.venv`)
- Build artifacts (`build/`, `dist/`)
- Cache directories (`__pycache__/`)
- Virtual environments (`venv/`)

**ID Generation:**
```python
# Format: file_path:function_name:index
id = f"{rel_path}:{definition['name']}:{i}"
# Example: "src/kernel/sched.c:schedule:0"
```

**Error Handling:**
- Encoding errors: Skip file with warning
- Parse errors: Skip file with error log
- Graceful degradation: Indexing continues even if some files fail

**Design Decisions:**
- **Relative paths**: Portability across machines
- **Unique IDs**: Prevent duplicate entries, enable updates
- **Batch processing**: Could be optimized for large codebases

### 4. Vector Store (`src/indexing/vector_store.py`)

**Purpose**: Manage embeddings and similarity search using ChromaDB

**Architecture:**

```
VectorStore
    ↓
ChromaDB (Persistent)
    ↓
Embedding Function (OpenAI, Gemini, Ollama, or Default)
    ↓
SQLite Database (./db/chroma.sqlite3)
```

**Embedding Providers:**

| Provider | Model | Dimensions | Use Case |
|----------|-------|------------|----------|
| OpenAI | `text-embedding-3-small` | 1536 | High quality, paid |
| Gemini | `embedding-001` | 768 | Free tier available |
| Ollama | `mxbai-embed-large` | 1024 | Free, private, local |
| Default | Sentence Transformers | 384 | Offline, no API key |

**Key Operations:**

```python
# Add documents
vector_store.add_documents(
    documents=["def foo(): ..."],
    metadatas=[{'file': 'src/foo.py', 'type': 'function', 'name': 'foo'}],
    ids=["src/foo.py:foo:0"]
)

# Query
results = vector_store.query(
    query_text="memory allocation functions",
    n_results=5
)
# Returns: {
#   'ids': [...],
#   'distances': [...],
#   'metadatas': [...],
#   'documents': [...]
# }
```

**Metadata Schema:**
```python
{
    'file_path': str,      # Relative path to file
    'name': str,           # Function/class name
    'type': str,           # function, class, struct
    'start_line': int,     # Starting line number
    'end_line': int,       # Ending line number
    'language': str        # python, c, cpp
}
```

**Design Decisions:**
- **Persistent storage**: Database survives restarts
- **Multiple embedding options**: Flexibility for users
- **Automatic fallback**: Works offline with default embeddings

### 5. Agent Core (`src/agent/core.py`)

**Purpose**: AI chat interface with tool calling

**Architecture:**

```
CodeAgent
    ↓
┌──────────────┬──────────────┬──────────────┐
│              │              │              │
OpenAI API     Gemini API     Ollama (Via    SearchTool
(Function      (Function      OpenAI API)    (Vector Search)
Calling)       Calling)
```

**Conversation Flow:**

```
User Input
    ↓
Add to message history
    ↓
Send to AI (with tool definitions)
    ↓
AI Response (may include tool calls)
    ↓
Execute tool(s) → search_codebase()
    ↓
Add tool results to message history
    ↓
Send back to AI
    ↓
Final Response to User
```

**Provider-Specific Implementation:**

**OpenAI / Ollama:**
```python
# Send message with tools
response = client.chat.completions.create(
    model="gpt-4o",  # or llama3.2
    messages=self.messages,
    tools=[search_tool_definition],
    tool_choice="auto"
)

# Handle tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        # Add result to messages
```

**Gemini:**
```python
# Configure model with tools
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[search_codebase_function],
    system_instruction="You are an expert software engineer..."
)

# Start chat with automatic function calling
chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message(user_input)
```

**Design Decisions:**
- **Provider abstraction**: Switch providers without changing interface
- **Automatic tool calling**: Gemini handles tool execution internally
- **Message history**: OpenAI/Ollama require manual management
- **System prompts**: Guides the AI to reference code and be specific

### 6. Search Tool (`src/tools/search_tool.py`)

**Purpose**: Bridge between Agent and Vector Store

**Interface:**

```python
def search_codebase(query: str, n_results: int = 5) -> str:
    """
    Tool function that the AI can call to search code.
    
    Returns formatted string with:
    - File path
    - Function/class type and name
    - Line numbers
    - Code content
    """
```

**Tool Definition (OpenAI Format):**
```json
{
    "type": "function",
    "function": {
        "name": "search_codebase",
        "description": "Search the codebase for relevant code snippets",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }
}
```

**Design Decisions:**
- **String return type**: Easier for AI to parse
- **Formatted output**: Consistent structure for AI consumption
- **Provider-agnostic**: Works with both OpenAI, Gemini and Ollama

---

## Data Flow

### Indexing Flow

```
User: python src/main.py index /path/to/project
    ↓
main.py: index_project(path)
    ↓
CodeIndexer.__init__(path)
    ↓
CodeIndexer.index_project()
    ↓
os.walk(root_path) → for each file
    ↓
CodeIndexer._index_file(file_path)
    ↓
CodeParser.extract_definitions(code, language)
    ↓
Tree-sitter: parse → query → extract nodes
    ↓
VectorStore.add_documents(docs, metadata, ids)
    ↓
ChromaDB: embed → store in SQLite
```

### Search Flow

```
User: python src/main.py search "query"
    ↓
main.py: search_code(query)
    ↓
VectorStore.query(query, n_results=5)
    ↓
ChromaDB: embed query → similarity search
    ↓
Return: {ids, distances, metadatas, documents}
    ↓
main.py: format and print results
```

### Chat Flow

```
User: python src/main.py chat
    ↓
main.py: start_chat()
    ↓
CodeAgent.__init__()
    ↓
Initialize provider (OpenAI, Gemini, or Ollama)
Load SearchTool
    ↓
User Input Loop:
    ↓
CodeAgent.chat(user_input)
    ↓
AI Provider: process message
    ↓
[If AI needs info] → Tool Call: search_codebase(query)
    ↓
SearchTool → VectorStore → ChromaDB
    ↓
Return results to AI
    ↓
AI: generate final response
    ↓
Display to user
```

---

## Understanding the Codebase

### Entry Point: `src/main.py`

**CLI Commands:**
- `index <path>`: Calls `index_project(path)`
- `search <query>`: Calls `search_code(query)`
- `chat`: Calls `start_chat()`

**Key Functions:**

```python
def index_project(project_path: str):
    """
    Index a project directory.
    Creates CodeIndexer and runs indexing pipeline.
    """

def search_code(query: str):
    """
    Perform semantic search on indexed code.
    Uses VectorStore directly.
    """

def start_chat():
    """
    Start interactive chat session.
    Creates CodeAgent and enters input loop.
    """
```

### Adding a New Language

To add support for a new language (e.g., Java):

1. **Install Tree-sitter grammar:**
   ```bash
   pip install tree-sitter-java
   ```

2. **Update `src/indexing/parser.py`:**
   ```python
   import tree_sitter_java
   
   # In __init__:
   java_lang = Language(tree_sitter_java.language())
   self.parsers['java'] = Parser(java_lang)
   self.languages['java'] = java_lang
   
   # In extract_definitions:
   elif language == 'java':
       query_scm = """
       (method_declaration) @method
       (class_declaration) @class
       """
   ```

3. **Update `src/indexing/indexer.py`:**
   ```python
   # In _index_file:
   elif ext == '.java':
       lang = 'java'
   ```

### Adding a New Tool

To add a file reading tool:

1. **Create tool file** `src/tools/file_tool.py`:
   ```python
   class FileTool:
       def read_file(self, file_path: str) -> str:
           """Read and return file contents."""
           with open(file_path, 'r') as f:
               return f.read()
       
       def get_tool_definition(self):
           return {
               "type": "function",
               "function": {
                   "name": "read_file",
                   "description": "Read a file from the codebase",
                   "parameters": {...}
               }
           }
   ```

2. **Register in `src/agent/core.py`:**
   ```python
   from tools.file_tool import FileTool
   
   def __init__(self):
       self.file_tool = FileTool()
       self.tools = [
           self.search_tool.get_tool_definition(),
           self.file_tool.get_tool_definition()
       ]
   ```

3. **Handle tool call:**
   ```python
   # In _chat_openai:
   elif function_name == "read_file":
       tool_output = self.file_tool.read_file(**function_args)
   ```

---

## Development Setup

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

### Code Formatting

```bash
# Auto-format code
black src/

# Check formatting
black --check src/
```

### Type Checking

```bash
# Run type checker
mypy src/
```

### Linting

```bash
# Run linter
flake8 src/
```

---

## Testing

### Manual Testing

**Test Indexing:**
```bash
# Create test project
mkdir test_project
echo "def hello(): print('world')" > test_project/test.py

# Index it
python src/main.py index test_project

# Verify database
ls -lh db/
```

**Test Search:**
```bash
python src/main.py search "print function"
```

**Test Chat:**
```bash
python src/main.py chat
# Try: "Show me the hello function"
```

### Unit Testing (Future)

Recommended structure:
```
tests/
├── test_config.py          # Test configuration loading
├── test_parser.py          # Test Tree-sitter parsing
├── test_indexer.py         # Test file indexing
├── test_vector_store.py    # Test embedding and search
└── test_agent.py           # Test agent interactions
```

---

## Contributing

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Make changes** and test thoroughly
4. **Format code**: `black src/`
5. **Commit**: `git commit -m "Add feature X"`
6. **Push**: `git push origin feature/my-feature`
7. **Create Pull Request**

### Code Style Guidelines

- **PEP 8** for Python code
- **Type hints** for all functions
- **Docstrings** for classes and public methods
- **Comments** for complex logic
- **Error handling** with try-except where appropriate

### Areas for Contribution

- ✅ Add more language support (Rust, Go, Java, etc.)
- ✅ Implement incremental indexing
- ✅ Add more tools (file read, write, execute)
- ✅ Improve error handling and logging
- ✅ Create comprehensive tests
- ✅ Optimize indexing performance
- ✅ Add CLI progress bars
- ✅ Implement configuration file support

---

## Additional Resources

- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Gemini Function Calling Guide](https://ai.google.dev/docs/function_calling)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

---

**Happy developing!** 🔧
