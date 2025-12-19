# API Reference

## 📖 Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Indexing Module](#indexing-module)
- [Agent Module](#agent-module)
- [Tools Module](#tools-module)
- [Usage Examples](#usage-examples)

---

## Overview

This document provides detailed API documentation for all public classes and methods in the OS Devel Agent project.

### Installation

```bash
pip install -e .
```

### Basic Import

```python
from config import config
from indexing.indexer import CodeIndexer
from indexing.parser import CodeParser
from indexing.vector_store import VectorStore
from agent.core import CodeAgent
from tools.search_tool import SearchTool
```

---

## Configuration

### `config.AgentConfig`

Global configuration singleton for the application.

#### Attributes

| Attribute | Type | Description | Default |
|-----------|------|-------------|---------|
| `openai_api_key` | `str \| None` | OpenAI API key from environment | `None` |
| `gemini_api_key` | `str \| None` | Gemini API key from environment | `None` |
| `model_provider` | `str` | AI provider: "openai" or "gemini" | `"openai"` |
| `model_name` | `str` | Model name to use | `"gpt-4o"` |
| `project_root` | `str` | Root directory for projects | `"./"` |

#### Properties

##### `is_valid`

```python
@property
def is_valid(self) -> bool
```

Check if the configuration is valid based on the selected provider.

**Returns:**
- `bool`: `True` if the required API key for the selected provider is set

**Example:**
```python
from config import config

if config.is_valid:
    print(f"Using {config.model_provider}")
else:
    print("API key not configured")
```

#### Usage

```python
import os
from dotenv import load_dotenv
from config import config

# Load environment variables
load_dotenv()

# Access configuration
print(f"Provider: {config.model_provider}")
print(f"Model: {config.model_name}")
print(f"Valid: {config.is_valid}")
```

---

## Indexing Module

### `indexing.parser.CodeParser`

Parse source code using Tree-sitter to extract definitions.

#### Constructor

```python
def __init__(self)
```

Initialize the parser with support for Python, C, and C++.

**Example:**
```python
from indexing.parser import CodeParser

parser = CodeParser()
```

#### Methods

##### `parse(code: str, language: str)`

Parse code into an Abstract Syntax Tree (AST).

**Parameters:**
- `code` (`str`): Source code to parse
- `language` (`str`): Language identifier: "python", "c", or "cpp"

**Returns:**
- `tree_sitter.Tree`: Parsed syntax tree

**Raises:**
- `ValueError`: If the language is not supported

**Example:**
```python
code = "def hello(): pass"
tree = parser.parse(code, "python")
print(tree.root_node)
```

##### `extract_definitions(code: str, language: str)`

Extract function, class, and struct definitions from code.

**Parameters:**
- `code` (`str`): Source code to analyze
- `language` (`str`): Language identifier: "python", "c", or "cpp"

**Returns:**
- `list[dict]`: List of definitions, each containing:
  - `type` (`str`): Definition type (function, class, struct)
  - `name` (`str`): Name of the definition
  - `start_line` (`int`): Starting line number (0-indexed)
  - `end_line` (`int`): Ending line number (0-indexed)
  - `content` (`str`): Full source code of the definition

**Example:**
```python
code = """
def foo():
    return 42

class Bar:
    pass
"""

definitions = parser.extract_definitions(code, "python")
for defn in definitions:
    print(f"{defn['type']}: {defn['name']} at line {defn['start_line']}")
# Output:
# function: foo at line 1
# class: Bar at line 4
```

---

### `indexing.indexer.CodeIndexer`

Orchestrate the indexing pipeline for a project directory.

#### Constructor

```python
def __init__(self, root_path: str)
```

Initialize the indexer for a specific project directory.

**Parameters:**
- `root_path` (`str`): Absolute or relative path to the project root

**Example:**
```python
from indexing.indexer import CodeIndexer

indexer = CodeIndexer("/path/to/project")
```

#### Methods

##### `index_project()`

Index all supported files in the project directory.

**Returns:**
- `None`

**Side Effects:**
- Walks the directory tree starting from `root_path`
- Parses all `.py`, `.c`, `.h`, `.cpp`, `.hpp`, `.cc`, `.cxx` files
- Stores extracted definitions in the vector database
- Prints progress to stdout

**Example:**
```python
indexer = CodeIndexer("./my_project")
indexer.index_project()
# Output:
# Indexing project at /path/to/my_project...
# Indexed 15 definitions from src/main.py
# Indexed 8 definitions from src/utils.c
# Indexing complete. Processed 42 files.
```

---

### `indexing.vector_store.VectorStore`

Manage vector embeddings and similarity search using ChromaDB.

#### Constructor

```python
def __init__(self, collection_name: str = "code_chunks", persist_path: str = "./db")
```

Initialize the vector store with a persistent database.

**Parameters:**
- `collection_name` (`str`, optional): Name of the ChromaDB collection (default: "code_chunks")
- `persist_path` (`str`, optional): Path to the database directory (default: "./db")

**Example:**
```python
from indexing.vector_store import VectorStore

# Default configuration
store = VectorStore()

# Custom configuration
store = VectorStore(collection_name="my_code", persist_path="./data/vectors")
```

#### Methods

##### `add_documents(documents: list[str], metadatas: list[dict], ids: list[str])`

Add code snippets to the vector database.

**Parameters:**
- `documents` (`list[str]`): List of code snippets to store
- `metadatas` (`list[dict]`): List of metadata dicts with keys:
  - `file_path` (`str`): Relative file path
  - `name` (`str`): Function/class name
  - `type` (`str`): Definition type (function, class, struct)
  - `start_line` (`int`): Starting line number
  - `end_line` (`int`): Ending line number
  - `language` (`str`): Programming language
- `ids` (`list[str]`): Unique identifiers for each document

**Returns:**
- `None`

**Example:**
```python
store = VectorStore()

documents = [
    "def add(a, b):\n    return a + b",
    "def multiply(a, b):\n    return a * b"
]

metadatas = [
    {'file_path': 'math.py', 'name': 'add', 'type': 'function', 
     'start_line': 1, 'end_line': 2, 'language': 'python'},
    {'file_path': 'math.py', 'name': 'multiply', 'type': 'function', 
     'start_line': 4, 'end_line': 5, 'language': 'python'}
]

ids = ["math.py:add:0", "math.py:multiply:0"]

store.add_documents(documents, metadatas, ids)
```

##### `query(query_text: str, n_results: int = 5)`

Search for similar code snippets using semantic search.

**Parameters:**
- `query_text` (`str`): Natural language query
- `n_results` (`int`, optional): Number of results to return (default: 5)

**Returns:**
- `dict`: Query results with keys:
  - `ids` (`list[list[str]]`): Document IDs
  - `distances` (`list[list[float]]`): Similarity distances
  - `metadatas` (`list[list[dict]]`): Document metadata
  - `documents` (`list[list[str]]`): Document contents

**Example:**
```python
results = store.query("functions that add numbers", n_results=3)

for i, doc in enumerate(results['documents'][0]):
    meta = results['metadatas'][0][i]
    print(f"Found: {meta['name']} in {meta['file_path']}")
    print(f"Code: {doc}\n")
```

---

## Agent Module

### `agent.core.CodeAgent`

AI chat agent with code search capabilities.

#### Constructor

```python
def __init__(self)
```

Initialize the agent with the configured AI provider (OpenAI or Gemini).

**Raises:**
- `ValueError`: If API key is not configured or provider is invalid

**Example:**
```python
from agent.core import CodeAgent

try:
    agent = CodeAgent()
except ValueError as e:
    print(f"Configuration error: {e}")
```

#### Methods

##### `chat(user_input: str)`

Send a message to the agent and get a response.

**Parameters:**
- `user_input` (`str`): User's question or message

**Returns:**
- `str`: Agent's response

**Side Effects:**
- May call the `search_codebase` tool internally
- Updates conversation history

**Example:**
```python
agent = CodeAgent()

response = agent.chat("How does memory allocation work?")
print(response)

# Follow-up question
response = agent.chat("Show me the malloc implementation")
print(response)
```

##### `reset()`

Reset the conversation history.

**Returns:**
- `None`

**Example:**
```python
agent = CodeAgent()

agent.chat("Tell me about the scheduler")
agent.chat("What about memory?")

# Start fresh conversation
agent.reset()
agent.chat("Explain the file system")
```

---

## Tools Module

### `tools.search_tool.SearchTool`

Interface for searching the codebase.

#### Constructor

```python
def __init__(self)
```

Initialize the search tool with a vector store.

**Example:**
```python
from tools.search_tool import SearchTool

search = SearchTool()
```

#### Methods

##### `search_codebase(query: str, n_results: int = 5)`

Search for relevant code snippets.

**Parameters:**
- `query` (`str`): Natural language search query
- `n_results` (`int`, optional): Number of results (default: 5)

**Returns:**
- `str`: Formatted search results

**Example:**
```python
search = SearchTool()

results = search.search_codebase("memory allocation functions", n_results=3)
print(results)
# Output:
# Result 1:
# File: kernel/mm/kmalloc.c
# Type: function
# Name: kmalloc
# Line: 45-67
# Content:
# void *kmalloc(size_t size, gfp_t flags) {
#     ...
# }
```

##### `get_tool_definition()`

Get the OpenAI function calling definition for this tool.

**Returns:**
- `dict`: Tool definition in OpenAI format

**Example:**
```python
search = SearchTool()
tool_def = search.get_tool_definition()

# Use with OpenAI API
import openai
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    tools=[tool_def]
)
```

---

## Usage Examples

### Example 1: Programmatic Indexing

```python
from dotenv import load_dotenv
from indexing.indexer import CodeIndexer

# Load environment
load_dotenv()

# Index a project
indexer = CodeIndexer("/path/to/linux/kernel")
indexer.index_project()

print("Indexing complete!")
```

### Example 2: Direct Search

```python
from dotenv import load_dotenv
from indexing.vector_store import VectorStore

load_dotenv()

# Search without using the agent
store = VectorStore()
results = store.query("TCP connection handling", n_results=5)

for i, doc in enumerate(results['documents'][0]):
    meta = results['metadatas'][0][i]
    print(f"\n=== Result {i+1} ===")
    print(f"File: {meta['file_path']}")
    print(f"Function: {meta['name']}")
    print(f"Lines: {meta['start_line']}-{meta['end_line']}")
    print(f"\n{doc[:200]}...")
```

### Example 3: Interactive Agent

```python
from dotenv import load_dotenv
from agent.core import CodeAgent

load_dotenv()

# Create agent
agent = CodeAgent()

# Ask questions
questions = [
    "What are the main data structures for networking?",
    "How does the scheduler select the next process?",
    "Explain memory allocation in the kernel"
]

for question in questions:
    print(f"\nQ: {question}")
    response = agent.chat(question)
    print(f"A: {response}\n")
    print("-" * 80)
```

### Example 4: Custom Parser Usage

```python
from indexing.parser import CodeParser

parser = CodeParser()

# Parse C code
c_code = """
struct task_struct {
    int pid;
    char name[16];
};

void schedule(void) {
    // Scheduling logic
}
"""

definitions = parser.extract_definitions(c_code, "c")

for defn in definitions:
    print(f"{defn['type'].upper()}: {defn['name']}")
    print(f"Lines: {defn['start_line']}-{defn['end_line']}")
    print(f"Code:\n{defn['content']}\n")
```

### Example 5: Batch Indexing Multiple Projects

```python
from indexing.indexer import CodeIndexer
import os

projects = [
    "/path/to/project1",
    "/path/to/project2",
    "/path/to/project3"
]

for project_path in projects:
    if os.path.exists(project_path):
        print(f"\n{'='*60}")
        print(f"Indexing: {project_path}")
        print('='*60)
        
        indexer = CodeIndexer(project_path)
        indexer.index_project()
    else:
        print(f"Skipping {project_path} (not found)")

print("\nAll projects indexed!")
```

### Example 6: Custom Embedding Provider

```python
from indexing.vector_store import VectorStore
import os

# Force use of OpenAI embeddings
os.environ['OPENAI_API_KEY'] = 'your-key-here'

# Ensure Gemini key is not set
if 'GEMINI_API_KEY' in os.environ:
    del os.environ['GEMINI_API_KEY']

# Create store (will use OpenAI)
store = VectorStore()

# Add documents
store.add_documents(
    documents=["def foo(): pass"],
    metadatas=[{'file_path': 'test.py', 'name': 'foo', 'type': 'function',
                'start_line': 0, 'end_line': 0, 'language': 'python'}],
    ids=["test.py:foo:0"]
)

# Query
results = store.query("foo function")
print(results)
```

---

## Type Definitions

### Definition Dictionary

```python
{
    'type': str,        # 'function', 'class', or 'struct'
    'name': str,        # Name of the definition
    'start_line': int,  # Starting line (0-indexed)
    'end_line': int,    # Ending line (0-indexed)
    'content': str      # Full source code
}
```

### Metadata Dictionary

```python
{
    'file_path': str,   # Relative path to file
    'name': str,        # Function/class name
    'type': str,        # 'function', 'class', or 'struct'
    'start_line': int,  # Starting line (0-indexed)
    'end_line': int,    # Ending line (0-indexed)
    'language': str     # 'python', 'c', or 'cpp'
}
```

### Query Results Dictionary

```python
{
    'ids': list[list[str]],           # Document IDs
    'distances': list[list[float]],   # Similarity scores
    'metadatas': list[list[dict]],    # Document metadata
    'documents': list[list[str]]      # Document contents
}
```

---

## Error Handling

### Common Exceptions

| Exception | When Raised | How to Handle |
|-----------|-------------|---------------|
| `ValueError` | Invalid provider or missing API key | Check configuration |
| `FileNotFoundError` | Project path doesn't exist | Verify path before indexing |
| `UnicodeDecodeError` | Binary file encountered | Automatically skipped by indexer |
| `Exception` | General parsing/embedding errors | Check logs, verify file format |

### Example Error Handling

```python
from agent.core import CodeAgent
from indexing.indexer import CodeIndexer

# Safe agent initialization
try:
    agent = CodeAgent()
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Please set OPENAI_API_KEY or GEMINI_API_KEY in .env")
    exit(1)

# Safe indexing
try:
    indexer = CodeIndexer("/nonexistent/path")
    indexer.index_project()
except FileNotFoundError:
    print("Project path not found")
except Exception as e:
    print(f"Indexing failed: {e}")
```

---

## Advanced Usage

### Custom Collection Names

Use different collections for different projects:

```python
from indexing.vector_store import VectorStore

# Linux kernel
linux_store = VectorStore(collection_name="linux_kernel", persist_path="./db")

# FreeBSD kernel  
freebsd_store = VectorStore(collection_name="freebsd_kernel", persist_path="./db")

# Query specific collection
results = linux_store.query("scheduler")
```

### Programmatic Chat

Build custom interfaces on top of the agent:

```python
from agent.core import CodeAgent

class CodeAssistant:
    def __init__(self):
        self.agent = CodeAgent()
        self.history = []
    
    def ask(self, question: str) -> str:
        response = self.agent.chat(question)
        self.history.append({'question': question, 'answer': response})
        return response
    
    def get_history(self):
        return self.history

assistant = CodeAssistant()
assistant.ask("What is the main scheduler function?")
assistant.ask("How does it select the next process?")

for entry in assistant.get_history():
    print(f"Q: {entry['question']}")
    print(f"A: {entry['answer']}\n")
```

---

**For more examples, see the [User Guide](./USER_GUIDE.md) and [Developer Guide](./DEVELOPER_GUIDE.md).**
