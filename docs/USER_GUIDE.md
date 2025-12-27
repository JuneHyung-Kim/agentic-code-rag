# User Guide

## 📖 Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Indexing a Project](#indexing-a-project)
  - [Searching Code](#searching-code)
  - [Interactive Chat](#interactive-chat)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Introduction

This is an AI-powered code assistant designed to help you understand and navigate large codebases. It uses advanced code parsing, semantic search, and AI chat capabilities to answer questions about your code.

### Key Features

✨ **Smart Code Indexing**: Automatically parses C, C++, and Python code to understand structure  
🔍 **Semantic Search**: Find relevant code using natural language queries  
💬 **AI Chat Interface**: Ask questions and get intelligent responses about your codebase  
🔌 **Multi-Provider Support**: Works with OpenAI, Google Gemini, and local Ollama models

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Git (for cloning the repository)
- API Key (OpenAI or Gemini) OR local Ollama setup

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd <project-directory>
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -e .
```

### Step 4: Configure API Keys

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to configure your models. See [Configuration](#configuration) for details.

---

## Quick Start

### 1. Index Your First Project

Let's index a sample C project:

```bash
# Index a project directory
python src/main.py index /path/to/your/c/project

# Example: Index the Linux kernel networking subsystem
python src/main.py index /usr/src/linux/net
```

**What happens during indexing:**
- Scans all `.c`, `.cpp`, `.h`, `.hpp`, and `.py` files
- Parses function definitions, class definitions, and structs
- Creates semantic embeddings and stores them in a vector database
- Progress is shown in real-time

### 2. Search for Code

```bash
python src/main.py search "memory allocation functions"
```

**Example searches:**
```bash
# Find network protocol implementations
python src/main.py search "TCP connection handling"

# Find memory management code
python src/main.py search "kmalloc and memory allocation"

# Find scheduler code
python src/main.py search "process scheduling algorithms"
```

### 3. Start Interactive Chat

```bash
python src/main.py chat
```

**Example conversation:**
```
You: How does the TCP handshake work in this codebase?
Agent: [Searches code and provides detailed explanation with file references]

You: Show me the implementation of socket creation
Agent: [Returns relevant code snippets with line numbers]

You: exit
```

---

## Detailed Usage

### Indexing a Project

#### Basic Indexing

```bash
python src/main.py index <project_path>
```

**Options:**
- `<project_path>`: Absolute or relative path to the project directory

**What gets indexed:**
- All `.py` files (Python)
- All `.c`, `.h` files (C)
- All `.cpp`, `.hpp`, `.cc`, `.cxx` files (C++)

**What gets skipped:**
- Hidden directories (starting with `.`)
- Build directories (`build/`, `dist/`)
- Virtual environments (`venv/`, `.venv/`)
- Cache directories (`__pycache__/`)

#### Re-indexing

To re-index a project (e.g., after code changes):

```bash
# Delete existing database
rm -rf db/

# Re-index
python src/main.py index /path/to/project
```

#### Indexing Large Projects

For very large projects (e.g., entire Linux kernel):

1. Index specific subdirectories:
   ```bash
   python src/main.py index /usr/src/linux/mm        # Memory management
   python src/main.py index /usr/src/linux/kernel    # Core kernel
   python src/main.py index /usr/src/linux/net       # Networking
   ```

2. Or use the full path (may take 10-30 minutes):
   ```bash
   python src/main.py index /usr/src/linux
   ```

---

### Searching Code

#### Basic Search

```bash
python src/main.py search "your search query"
```

**Search Tips:**

✅ **Good queries:**
- "memory allocation and deallocation"
- "TCP packet processing"
- "process scheduler implementation"
- "file system operations"

❌ **Bad queries:**
- Single words like "memory" or "function"
- Too generic: "code"
- Too specific: "line 123 in file.c"

#### Understanding Search Results

Results include:
- **File path**: Location of the code
- **Type**: function, class, struct
- **Name**: Function/class/struct name
- **Line numbers**: Start and end lines
- **Content**: The actual code snippet

Example output:
```
--- Result 1 (kernel/sched/core.c:4567) ---
Type: function, Name: schedule
void schedule(void)
{
    struct task_struct *tsk = current;
    ...
}
```

---

### Interactive Chat

#### Starting a Chat Session

```bash
python src/main.py chat
```

#### Chat Commands

- **Regular questions**: Just type your question
- **Exit**: Type `exit` or `quit` to end the session

#### Effective Questions

**Architecture Questions:**
```
You: Explain the memory management architecture
You: How are processes scheduled?
You: What is the call flow for socket creation?
```

**Code Exploration:**
```
You: Find all functions that allocate memory
You: Show me how interrupts are handled
You: What are the main data structures for network packets?
```

**Debugging Help:**
```
You: Where could a memory leak occur in this code?
You: Show me error handling in the TCP stack
You: Find race condition risks in the scheduler
```

#### How the Agent Works

1. **Receives your question**
2. **Searches the indexed codebase** for relevant snippets
3. **Analyzes the code** using AI
4. **Provides an answer** with specific file and line references

---

## Configuration

The agent supports independent configuration for embedding (search) and chat (conversations). You can mix and match providers.

### Environment Variables

All configuration is in the `.env` file:

```ini
# 1. Choose Embedding Provider (for search)
# Options: openai, gemini, ollama, default (Sentence Transformers)
EMBEDDING_PROVIDER=default
EMBEDDING_MODEL=text-embedding-3-small  # for openai
# EMBEDDING_MODEL=mxbai-embed-large     # for ollama

# 2. Choose Chat Provider (for conversation)
# Options: openai, gemini, ollama
CHAT_PROVIDER=gemini
CHAT_MODEL=gemini-1.5-flash

# 3. API Keys (set only the ones you're using)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# 4. Local Configuration (for Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# 5. Project Settings
PROJECT_ROOT=./
```

### Choosing Providers

#### 1. All Local (Free, Private)
Uses Ollama for both embeddings and chat. Best for privacy and cost, but requires a capable local machine.
- `EMBEDDING_PROVIDER=ollama`
- `CHAT_PROVIDER=ollama`
- Models: `mxbai-embed-large`, `qwen2.5`

#### 2. All Cloud (Best Quality)
Uses OpenAI for both. Best performance but costs money.
- `EMBEDDING_PROVIDER=openai`
- `CHAT_PROVIDER=openai`
- Models: `text-embedding-3-small`, `gpt-4o`

#### 3. Hybrid (Cost Effective)
Uses local/free embeddings and a cost-effective chat model.
- `EMBEDDING_PROVIDER=default` (local Sentence Transformers)
- `CHAT_PROVIDER=gemini` (Free tier available)

See [Model Configuration](./MODEL_CONFIGURATION.md) for detailed examples.

### Database Location

The vector database is stored in `./db/` by default. To change:

1. Edit `src/indexing/vector_store.py`
2. Modify the `persist_path` parameter in `VectorStore.__init__()`

---

## Troubleshooting

### Common Issues

#### Issue: "OPENAI_API_KEY is not set" or "GEMINI_API_KEY is not set"

**Solution:**
1. Make sure `.env` file exists in the project root
2. Check that the API key is correctly set
3. Verify your `CHAT_PROVIDER` and `EMBEDDING_PROVIDER` settings match your keys.

#### Issue: "No results found" when searching

**Possible causes:**
- Project hasn't been indexed yet
- Search query is too specific
- Database is corrupted

**Solution:**
```bash
# Re-index the project
rm -rf db/
python src/main.py index /path/to/project

# Try broader search terms
python src/main.py search "memory management"
```

#### Issue: Indexing fails with encoding errors

**Solution:**
This is normal for binary or non-UTF-8 files. The indexer automatically skips these files.

#### Issue: Out of memory during indexing

**Solution:**
Index smaller subdirectories instead of the entire project:
```bash
python src/main.py index /project/src/subsystem1
python src/main.py index /project/src/subsystem2
```

#### Issue: Chat responses are slow

**Possible causes:**
- Large codebase with many results
- API rate limits
- Local model performance (Ollama)

**Solution:**
- Use Gemini Flash for faster responses
- Wait a few seconds between queries
- Reduce the scope of your questions

---

## Best Practices

### Indexing

1. **Index once, search many**: Don't re-index unless code changes significantly
2. **Organize by subsystem**: For large projects, index logical subsystems separately
3. **Clean before re-indexing**: Delete the `db/` directory before re-indexing
4. **Exclude test files**: If needed, modify the indexer to skip `test/` directories

### Searching

1. **Use descriptive queries**: "TCP connection establishment" > "TCP"
2. **Combine concepts**: "memory allocation error handling"
3. **Iterate**: Start broad, then narrow down
4. **Use chat for exploration**: Chat is better for "why" and "how" questions

### Chat

1. **Be specific**: "How does the scheduler pick the next process?" > "scheduler"
2. **Reference results**: "In the schedule() function you found, explain the logic"
3. **Ask follow-ups**: Build on previous answers
4. **Verify**: Always check the referenced code files

### Performance

1. **Limit scope**: Don't index your entire filesystem
2. **Use SSD storage**: Vector database benefits from fast I/O
3. **Monitor API usage**: Both OpenAI and Gemini have rate limits
4. **Cache results**: Review search results before asking the same question

---

## Next Steps

- 📚 Read the [Developer Guide](./DEVELOPER_GUIDE.md) to understand the architecture
- 🔧 Check the [API Reference](./API_REFERENCE.md) for programmatic usage
- 🐛 Report issues on [GitHub Issues](https://github.com/JuneHyung-Kim/os-devel-agent/issues)

---

**Happy Coding!** 🚀
