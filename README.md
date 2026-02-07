# Code RAG Agent

> **Code Knowledge Expert** - A specialized agent for understanding large codebases through semantic code search and retrieval-augmented generation.

## 🎯 Project Role

This project serves as a **Code Knowledge Expert Agent** in multi-agent systems:
- **Primary Function**: Semantic code search and analysis
- **Specialization**: Understanding and navigating large codebases (C, C++, Python)
- **Integration**: Can be integrated into multi-agent workflows as a code understanding specialist

> 💡 **Learning Project**: This is a step-by-step educational project developed with AI assistance. Features are added incrementally as I learn and test each component.

---

## Prerequisites: Native Tool Calling Support Required

This agent uses LangGraph's **ToolNode**, which requires the LLM to invoke tools directly. You must use a model that supports **Native Tool Calling**.

**What is Native Tool Calling?**
Unlike regular LLMs that only generate text, native tool calling models are **trained from the ground up to know how to use tools**. They can decide when to call a tool, output structured tool calls (name + JSON arguments) through a dedicated channel, and process the results — all learned during the training phase, not through prompt engineering hacks.

| Provider | Supported Models | Notes |
|----------|-----------------|-------|
| **Gemini** | `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash` | Natively supported |
| **Ollama** | `llama3.2`, `qwen2.5`, etc. | Varies by model — check the **Tools** tag on [Ollama models](https://ollama.com/search?c=tools) |

> **Warning**: Models without tool calling support (e.g. `deepseek-r1`, `phi-3`) will not work — the agent will fail to invoke any tools.

---

## Quick Start

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env: Configure embedding and chat models
# See .env.example for all options

# Example configurations:
# - All local (Ollama): EMBEDDING_PROVIDER=ollama, CHAT_PROVIDER=ollama
# - Cloud only: EMBEDDING_PROVIDER=openai, CHAT_PROVIDER=openai  
# - Hybrid: EMBEDDING_PROVIDER=default, CHAT_PROVIDER=ollama

# 3. Use
python src/main.py index /path/to/code
python src/main.py search "your query"
python src/main.py chat
```

---

## Documentation

- [User Guide](./docs/USER_GUIDE.md)
- [Developer Guide](./docs/DEVELOPER_GUIDE.md)
- [Indexing and Parsing Guide](./docs/INDEXING_AND_PARSING.md)
- [API Reference](./docs/API_REFERENCE.md)
- [Roadmap](./ROADMAP.md) - Future improvements and features

---

## Development

```bash
pip install -e ".[dev]"
black src/
mypy src/
```

### Testing

```bash
# Graph structure + tool registration + node units + E2E (mock-based, no API keys needed)
pytest tests/test_agent_graph.py tests/test_agent_tools.py tests/test_agent_nodes.py tests/test_agent_e2e.py -v

# Live LLM tests (requires API keys, prompts user for confirmation before calling)
pytest tests/test_agent_e2e.py -m live -s -v
```

### Visualization & Monitoring

```bash
# Graph visualization — print Mermaid diagram to stdout (no API keys needed)
python scripts/visualize_graph.py --mermaid

# Save as PNG (uses external Mermaid rendering API)
python scripts/visualize_graph.py --png graph.png

# ASCII art in terminal (requires grandalf package)
python scripts/visualize_graph.py --ascii

# Real-time agent execution monitoring (requires API keys)
python scripts/monitor_agent.py "How does the search engine work?"

# Verbose mode — full state dump at each node
python scripts/monitor_agent.py "How does the search engine work?" --verbose
```

---

## License

MIT
