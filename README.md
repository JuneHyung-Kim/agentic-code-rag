# Code RAG Agent

> **Code Knowledge Expert** - A specialized agent for understanding large codebases through semantic code search and retrieval-augmented generation.

## 🎯 Project Role

This project serves as a **Code Knowledge Expert Agent** in multi-agent systems:
- **Primary Function**: Semantic code search and analysis
- **Specialization**: Understanding and navigating large codebases (C, C++, Python)
- **Integration**: Can be integrated into multi-agent workflows as a code understanding specialist

> 💡 **Learning Project**: This is a step-by-step educational project developed with AI assistance. Features are added incrementally as I learn and test each component.

---

## Quick Start

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env: add GEMINI_API_KEY or OPENAI_API_KEY

# 3. Use
python src/main.py index /path/to/code
python src/main.py search "your query"
python src/main.py chat
```

---

## Documentation

- [User Guide](./docs/USER_GUIDE.md)
- [Developer Guide](./docs/DEVELOPER_GUIDE.md)
- [API Reference](./docs/API_REFERENCE.md)
- [Roadmap](./ROADMAP.md) - Future improvements and features

---

## Development

```bash
pip install -e ".[dev]"
black src/
mypy src/
```

---

## License

MIT
