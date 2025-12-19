# Code RAG Agent

> A proof-of-concept project for learning and experimenting with code retrieval-augmented generation.

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
