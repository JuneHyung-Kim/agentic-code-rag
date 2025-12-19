# Tests

## Running Tests

### Prerequisites

Make sure you have the development dependencies installed:

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
# From project root
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test Files

```bash
# Parser tests only
pytest tests/test_parser.py -v

# Vector store tests only
pytest tests/test_vector_store.py -v

# Integration tests only
pytest tests/test_integration.py -v
```

### Run Specific Test Functions

```bash
pytest tests/test_parser.py::test_parse_python_function -v
```

## Test Structure

- **test_parser.py**: Unit tests for code parsing functionality
  - Tests Tree-sitter parsing for Python, C, C++
  - Tests definition extraction

- **test_vector_store.py**: Unit tests for vector database operations
  - Tests document storage and retrieval
  - Tests semantic search

- **test_integration.py**: End-to-end integration tests
  - Tests complete indexing workflow
  - Tests search after indexing
  - Uses temporary directories and databases

## Notes

- Integration tests use temporary directories and databases
- Tests are isolated and don't affect the main `db/` directory
- Vector store tests may be slower as they use actual embeddings
- You need either OPENAI_API_KEY or GEMINI_API_KEY in .env for tests to work properly

## Troubleshooting

If tests fail:

1. **Check API keys**: Make sure .env has valid API keys
2. **Check dependencies**: Run `pip install -e ".[dev]"` again
3. **Clear cache**: Delete any `__pycache__` directories
4. **Check permissions**: Ensure write permissions for temp directories

## Future Test Plans

See [ROADMAP.md](../ROADMAP.md) for planned test improvements:
- Performance benchmarks
- Larger test codebases
- Edge case coverage
- Mock API calls to avoid rate limits
