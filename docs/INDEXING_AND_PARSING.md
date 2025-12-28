# Indexing and Parsing Guide

This document explains how the indexing and parsing pipeline works in detail.
It is intended for developers working in `src/indexing/` or extending language support.

## Scope and Entry Points

- Primary command: `python src/main.py index <path>`
- Orchestrator: `src/indexing/indexer.py` (`CodeIndexer`)
- Parser dispatch: `src/indexing/parser.py` (`CodeParser`)
- Language implementations: `src/indexing/parsers/python_parser.py`, `src/indexing/parsers/cpp_parser.py`
- Storage: `src/indexing/vector_store.py` (`VectorStore`)
- Schema: `src/indexing/schema.py` (`CodeNode`)

---

## Pipeline Overview

1. Discover files under the target directory.
2. Filter by extension and skip excluded directories.
3. Parse each file into a list of `CodeNode` objects.
4. Build embedding text for each node (docstring, signature, etc.).
5. Build metadata for each node (flattened for DB compatibility).
6. Generate stable, collision-resistant IDs.
7. Persist documents and metadata to ChromaDB.

---

## File Discovery and Filtering

File discovery uses `os.walk` and skips:
- Hidden directories (any path segment starting with `.`)
- Known build or cache directories: `build`, `venv`, `__pycache__`, `node_modules`, `.git`, `dist`, `egg-info`

Supported file extensions:
- Python: `.py`
- C: `.c`, `.h`
- C++: `.cpp`, `.hpp`, `.cc`, `.cxx`

Filtering happens in `CodeIndexer._index_file` in `src/indexing/indexer.py`.

---

## CodeNode Schema

`src/indexing/schema.py` defines the canonical structure used during indexing.
Lines are 0-based, following Tree-sitter defaults.

Key fields:
- `type`: `function`, `class`, `struct`, `method`, `enum`, `typedef`, `macro`, `global_var`, `function_decl`
- `name`: extracted identifier
- `file_path`: absolute during parsing, relative when stored
- `start_line`, `end_line`: 0-based
- `content`: raw source text for the node
- `language`: `python`, `c`, or `cpp`
- `docstring`, `signature`, `return_type`
- `arguments`: list of strings
- `parent_name`: class or namespace owner if applicable
- `imports`: file-level import/include statements
- `metadata`: reserved for extensions

---

## Python Parsing Details

Implementation: `src/indexing/parsers/python_parser.py`

Captured node types:
- `function_definition` and `async_function_definition` -> `function`
- `class_definition` -> `class`
- `decorated_definition` wrapping functions/classes
- `assignment` -> `global_var` (module-level only)

Key behaviors:
- Functions defined inside classes are labeled as `method` and get `parent_name`.
- Docstring extraction:
  - First statement in the body must be a string literal.
  - Comments are skipped when searching for docstrings.
- Signature extraction:
  - From node start until the body begins, with trailing `:` removed.
- Return type extraction:
  - Uses the `return_type` field when annotated.
- Argument extraction:
  - Identifiers and parameter nodes, including splats.
- File-level imports:
  - Captures `import_statement` and `import_from_statement`.

---

## C and C++ Parsing Details

Implementation: `src/indexing/parsers/cpp_parser.py` with `lang_type` set to `c` or `cpp`.

Captured node types:
- `function_definition` -> `function`
- `struct_specifier` -> `struct`
- `class_specifier` -> `class` (C++ only)
- `enum_specifier` -> `enum`
- `type_definition` -> `typedef`
- `preproc_def` and `preproc_function_def` -> `macro`
- `declaration` -> `function_decl` or `global_var` (top-level only)

Filters and classification:
- Non-top-level `declaration` nodes are ignored.
- Anonymous structs/classes are skipped.
- Type nodes without bodies are ignored.
- `declaration` is classified as:
  - `function_decl` if it has a function declarator
  - `global_var` otherwise

Name resolution:
- Handles pointer, reference, and nested declarators.
- Supports qualified identifiers (namespaces, class scope).
- Macros use the second token in the preprocessor line as the name.

Docstring extraction:
- Collects contiguous comment nodes immediately preceding the definition.

Signature and return type:
- Function definitions: signature is the text before the body.
- Function declarations: signature is the declaration without the trailing `;`.
- Return type is inferred by slicing from node start to the name node.

Arguments:
- Extracted from `parameter_declaration` nodes inside the function declarator.

Parent name:
- Derived from `ClassName::method` when present.
- Otherwise resolved by walking parent class/struct nodes.

Includes:
- Captures `preproc_include` as file-level imports.

---

## Embedding Text Construction

`CodeIndexer._index_file` builds the embedding text as:

1. `Docstring: ...`
2. `Signature: ...`
3. `Returns: ...`
4. `Parameters: ...`
5. `Code:\n<raw node content>`

Only fields that exist are included.

---

## Metadata and ID Strategy

Metadata stored with each document includes:
- `file_path`, `name`, `type`, `language`, `start_line`, `end_line`
- Optional: `parent_name`, `signature`, `return_type`
- `imports`: joined into a single string and truncated
- `arguments`: JSON-serialized list

ID format in `CodeIndexer`:
```
<rel_path>:<type>:<name>:<start_line>:<short_hash>
```

The hash is built from node content plus signature/return/arguments to:
- Avoid collisions across overloads or renamed symbols.
- Allow updates to produce new IDs when the definition changes.

If duplicate IDs still occur in a batch, a numeric suffix is appended.

---

## Vector Store Notes

Implementation: `src/indexing/vector_store.py`

- Uses ChromaDB with a persistent client (default path `./db`).
- Collection name: `code_chunks`
- Embedding provider is selected by `EMBEDDING_PROVIDER` in `src/config.py`.
- If the embedding function changes, the existing collection is deleted and recreated.

Query response includes:
- `ids`, `distances`, `metadatas`, `documents`

---

## Extending Language Support

To add a new language:

1. Add a new parser in `src/indexing/parsers/`.
2. Register it in `src/indexing/parser.py`.
3. Map its file extensions in `src/indexing/parser.py` and `src/indexing/indexer.py`.
4. Emit `CodeNode` objects with consistent fields.

If the new language introduces new node types, update `CodeNode.type` usage accordingly.
