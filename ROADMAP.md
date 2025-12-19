# Development Roadmap

> **Philosophy**: Step-by-step evolution through testing and learning. Each phase should be completed and tested before moving to the next.

---

## 🎯 Current Status

**Version**: 0.1.0 - Basic Code RAG Agent  
**Role**: Code Knowledge Expert for semantic code search

### ✅ Implemented Features
- Tree-sitter based code parsing (Python, C, C++)
- Function/class definition extraction
- Vector database indexing (ChromaDB)
- Semantic code search
- AI chat interface with function calling
- Multi-provider support (OpenAI, Gemini)

---

## 📋 Phase 1: Enhanced Indexing (Next)

### 1.1 Richer Metadata Extraction
**Goal**: Extract more context from code for better search results

- [ ] **Function Signatures**
  - Parameter names and types
  - Return type information
  - Generic/template parameters (C++)
  
- [ ] **Documentation Extraction**
  - Docstrings (Python)
  - Doxygen comments (C/C++)
  - Inline comments near definitions
  
- [ ] **Code Metrics**
  - Function complexity (cyclomatic complexity)
  - Lines of code
  - Number of parameters

**Why**: Better metadata = more accurate search and smarter AI responses

---

### 1.2 Language Support Expansion
**Goal**: Support more programming languages

- [ ] **JavaScript/TypeScript**
  - Add tree-sitter-javascript
  - Add tree-sitter-typescript
  - Extract functions, classes, interfaces
  
- [ ] **Rust** (if needed)
  - Add tree-sitter-rust
  - Extract functions, structs, traits

**Why**: Enable analysis of polyglot codebases

---

### 1.3 Incremental Indexing
**Goal**: Update index when files change without full re-indexing

- [ ] **File Change Detection**
  - Track file modification times
  - Identify changed/added/deleted files
  
- [ ] **Partial Re-indexing**
  - Remove old entries for changed files
  - Add new entries for changed/new files
  - Skip unchanged files

- [ ] **Index Versioning**
  - Store index schema version
  - Handle migrations when parser logic changes

**Why**: Faster iteration when working with large codebases

---

## 📋 Phase 2: Advanced Search Capabilities

### 2.1 Multi-Strategy Search
**Goal**: Combine different search methods for better results

- [ ] **Keyword Search**
  - Exact symbol name matching
  - Fuzzy matching for typos
  
- [ ] **Regex Search**
  - Pattern-based code search
  - Use case: Find all functions matching a pattern
  
- [ ] **Hybrid Search**
  - Combine semantic + keyword
  - Weighted scoring (e.g., 70% semantic, 30% keyword)
  
- [ ] **Intelligent Strategy Selection**
  - Analyze query to pick best strategy
  - "find function named X" → keyword
  - "how to handle errors" → semantic

**Why**: Different queries need different search approaches

---

### 2.2 Contextual Search
**Goal**: Understand code relationships and context

- [ ] **Call Graph Analysis**
  - Find all callers of a function
  - Find all functions called by a function
  - Requires: static analysis or tree-sitter query extensions
  
- [ ] **Dependency Tracking**
  - Import/include relationships
  - Module dependencies
  
- [ ] **Related Code Suggestions**
  - Find similar functions
  - Find code in same module/class
  - Find code that uses same APIs

**Why**: Code understanding requires context, not just individual snippets

---

### 2.3 Search Result Ranking Improvements
**Goal**: Return most relevant results first

- [ ] **Relevance Scoring**
  - Boost results from recently modified files
  - Boost results with documentation
  - Penalty for test files (optional)
  
- [ ] **Query Expansion**
  - Expand synonyms (e.g., "remove" → "delete", "erase")
  - Use common abbreviations
  
- [ ] **User Feedback Loop**
  - Track which results users select
  - Learn to improve ranking over time

**Why**: Top results should be what users actually need

---

## 📋 Phase 3: Code Analysis & Understanding

### 3.1 Static Analysis Integration
**Goal**: Deeper code understanding beyond syntax

- [ ] **Type Information**
  - Infer types from usage (Python)
  - Parse type annotations
  - Extract C++ template information
  
- [ ] **Symbol Resolution**
  - Resolve variable/function references
  - Handle imports and includes
  - Cross-file symbol tracking

**Why**: Understanding types and references enables better code explanation

---

### 3.2 Code Quality Insights
**Goal**: Provide quality metrics alongside search

- [ ] **Complexity Warnings**
  - Flag overly complex functions
  - Suggest refactoring opportunities
  
- [ ] **Code Smell Detection**
  - Long functions
  - Too many parameters
  - Duplicated code patterns
  
- [ ] **Test Coverage Info**
  - Identify which functions have tests
  - Link to test files

**Why**: Help developers understand code quality, not just functionality

---

## 📋 Phase 4: Agent Collaboration Features

### 4.1 Multi-Agent Messaging Protocol
**Goal**: Standardize communication with other agents

- [ ] **Message Format**
  - Define AgentMessage schema
  - Request/response patterns
  - Error handling
  
- [ ] **Task Types**
  - "search": semantic code search
  - "analyze": deep code analysis
  - "explain": natural language explanation
  - "find_definition": symbol lookup
  - "find_usages": find references

**Why**: Enable seamless integration with other specialized agents

---

### 4.2 Context Sharing
**Goal**: Share knowledge between agent sessions

- [ ] **Conversation History**
  - Track what code has been discussed
  - Avoid redundant searches
  
- [ ] **Code Understanding Cache**
  - Cache expensive analysis results
  - Share insights across queries
  
- [ ] **Project State Awareness**
  - Know what files are currently open
  - Prioritize relevant code sections

**Why**: Agents should build on previous interactions

---

## 📋 Phase 5: Performance & Scalability

### 5.1 Large Codebase Optimization
**Goal**: Handle projects like Linux kernel, Chromium

- [ ] **Chunking Strategy Optimization**
  - Smart code splitting (not just by function)
  - Preserve context at boundaries
  
- [ ] **Lazy Loading**
  - Don't load entire index into memory
  - Stream results as needed
  
- [ ] **Distributed Indexing**
  - Parallel file processing
  - Multiple worker processes

**Why**: Real-world codebases can be millions of lines

---

### 5.2 Query Performance
**Goal**: Sub-second search response time

- [ ] **Index Optimization**
  - Optimize ChromaDB settings
  - Consider alternative vector DBs (Qdrant, Milvus)
  
- [ ] **Caching**
  - Cache frequent queries
  - Cache embedding results
  
- [ ] **Benchmark Suite**
  - Measure search latency
  - Measure indexing speed
  - Compare different configurations

**Why**: Agent should be fast enough for interactive use

---

## 🚫 Out of Scope (For Now)

These are valuable but deferred:
- ❌ **API Server / REST API**: Focus on core functionality first
- ❌ **Web UI**: Command-line is sufficient for learning
- ❌ **Cloud Deployment**: Local development only
- ❌ **Authentication/Authorization**: Not needed for single-user
- ❌ **Code Editing/Refactoring**: This is a knowledge expert, not a code writer
- ❌ **Real-time File Watching**: Manual re-indexing is acceptable for now

---

## 📝 Testing Strategy

Each phase should include:

1. **Unit Tests**
   - Test individual components
   - Mock external dependencies
   
2. **Integration Tests**
   - Test end-to-end workflows
   - Use sample codebases (small C/Python projects)
   
3. **Manual Testing**
   - Index real projects (llama.cpp, Python stdlib)
   - Verify search quality
   - Test AI chat responses

4. **Performance Testing**
   - Measure indexing time
   - Measure search latency
   - Monitor memory usage

---

## 🎓 Learning Goals

Through this project, I aim to understand:

- ✅ Tree-sitter for code parsing
- ✅ Vector databases and embeddings
- ✅ RAG (Retrieval-Augmented Generation)
- ✅ LLM function calling
- ⏳ Static analysis techniques
- ⏳ Information retrieval algorithms
- ⏳ Multi-agent system design
- ⏳ Performance optimization for large datasets

---

## 📊 Success Metrics

How to know each phase is successful:

**Phase 1**: 
- Can extract docstrings and signatures
- Can index 10+ languages
- Incremental indexing is 10x faster than full re-index

**Phase 2**:
- Search relevance improved (subjective, but noticeable)
- Can find functions by exact name
- Can trace call graphs

**Phase 3**:
- Can resolve types across files
- Can detect code smells accurately

**Phase 4**:
- Other agents can call this agent programmatically
- Context is preserved across multi-turn conversations

**Phase 5**:
- Can index 100k+ line codebase in < 5 minutes
- Search returns results in < 500ms

---

## 🔄 Iteration Process

1. **Plan**: Choose next feature from roadmap
2. **Research**: Study how others solve this (papers, tools, libraries)
3. **Design**: Sketch API/architecture
4. **Implement**: Write code incrementally
5. **Test**: Unit tests + manual testing
6. **Reflect**: Document what worked, what didn't
7. **Repeat**: Move to next feature

**Key Principle**: Don't move to next phase until current phase is solid and tested.

---

*Last Updated: December 19, 2025*
