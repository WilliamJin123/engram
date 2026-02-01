# External Integrations

**Analysis Date:** 2026-01-31

## APIs & External Services

**LLM Integration (Optional):**
- Framework: Abstract interface pattern
  - Base class: `src/agentic/llm_interface.py` - `BaseLLMReranker`
  - Implementation approach: Extend `BaseLLMReranker` and implement `rerank()` method
  - Purpose: Semantic reranking of retrieved memory patterns
  - Current state: MockLLMReranker provided for testing without API calls

**Keycycle Integration (Documentation):**
- Service: Keycycle API key rotation manager
  - Documented in: `docs/KEYCYCLE.md`
  - Supported Providers: OpenAI, Anthropic, Groq, Gemini, Cohere, TwelveLabs, and others
  - Purpose: Automatic API key rotation and rate limit management
  - Status: Documented as integration path, not currently implemented in codebase
  - Note: KEYCYCLE.md specifies this as the recommended approach for LLM integration (DO NOT use Anthropic API directly)

## Data Storage

**Databases:**
- None integrated - In-memory storage only
  - Memory implementation: `src/agentic/memory_store.py` - `MemoryStore` class
  - Storage mechanism: Python dictionary (`self.patterns: dict[str, EvolvingPattern]`)
  - Persistence: None - all data lost on process exit
  - Optional database reference: `docs/KEYCYCLE.md` mentions `TIDB_DB_URL` for optional usage persistence
    - Environment variable: `TIDB_DB_URL=mysql://user:pass@host:3306/db`
    - Status: Optional for Keycycle integration, not implemented for memory storage

**File Storage:**
- Local filesystem only - No file storage currently implemented
- No S3, cloud storage, or remote file system integrations
- PyTorch can load models from disk via fsspec (available), but not actively used

**Caching:**
- None - No caching layer implemented
- Pattern similarity computations are calculated on-demand

## Authentication & Identity

**Auth Provider:**
- Custom/None - No authentication layer in core library
  - LLM authentication: Handled by Keycycle when integrated (API keys via environment variables)
  - No JWT, OAuth, or session management

**API Key Management:**
- Environment variables (documented approach for Keycycle integration)
- Environment variable pattern: `{PROVIDER}_API_KEY_N` (where N is key number)
- Examples from docs/KEYCYCLE.md:
  - `OPENAI_API_KEY_1`, `OPENAI_API_KEY_2`
  - `ANTHROPIC_API_KEY_1`, `ANTHROPIC_API_KEY_2`
  - `TWELVELABS_API_KEY_1`, `TWELVELABS_API_KEY_2`
  - `TWELVELABS_INDEX_ID_1`, `TWELVELABS_INDEX_ID_2` (extra parameters)
- Key count variable: `NUM_{PROVIDER}` (e.g., `NUM_OPENAI=2`)

## Monitoring & Observability

**Error Tracking:**
- None - No external error tracking service integrated
- Python built-in exception handling used

**Logs:**
- Console output only (via standard Python logging or print statements)
- Colorama available for colored terminal output on Windows

**Metrics:**
- Usage tracking optional via database (Keycycle feature when integrated)
  - Requires: `TIDB_DB_URL` environment variable

## CI/CD & Deployment

**Hosting:**
- Not specified - Library component, deployment context varies
- Designed for embedding in agent applications

**CI Pipeline:**
- None detected in repository
  - Testing framework: pytest configured and ready
  - Test discovery: `tests/` directory
  - Run command: `pytest` (configured in `pyproject.toml`)

## Environment Configuration

**Required env vars:**
- None required for core functionality
- Optional for LLM integration (when Keycycle is used):
  - `NUM_{PROVIDER}` - Number of API keys for provider
  - `{PROVIDER}_API_KEY_N` - Individual API keys
  - `TIDB_DB_URL` - Optional database URL for usage persistence (Keycycle feature)

**Secrets location:**
- Environment variables (recommended approach per docs/KEYCYCLE.md)
- No `.env` file support configured
- No secrets management library in dependencies

## Webhooks & Callbacks

**Incoming:**
- None - Library does not expose HTTP endpoints

**Outgoing:**
- None - Library does not make HTTP callbacks
- LLM integration (when added) may make HTTP requests to LLM APIs (handled by LLM client libraries like OpenAI SDK)

## Network Communication

**HTTP Clients:**
- None in core dependencies
- External LLM integration will require client library (e.g., openai, anthropic SDKs)
- These should be added as optional dependencies or managed by application

## Pattern Processing Pipeline

**Text Encoding:**
- Internal: `src/agentic/text_encoder.py` - TextEncoder class
- No external NLP or language model services used
- Purely algorithmic encoding via SHA-256 hashing and mathematical transformations

**Pattern Similarity:**
- Internal computation using PyTorch
- Methods: Jaccard similarity (bitwise) or quantum interference-based retrieval
- No external services or APIs

**Memory Retrieval:**
- Internal pattern matching
- Optional LLM reranking via pluggable interface (docs/KEYCYCLE.md for integration)

---

*Integration audit: 2026-01-31*
