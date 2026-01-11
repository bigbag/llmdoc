# LLMDoc

[![CI](https://github.com/bigbag/llmdoc/workflows/CI/badge.svg)](https://github.com/bigbag/llmdoc/actions?query=workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/llmdoc.svg)](https://pypi.python.org/pypi/llmdoc)
[![downloads](https://img.shields.io/pypi/dm/llmdoc.svg)](https://pypistats.org/packages/llmdoc)
[![versions](https://img.shields.io/pypi/pyversions/llmdoc.svg)](https://github.com/bigbag/llmdoc)
[![license](https://img.shields.io/github/license/bigbag/llmdoc.svg)](https://github.com/bigbag/llmdoc/blob/master/LICENSE)

MCP server with RAG (BM25) for llms.txt documentation. Provides semantic search across documentation sources with automatic background refresh.

## Features

- **llms.txt support** - Automatically parses and indexes documentation from llms.txt files
- **Hybrid two-stage search** - DuckDB FTS with Porter stemming for broad recall, BM25 reranking for precision
- **Named sources** - Configure sources with names like `fast_mcp:https://...` for easy filtering
- **Source filtering** - Search across all sources or filter by specific source name
- **Persistent storage** - DuckDB-based index that survives restarts
- **Background refresh** - Configurable auto-refresh interval (default: 6 hours)
- **Source attribution** - Every search result includes source name and URL

## Quick Start

1. Add to Claude Code (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "llmdoc": {
      "command": "uvx",
      "args": ["llmdoc"],
      "env": {
        "LLMDOC_SOURCES": "fast_mcp:https://gofastmcp.com/llms.txt"
      }
    }
  }
}
```

2. Restart Claude Code - the server will automatically fetch and index documentation.

3. Ask Claude questions like "How do I create a tool in FastMCP?" and it will search the indexed docs.

## What is llms.txt?

[llms.txt](https://llmstxt.org) is a specification for providing LLM-friendly documentation. Websites add a `/llms.txt` markdown file to their root directory containing curated, concise content optimized for AI consumption. LLMDoc indexes these files and their linked documents to enable semantic search.

Example sources:
- [FastMCP](https://gofastmcp.com/llms.txt)
- [PydanticAI](https://ai.pydantic.dev/llms.txt)
- [Pydantic](https://docs.pydantic.dev/latest/llms.txt)
- [LangGraph](https://langchain-ai.github.io/langgraph/llms.txt)

## Installation

```bash
# Run directly with uvx (no install needed)
uvx llmdoc

# Or install with uv
uv tool install llmdoc

# Or install with pip
pip install llmdoc

# Or install with pipx
pipx install llmdoc
```

## Configuration

### Source Format

Sources can be specified in two formats:
- **Named**: `name:url` - e.g., `fast_mcp:https://gofastmcp.com/llms.txt`
- **Unnamed**: Just the URL - name is auto-generated from domain

Named sources allow you to filter search results by source name.

### Environment Variables

```bash
# Comma-separated list of sources (named or unnamed)
export LLMDOC_SOURCES="fast_mcp:https://gofastmcp.com/llms.txt,pydantic_ai:https://ai.pydantic.dev/llms.txt"

# Optional: Custom database path (default: ~/.llmdoc/index.db)
export LLMDOC_DB_PATH="/path/to/index.db"

# Optional: Refresh interval in hours (default: 6)
export LLMDOC_REFRESH_INTERVAL="6"

# Optional: Max concurrent document fetches (default: 5)
export LLMDOC_MAX_CONCURRENT="5"

# Optional: Skip refresh on startup (default: false)
export LLMDOC_SKIP_STARTUP_REFRESH="true"
```

### Config File

Create `llmdoc.json` in the working directory:

```json
{
  "sources": [
    "fast_mcp:https://gofastmcp.com/llms.txt",
    "pydantic_ai:https://ai.pydantic.dev/llms.txt"
  ],
  "db_path": "~/.llmdoc/index.db",
  "refresh_interval_hours": 6,
  "max_concurrent_fetches": 5,
  "skip_startup_refresh": false
}
```

Or with explicit name/url objects:

```json
{
  "sources": [
    {"name": "fast_mcp", "url": "https://gofastmcp.com/llms.txt"},
    {"name": "pydantic_ai", "url": "https://ai.pydantic.dev/llms.txt"}
  ]
}
```

## Running the Server

LLMDoc uses stdio transport and is designed to be launched by MCP clients. Configure it in your MCP client (see below), and the client will start the server automatically.

For manual testing:

```bash
# Using uvx
uvx llmdoc

# Or as module
python -m llmdoc
```

## MCP Tools

- `search_docs(query, limit, source)` - Search documentation and return relevant passages with source URLs. Optional `source` parameter filters by source name (e.g., `fast_mcp`)
- `get_doc(url, offset, limit)` - Get document content with pagination support for large documents. Parameters: `offset` (default: 0) start position in bytes, `limit` (default: 50000, max: 100000) max bytes per call. Returns pagination metadata (`has_more`, `total_length`)
- `get_doc_excerpt(url, query, max_chunks, context_chars)` - Get relevant excerpts from a large document matching a query
- `list_sources()` - List all configured documentation sources with statistics
- `refresh_sources()` - Manually trigger a refresh of all documentation

## MCP Resources

- `doc://sources` - Returns JSON with configured sources list and refresh interval

## Adding to MCP Clients

### Claude Code

Add to `~/.claude/claude_code_config.json`:

```json
{
  "mcpServers": {
    "llmdoc": {
      "command": "uvx",
      "args": ["llmdoc"],
      "env": {
        "LLMDOC_SOURCES": "fast_mcp:https://gofastmcp.com/llms.txt,pydantic_ai:https://ai.pydantic.dev/llms.txt"
      }
    }
  }
}
```

### Standard MCP Configuration

Add to your MCP client's configuration file:

```json
{
  "mcpServers": {
    "llmdoc": {
      "command": "uvx",
      "args": ["llmdoc"],
      "env": {
        "LLMDOC_SOURCES": "fast_mcp:https://gofastmcp.com/llms.txt"
      }
    }
  }
}
```

## Example Usage

Once configured, the LLM can use these tools:

```
User: How do I create a tool in FastMCP?

LLM: [calls search_docs("create tool FastMCP")]

Result:
[
  {
    "title": "Tools",
    "snippet": "Creating a tool is as simple as decorating a Python function with @mcp.tool...",
    "url": "https://gofastmcp.com/servers/tools.md",
    "source": "fast_mcp",
    "source_url": "https://gofastmcp.com/llms.txt",
    "score": 12.5
  }
]
```

### Filtering by Source

You can filter results to a specific documentation source:

```
User: How do I create an agent in PydanticAI?

LLM: [calls search_docs("create agent", source="pydantic_ai")]

Result:
[
  {
    "title": "Agents",
    "snippet": "Agents are the primary interface for interacting with LLMs in PydanticAI...",
    "url": "https://ai.pydantic.dev/agents.md",
    "source": "pydantic_ai",
    "source_url": "https://ai.pydantic.dev/llms.txt",
    "score": 10.2
  }
]
```

### Getting Full Document Content

Use `get_doc` to retrieve document content (supports pagination for large documents):

```
LLM: [calls get_doc("https://ai.pydantic.dev/agents.md")]

Result:
{
  "title": "Agents",
  "content": "# Agents\n\nAgents are the primary interface for interacting with LLMs in PydanticAI...",
  "url": "https://ai.pydantic.dev/agents.md",
  "source": "pydantic_ai",
  "source_url": "https://ai.pydantic.dev/llms.txt",
  "offset": 0,
  "length": 5432,
  "total_length": 5432,
  "has_more": false
}
```

## Architecture

```
+------------------+
|    MCP Client    |
| (Claude, Cursor) |
+--------+---------+
         | stdio
         v
+------------------+     +------------------+     +------------------+
|  FastMCP Server  |---->|  Document Store  |<----|Document Fetcher  |
|                  |     |    (DuckDB)      |     | (async HTTP)     |
|  - search_docs   |     |                  |     |                  |
|  - get_doc       |     |  - Persistence   |     | - llms.txt parse |
|  - list_sources  |     |  - Deduplication |     | - HTML→Markdown  |
|  - refresh       |     |  - Change detect |     | - Concurrent     |
+--------+---------+     +------------------+     +------------------+
         |
         v
+------------------+
|   BM25 Index     |
|   (in-memory)    |
|                  |
|  - Chunking      |
|  - Tokenization  |
|  - Scoring       |
+------------------+
```

LLMDoc fetches documentation from llms.txt sources, stores it in DuckDB, and provides fast BM25 search through the MCP protocol.

## How It Works

### Document Fetching
When configured with documentation sources, LLMDoc:
1. Parses llms.txt files to discover all linked documents
2. Fetches each document concurrently (with rate limiting)
3. Converts HTML pages to Markdown automatically
4. Extracts titles from the first H1 heading

### Indexing
Documents are processed for efficient search:
1. **Chunking**: Large documents are split into ~500 character chunks at sentence boundaries
2. **Tokenization**: Text is lowercased and stopwords are removed
3. **Indexing**: BM25 algorithm indexes all chunks for relevance scoring

### Search
LLMDoc uses a hybrid two-stage retrieval approach:

**Stage 1 - DuckDB FTS (Recall):**
1. Query is processed by DuckDB's full-text search with Porter stemming
2. "running" matches "run", "café" matches "cafe"
3. Top 100 candidate chunks are retrieved

**Stage 2 - Python BM25 (Precision):**
1. Candidates are re-scored using exact-match BM25
2. Documents with exact query terms rank higher
3. Results are deduplicated by document URL
4. Top results are returned with relevance scores and snippets

### Background Refresh
LLMDoc automatically keeps documentation up-to-date:
- Checks for staleness on startup
- Refreshes every 6 hours (configurable)
- Uses content hashing to skip unchanged documents
- Removes documents no longer in llms.txt

## Technical Details

### Hybrid Two-Stage Search

LLMDoc combines DuckDB's native FTS with Python BM25 for optimal search quality:

**Stage 1 - DuckDB FTS:**
- Porter stemming normalizes words (running → run, documents → document)
- Accent handling (café → cafe)
- 571 built-in English stopwords
- Fast candidate retrieval using native C implementation

**Stage 2 - Python BM25:**
- BM25Okapi algorithm from `rank_bm25` library
- Exact term matching boosts precise matches
- Term frequency saturation, document length normalization, IDF weighting
- Thread-safe using `threading.RLock()`

### Chunking Strategy

Documents are chunked using a multi-level approach:

1. **Paragraph splitting**: First split on double newlines (`\n\n`)
2. **Sentence-boundary aware**: Long paragraphs split at `.!?` followed by whitespace
3. **Overlap**: 100 character overlap between chunks maintains context

Configuration:
- `chunk_size`: 500 characters (default)
- `chunk_overlap`: 100 characters (default)

### Database Schema

DuckDB stores documents and chunks:

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    source_name TEXT NOT NULL,    -- e.g., 'fast_mcp'
    source_url TEXT NOT NULL,     -- llms.txt URL
    doc_url TEXT NOT NULL UNIQUE, -- document URL
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,   -- SHA256 for change detection
    updated_at TIMESTAMP NOT NULL
)

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,      -- references documents.id
    content TEXT NOT NULL,        -- chunk text for FTS indexing
    start_pos INTEGER NOT NULL,   -- position in original document
    end_pos INTEGER NOT NULL
)
```

FTS index on chunks table with Porter stemmer for hybrid search.

### Concurrency Model

LLMDoc supports multiple concurrent instances:

- **Read operations**: Multiple instances can search simultaneously (read-only DuckDB mode)
- **Write operations**: Single instance holds exclusive lock during refresh
- **Graceful handling**: If refresh is locked, operation skips with status message

Document fetching uses `asyncio.Semaphore` to limit concurrent HTTP requests (default: 5).

### Stopwords

Two stopword lists are used:
- **DuckDB FTS (Stage 1)**: 571 built-in English stopwords
- **Python BM25 (Stage 2)**: 209 custom stopwords including articles, prepositions, pronouns, auxiliaries, and common verbs

## License

MIT License - see [LICENSE](LICENSE) file.
