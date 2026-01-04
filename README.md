# LLMDoc

[![CI](https://github.com/bigbag/llmdoc/workflows/CI/badge.svg)](https://github.com/bigbag/llmdoc/actions?query=workflow%3ACI)
[![pypi](https://img.shields.io/pypi/v/llmdoc.svg)](https://pypi.python.org/pypi/llmdoc)
[![versions](https://img.shields.io/pypi/pyversions/llmdoc.svg)](https://github.com/bigbag/llmdoc)
[![license](https://img.shields.io/github/license/bigbag/llmdoc.svg)](https://github.com/bigbag/llmdoc/blob/master/LICENSE)

MCP server with RAG (BM25) for llms.txt documentation. Provides semantic search across documentation sources with automatic background refresh.

## Features

- **llms.txt support** - Automatically parses and indexes documentation from llms.txt files
- **BM25 search** - Fast, keyword-based retrieval with relevance scoring and stopword filtering
- **Named sources** - Configure sources with names like `fast_mcp:https://...` for easy filtering
- **Source filtering** - Search across all sources or filter by specific source name
- **Persistent storage** - DuckDB-based index that survives restarts
- **Background refresh** - Configurable auto-refresh interval (default: 6 hours)
- **Source attribution** - Every search result includes source name and URL

## What is llms.txt?

[llms.txt](https://llmstxt.org) is a specification for providing LLM-friendly documentation. Websites add a `/llms.txt` markdown file to their root directory containing curated, concise content optimized for AI consumption. LLMDoc indexes these files and their linked documents to enable semantic search.

Example sources:
- [FastMCP](https://gofastmcp.com/llms.txt)
- [PydanticAI](https://ai.pydantic.dev/llms.txt)
- [Pydantic](https://docs.pydantic.dev/latest/llms.txt)
- [LangGraph](https://langchain-ai.github.io/langgraph/llms.txt)

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
When you search:
1. Your query is tokenized the same way as documents
2. BM25 scores each chunk against your query
3. Results are deduplicated by document URL
4. Top results are returned with relevance scores and snippets

### Background Refresh
LLMDoc automatically keeps documentation up-to-date:
- Checks for staleness on startup
- Refreshes every 6 hours (configurable)
- Uses content hashing to skip unchanged documents
- Removes documents no longer in llms.txt

## Technical Details

### BM25 Search Algorithm

LLMDoc uses the BM25Okapi algorithm from the `rank_bm25` library. Key characteristics:

- **Term frequency saturation**: Diminishing returns for repeated terms
- **Document length normalization**: Shorter documents aren't unfairly penalized
- **IDF weighting**: Rare terms are weighted higher than common ones

The implementation is thread-safe using `threading.RLock()` for concurrent access.

### Chunking Strategy

Documents are chunked using a multi-level approach:

1. **Paragraph splitting**: First split on double newlines (`\n\n`)
2. **Sentence-boundary aware**: Long paragraphs split at `.!?` followed by whitespace
3. **Overlap**: 100 character overlap between chunks maintains context

Configuration:
- `chunk_size`: 500 characters (default)
- `chunk_overlap`: 100 characters (default)

### Database Schema

DuckDB stores documents with this schema:

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
```

Indexes on `source_url` and `source_name` for efficient filtering.

### Concurrency Model

LLMDoc supports multiple concurrent instances:

- **Read operations**: Multiple instances can search simultaneously (read-only DuckDB mode)
- **Write operations**: Single instance holds exclusive lock during refresh
- **Graceful handling**: If refresh is locked, operation skips with status message

Document fetching uses `asyncio.Semaphore` to limit concurrent HTTP requests (default: 5).

### Stopwords

213 English stopwords are filtered during tokenization, including:
- Articles: a, an, the
- Prepositions: in, on, at, by, for, with, about, etc.
- Pronouns: I, you, he, she, it, we, they, etc.
- Auxiliaries: is, are, was, were, be, been, being, etc.
- Common verbs: have, has, had, do, does, did, etc.

## Installation

```bash
# Run directly (no install needed)
uvx llmdoc

# Or install as tool
uv tool install llmdoc

# Or add to project
uv add llmdoc
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
  "max_concurrent_fetches": 5
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

| Tool | Description |
|------|-------------|
| `search_docs(query, limit, source)` | Search documentation and return relevant passages with source URLs. Optional `source` parameter filters by source name (e.g., `fast_mcp`) |
| `get_doc(url)` | Get the full content of a document by its URL (as returned by `search_docs`) |
| `get_doc_excerpt(url, query, max_chunks, context_chars)` | Get relevant excerpts from a large document matching a query |
| `list_sources()` | List all configured documentation sources with statistics |
| `refresh_sources()` | Manually trigger a refresh of all documentation |

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

Use `get_doc` to retrieve the complete content of a document:

```
LLM: [calls get_doc("https://ai.pydantic.dev/agents.md")]

Result:
{
  "title": "Agents",
  "content": "# Agents\n\nAgents are the primary interface for interacting with LLMs in PydanticAI...",
  "url": "https://ai.pydantic.dev/agents.md",
  "source": "pydantic_ai",
  "source_url": "https://ai.pydantic.dev/llms.txt"
}
```

## License

MIT License - see [LICENSE](LICENSE) file.
