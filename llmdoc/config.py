"""Configuration handling for LLMDoc."""

import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class Source:
    """A documentation source with a name and URL."""

    name: str
    url: str

    @classmethod
    def parse(cls, source_str: str) -> "Source":
        """Parse a source string in format 'name:url' or just 'url'.

        Args:
            source_str: Source string like 'fast_mcp:https://example.com/llms.txt'
                       or just 'https://example.com/llms.txt'

        Returns:
            Source object with name and url.
        """
        source_str = source_str.strip()

        # Check if it has a name prefix (name:url format)
        # We need to be careful not to split on the : in https://
        if "://" in source_str:
            # Find if there's a name before the protocol
            protocol_pos = source_str.find("://")
            prefix = source_str[:protocol_pos]

            # Check if the prefix contains a name (e.g., "fast_mcp:https")
            if ":" in prefix:
                name_end = prefix.rfind(":")
                name = prefix[:name_end]
                url = source_str[name_end + 1 :]
                return cls(name=name, url=url)
            else:
                # No name, just the URL
                url = source_str
                # Generate name from domain
                parsed = urlparse(url)
                name = parsed.netloc.replace(".", "_").replace("-", "_")
                return cls(name=name, url=url)
        else:
            # No protocol, might be a local file or invalid
            # Treat as URL and generate name
            name = Path(source_str).stem.replace(".", "_").replace("-", "_")
            return cls(name=name, url=source_str)


@dataclass
class Config:
    """Configuration for LLMDoc server."""

    sources: list[Source] = field(default_factory=list)
    db_path: str = "~/.llmdoc/index.db"
    refresh_interval_hours: int = 6
    max_concurrent_fetches: int = 5
    skip_startup_refresh: bool = False

    def __post_init__(self) -> None:
        """Expand paths and validate values after initialization."""
        self.db_path = os.path.expanduser(self.db_path)
        # Validate refresh_interval_hours (1 hour minimum, 168 hours = 1 week maximum)
        self.refresh_interval_hours = max(1, min(168, self.refresh_interval_hours))
        # Validate max_concurrent_fetches (1-20)
        self.max_concurrent_fetches = max(1, min(20, self.max_concurrent_fetches))

    @property
    def db_dir(self) -> Path:
        """Get the directory containing the database."""
        return Path(self.db_path).parent


def load_config() -> Config:
    """Load configuration from environment variables or config file.

    Priority:
    1. Environment variables (LLMDOC_SOURCES, LLMDOC_DB_PATH, LLMDOC_REFRESH_INTERVAL,
       LLMDOC_MAX_CONCURRENT, LLMDOC_SKIP_STARTUP_REFRESH)
    2. Config file (llmdoc.json in current directory)
    3. Default values

    Source format: 'name:url' or just 'url'
    Examples:
        - 'fast_mcp:https://gofastmcp.com/llms.txt'
        - 'pydantic_ai:https://ai.pydantic.dev/llms.txt'
        - 'https://example.com/llms.txt' (name auto-generated from domain)
    """
    sources: list[Source] = []
    db_path: str | None = None
    refresh_interval_hours: int | None = None
    max_concurrent_fetches: int | None = None
    skip_startup_refresh: bool | None = None

    # Try environment variables first
    env_sources = os.environ.get("LLMDOC_SOURCES")
    if env_sources:
        for s in env_sources.split(","):
            s = s.strip()
            if s:
                sources.append(Source.parse(s))

    env_db_path = os.environ.get("LLMDOC_DB_PATH")
    if env_db_path:
        db_path = env_db_path

    env_refresh = os.environ.get("LLMDOC_REFRESH_INTERVAL")
    if env_refresh:
        with contextlib.suppress(ValueError):
            refresh_interval_hours = int(env_refresh)

    env_max_concurrent = os.environ.get("LLMDOC_MAX_CONCURRENT")
    if env_max_concurrent:
        with contextlib.suppress(ValueError):
            max_concurrent_fetches = int(env_max_concurrent)

    env_skip_startup = os.environ.get("LLMDOC_SKIP_STARTUP_REFRESH")
    if env_skip_startup:
        skip_startup_refresh = env_skip_startup.lower() in ("true", "1", "yes")

    # Try config file if no sources from env
    if not sources:
        config_file = Path("llmdoc.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    data = json.load(f)
                    if "sources" in data:
                        for s in data["sources"]:
                            if isinstance(s, str):
                                sources.append(Source.parse(s))
                            elif isinstance(s, dict) and "name" in s and "url" in s:
                                sources.append(Source(name=s["name"], url=s["url"]))
                    if "db_path" in data and db_path is None:
                        db_path = data["db_path"]
                    if "refresh_interval_hours" in data and refresh_interval_hours is None:
                        refresh_interval_hours = data["refresh_interval_hours"]
                    if "max_concurrent_fetches" in data and max_concurrent_fetches is None:
                        max_concurrent_fetches = data["max_concurrent_fetches"]
                    if "skip_startup_refresh" in data and skip_startup_refresh is None:
                        skip_startup_refresh = data["skip_startup_refresh"]
            except (json.JSONDecodeError, OSError):
                pass

    # Build config with defaults for missing values
    config_kwargs: dict = {"sources": sources}
    if db_path is not None:
        config_kwargs["db_path"] = db_path
    if refresh_interval_hours is not None:
        config_kwargs["refresh_interval_hours"] = refresh_interval_hours
    if max_concurrent_fetches is not None:
        config_kwargs["max_concurrent_fetches"] = max_concurrent_fetches
    if skip_startup_refresh is not None:
        config_kwargs["skip_startup_refresh"] = skip_startup_refresh

    return Config(**config_kwargs)
