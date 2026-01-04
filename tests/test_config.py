"""Tests for llmdoc.config module."""

import json
import os
from pathlib import Path

from llmdoc.config import Config, Source, load_config


class TestSource:
    """Tests for Source class."""

    def test_parse_named_source(self):
        """Test parsing a named source string."""
        source = Source.parse("fast_mcp:https://example.com/llms.txt")
        assert source.name == "fast_mcp"
        assert source.url == "https://example.com/llms.txt"

    def test_parse_unnamed_source(self):
        """Test parsing an unnamed source string."""
        source = Source.parse("https://example.com/llms.txt")
        assert source.name == "example_com"
        assert source.url == "https://example.com/llms.txt"

    def test_parse_source_with_subdomain(self):
        """Test parsing source with subdomain."""
        source = Source.parse("https://docs.example.com/llms.txt")
        assert source.name == "docs_example_com"
        assert source.url == "https://docs.example.com/llms.txt"

    def test_parse_source_with_hyphen(self):
        """Test parsing source with hyphen in domain."""
        source = Source.parse("https://my-docs.example.com/llms.txt")
        assert source.name == "my_docs_example_com"
        assert source.url == "https://my-docs.example.com/llms.txt"

    def test_parse_named_source_with_underscores(self):
        """Test parsing named source with underscores."""
        source = Source.parse("my_custom_name:https://example.com/llms.txt")
        assert source.name == "my_custom_name"
        assert source.url == "https://example.com/llms.txt"

    def test_parse_http_source(self):
        """Test parsing HTTP (non-HTTPS) source."""
        source = Source.parse("http://localhost:8000/llms.txt")
        # Port is part of netloc, colon preserved
        assert source.name == "localhost:8000"
        assert source.url == "http://localhost:8000/llms.txt"

    def test_parse_strips_whitespace(self):
        """Test that parsing strips whitespace."""
        source = Source.parse("  fast_mcp:https://example.com/llms.txt  ")
        assert source.name == "fast_mcp"
        assert source.url == "https://example.com/llms.txt"


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.sources == []
        assert config.db_path.endswith(".llmdoc/index.db")
        assert config.refresh_interval_hours == 6
        assert config.max_concurrent_fetches == 5

    def test_config_with_sources(self, sample_sources):
        """Test configuration with sources."""
        config = Config(sources=sample_sources)
        assert len(config.sources) == 2
        assert config.sources[0].name == "source_a"

    def test_config_db_dir(self, temp_db_path):
        """Test db_dir property."""
        config = Config(db_path=temp_db_path)
        assert config.db_dir == Path(temp_db_path).parent

    def test_config_expands_user_path(self):
        """Test that ~ is expanded in db_path."""
        config = Config(db_path="~/.llmdoc/test.db")
        assert "~" not in config.db_path
        assert config.db_path.startswith("/")


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_env(self, monkeypatch, tmp_path):
        """Test loading configuration from environment variables."""
        db_path = str(tmp_path / "test.db")
        monkeypatch.setenv("LLMDOC_SOURCES", "test:https://example.com/llms.txt")
        monkeypatch.setenv("LLMDOC_DB_PATH", db_path)
        monkeypatch.setenv("LLMDOC_REFRESH_INTERVAL", "12")
        monkeypatch.setenv("LLMDOC_MAX_CONCURRENT", "10")

        config = load_config()

        assert len(config.sources) == 1
        assert config.sources[0].name == "test"
        assert config.db_path == db_path
        assert config.refresh_interval_hours == 12
        assert config.max_concurrent_fetches == 10

    def test_load_multiple_sources_from_env(self, monkeypatch):
        """Test loading multiple sources from environment."""
        monkeypatch.setenv(
            "LLMDOC_SOURCES", "source_a:https://a.example.com/llms.txt,source_b:https://b.example.com/llms.txt"
        )

        config = load_config()

        assert len(config.sources) == 2
        assert config.sources[0].name == "source_a"
        assert config.sources[1].name == "source_b"

    def test_load_from_json_file(self, monkeypatch, tmp_path):
        """Test loading configuration from JSON file."""
        # Clear any env vars
        monkeypatch.delenv("LLMDOC_SOURCES", raising=False)
        monkeypatch.delenv("LLMDOC_DB_PATH", raising=False)

        # Create config file
        config_data = {
            "sources": [{"name": "test", "url": "https://example.com/llms.txt"}],
            "db_path": str(tmp_path / "test.db"),
            "refresh_interval_hours": 3,
            "max_concurrent_fetches": 8,
        }
        config_file = tmp_path / "llmdoc.json"
        config_file.write_text(json.dumps(config_data))

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = load_config()
        finally:
            os.chdir(original_cwd)

        assert len(config.sources) == 1
        assert config.sources[0].name == "test"
        assert config.refresh_interval_hours == 3
        assert config.max_concurrent_fetches == 8

    def test_load_from_json_with_string_sources(self, monkeypatch, tmp_path):
        """Test loading sources as strings from JSON file."""
        monkeypatch.delenv("LLMDOC_SOURCES", raising=False)

        config_data = {"sources": ["test:https://example.com/llms.txt", "https://other.com/llms.txt"]}
        config_file = tmp_path / "llmdoc.json"
        config_file.write_text(json.dumps(config_data))

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = load_config()
        finally:
            os.chdir(original_cwd)

        assert len(config.sources) == 2
        assert config.sources[0].name == "test"
        assert config.sources[1].name == "other_com"

    def test_env_takes_precedence_over_file(self, monkeypatch, tmp_path):
        """Test that environment variables take precedence over config file."""
        # Set env var
        monkeypatch.setenv("LLMDOC_SOURCES", "env_source:https://env.example.com/llms.txt")

        # Create config file with different source
        config_data = {"sources": [{"name": "file_source", "url": "https://file.example.com/llms.txt"}]}
        config_file = tmp_path / "llmdoc.json"
        config_file.write_text(json.dumps(config_data))

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = load_config()
        finally:
            os.chdir(original_cwd)

        # Env should win
        assert len(config.sources) == 1
        assert config.sources[0].name == "env_source"

    def test_load_with_invalid_refresh_interval(self, monkeypatch):
        """Test that invalid refresh interval is ignored."""
        monkeypatch.setenv("LLMDOC_REFRESH_INTERVAL", "invalid")

        config = load_config()

        # Should use default
        assert config.refresh_interval_hours == 6

    def test_load_with_empty_sources(self, monkeypatch):
        """Test loading with no sources configured."""
        monkeypatch.delenv("LLMDOC_SOURCES", raising=False)

        config = load_config()

        assert config.sources == []
