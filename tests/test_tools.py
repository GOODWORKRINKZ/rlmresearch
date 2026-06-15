"""Tests for custom development tools."""

import tempfile
from pathlib import Path

from rlm_assistant.tools import build_custom_tools


class TestBuildCustomTools:
    """Test build_custom_tools factory function."""

    def test_returns_all_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            assert "read_file" in tools
            assert "write_file" in tools
            assert "run_command" in tools
            assert "search_code" in tools

    def test_tools_have_descriptions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            for name, entry in tools.items():
                assert "tool" in entry, f"{name} missing 'tool' key"
                assert "description" in entry, f"{name} missing 'description' key"
                assert callable(entry["tool"]), f"{name} tool is not callable"


class TestReadFile:
    """Test read_file tool."""

    def test_read_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            Path(tmpdir, "test.txt").write_text("hello world")
            result = tools["read_file"]["tool"]("test.txt")
            assert result == "hello world"

    def test_read_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            result = tools["read_file"]["tool"]("nope.txt")
            assert "ERROR" in result

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            result = tools["read_file"]["tool"]("/etc/passwd")
            assert "ERROR" in result or "outside" in result.lower()


class TestWriteFile:
    """Test write_file tool."""

    def test_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            result = tools["write_file"]["tool"]("out.txt", "data")
            assert "Wrote" in result
            assert Path(tmpdir, "out.txt").read_text() == "data"

    def test_write_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            result = tools["write_file"]["tool"]("sub/dir/file.txt", "deep")
            assert "Wrote" in result
            assert Path(tmpdir, "sub/dir/file.txt").read_text() == "deep"

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            result = tools["write_file"]["tool"]("/tmp/hacked.txt", "bad")
            assert "ERROR" in result or "outside" in result.lower()


class TestRunCommand:
    """Test run_command tool."""

    def test_echo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            result = tools["run_command"]["tool"]("echo test123")
            assert "test123" in result

    def test_empty_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            result = tools["run_command"]["tool"]("true")
            assert result == "(no output)"


class TestSearchCode:
    """Test search_code tool."""

    def test_finds_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            Path(tmpdir, "main.py").write_text("def hello():\n    pass\n")
            result = tools["search_code"]["tool"]("def hello", ".")
            assert "hello" in result
            assert "main.py" in result

    def test_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            Path(tmpdir, "main.py").write_text("x = 1\n")
            result = tools["search_code"]["tool"]("nonexistent", ".")
            assert "no matches" in result

    def test_limits_to_100(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = build_custom_tools(tmpdir)
            # Create a file with many matching lines
            lines = "match_line\n" * 150
            Path(tmpdir, "big.py").write_text(lines)
            result = tools["search_code"]["tool"]("match_line", ".")
            assert "truncated" in result
