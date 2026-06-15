"""Custom development tools for RLM REPL.

Provides workspace-sandboxed file operations, command execution, and code search.
Tools are injected into RLM via custom_tools parameter.
"""

import os
import re
import subprocess
from pathlib import Path


def build_custom_tools(workspace_dir: str) -> dict:
    """Build custom tools dict for RLM injection.

    Args:
        workspace_dir: Root directory to sandbox all file operations to.

    Returns:
        Dict in RLM custom_tools format: {name: {"tool": callable, "description": str}}
    """
    workspace = Path(workspace_dir).resolve()

    def _resolve_safe(path: str) -> Path:
        """Resolve path and ensure it's within workspace."""
        resolved = (workspace / path).resolve()
        if not str(resolved).startswith(str(workspace)):
            raise ValueError(f"Path '{path}' is outside workspace directory")
        return resolved

    def _read_file(path: str) -> str:
        """Read file contents."""
        try:
            target = _resolve_safe(path)
            return target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"ERROR: {e}"

    def _write_file(path: str, content: str) -> str:
        """Write content to file, creating parent directories."""
        try:
            target = _resolve_safe(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def _run_command(cmd: str) -> str:
        """Run shell command with 30s timeout."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=str(workspace),
            )
            output = result.stdout + result.stderr
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out (30s limit)"
        except Exception as e:
            return f"ERROR: {e}"

    def _search_code(pattern: str, path: str = ".") -> str:
        """Regex search in codebase files."""
        try:
            target = _resolve_safe(path)
            if not target.is_dir():
                return f"ERROR: '{path}' is not a directory"

            extensions = {".py", ".ts", ".js", ".rs", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}
            matches = []
            regex = re.compile(pattern, re.IGNORECASE)

            for file_path in sorted(target.rglob("*")):
                if not file_path.is_file():
                    continue
                if file_path.suffix not in extensions:
                    continue
                try:
                    rel_path = file_path.relative_to(workspace)
                    for line_num, line in enumerate(file_path.read_text(errors="replace").splitlines(), 1):
                        if regex.search(line):
                            matches.append(f"{rel_path}:{line_num}: {line.strip()}")
                            if len(matches) >= 100:
                                return "\n".join(matches) + "\n... (truncated at 100 matches)"
                except Exception:
                    continue

            return "\n".join(matches) if matches else "(no matches found)"
        except Exception as e:
            return f"ERROR: {e}"

    return {
        "read_file": {
            "tool": _read_file,
            "description": "Read file contents. Usage: read_file('path/to/file')",
        },
        "write_file": {
            "tool": _write_file,
            "description": "Write content to file. Creates dirs. Usage: write_file('path/to/file', 'content')",
        },
        "run_command": {
            "tool": _run_command,
            "description": "Run shell command (30s timeout). Returns stdout+stderr. Usage: run_command('ls -la')",
        },
        "search_code": {
            "tool": _search_code,
            "description": "Search codebase for pattern. Usage: search_code('def main', 'src/')",
        },
    }
