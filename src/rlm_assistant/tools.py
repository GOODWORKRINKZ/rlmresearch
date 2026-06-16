"""Custom development tools for RLM REPL.

Provides workspace-sandboxed file operations, command execution, and code search.
Tools are injected into RLM via custom_tools parameter.
"""

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

# --- Mapping from RLM vscode tool names → VS Code Copilot tool names ---
RLM_TO_VSCODE_TOOL_NAME: dict[str, str] = {
    "vscode_run_terminal": "run_in_terminal",
    "vscode_read_file": "read_file",
    "vscode_edit_file": "replace_string_in_file",
    "vscode_search": "semantic_search",
    "vscode_askQuestions": "vscode_askQuestions",
}


def map_rlm_tool_call_to_vscode(rlm_call: dict) -> dict:
    """Convert an RLM recorded tool call to OpenAI tool_call format for VS Code.

    Args:
        rlm_call: {"id": str, "name": str, "args": dict}

    Returns:
        OpenAI-format tool_call dict.
    """
    rlm_name = rlm_call["name"]
    vscode_name = RLM_TO_VSCODE_TOOL_NAME.get(rlm_name, rlm_name)
    args = rlm_call.get("args", {})

    # Add missing required args for specific tools
    if rlm_name == "vscode_run_terminal":
        args.setdefault("goal", args.get("explanation", "Execute command"))
        args.setdefault("mode", "sync")

    return {
        "id": rlm_call.get("id", f"call_{uuid.uuid4().hex[:12]}"),
        "type": "function",
        "function": {
            "name": vscode_name,
            "arguments": json.dumps(args),
        },
    }


def _strip_vscode_context(text: str) -> str:
    """Remove VS Code internal context blocks from tool result content.

    VS Code injects <context>, <editorContext>, <reminderInstructions> blocks
    into tool results. These are irrelevant for RLM and bloat the context.
    """
    import re
    # Remove <context>...</context> blocks
    text = re.sub(r'<context>.*?</context>\s*', '', text, flags=re.DOTALL)
    # Remove <editorContext>...</editorContext> blocks
    text = re.sub(r'<editorContext>.*?</editorContext>\s*', '', text, flags=re.DOTALL)
    # Remove <reminderInstructions>...</reminderInstructions> blocks
    text = re.sub(r'<reminderInstructions>.*?</reminderInstructions>\s*', '', text, flags=re.DOTALL)
    # Remove <userRequest>...</userRequest> blocks (already captured separately)
    text = re.sub(r'<userRequest>.*?</userRequest>\s*', '', text, flags=re.DOTALL)
    return text.strip()


# Max chars per tool result to keep RLM context manageable
MAX_TOOL_RESULT_CHARS = 3000


def format_tool_results_for_rlm(vscode_messages: list, id_map: dict[str, dict]) -> str:
    """Format VS Code tool result messages into a prompt for RLM.

    Truncates large results and strips VS Code internal context to keep
    the RLM context window manageable (< 30k chars total).

    Args:
        vscode_messages: Messages with role='tool' from VS Code.
        id_map: Mapping from VS Code tool_call_id → RLM tool call info.

    Returns:
        Formatted string for RLM consumption.
    """
    parts = []
    total_len = 0
    max_total = 30_000  # Hard cap on total tool results text

    for msg in vscode_messages:
        if msg.role != "tool":
            continue
        call_info = id_map.get(msg.tool_call_id, {})
        name = call_info.get("name", "unknown")
        args = call_info.get("args", {})
        content = msg.content or "(no output)"

        # Strip VS Code internal context blocks
        content = _strip_vscode_context(content)

        # Truncate individual result
        if len(content) > MAX_TOOL_RESULT_CHARS:
            content = content[:MAX_TOOL_RESULT_CHARS] + f"\n... [truncated, {len(content)} chars total]"

        part = f"Result of {name}({json.dumps(args, ensure_ascii=False)}):\n{content}"

        # Check total budget
        if total_len + len(part) > max_total:
            parts.append(f"\n... [remaining tool results omitted — {len(vscode_messages) - len(parts)} results total]")
            break
        parts.append(part)
        total_len += len(part)

    return "\n\n".join(parts) if parts else "(no tool results)"


def build_custom_tools(workspace_dir: str, include_local_tools: bool = True) -> dict:
    """Build custom tools dict for RLM injection.

    Args:
        workspace_dir: Root directory to sandbox all file operations to.
        include_local_tools: If True, include local filesystem tools (read_file, etc).
            If False, only include vscode_* deferred tools — RLM must go through VS Code
            for all operations. Use False when RLM runs on a remote server without
            filesystem access.

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

    # --- VS Code tool call tracker ---
    # When RLM calls a vscode_* tool, the call is recorded here instead of executing.
    # The server reads this list after RLM finishes and returns tool_calls to VS Code.
    _vscode_tool_calls: list[dict] = []
    # Map from OpenAI-format tool_call_id → {name, args} for result routing
    _vscode_id_map: dict[str, dict] = {}

    def _record_tool_call(name: str, args: dict) -> str:
        """Record a vscode tool call with a unique ID. Returns placeholder text."""
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        call_record = {"id": call_id, "name": name, "args": args}
        _vscode_tool_calls.append(call_record)
        _vscode_id_map[call_id] = {"name": name, "args": args}
        return f"[TOOL_REQUESTED: {name}({json.dumps(args, ensure_ascii=False)})] — waiting for VS Code to execute"

    def _vscode_run_terminal(command: str, explanation: str = "") -> str:
        """Execute a shell command in VS Code terminal."""
        return _record_tool_call("vscode_run_terminal", {"command": command, "explanation": explanation})

    def _vscode_read_file(filePath: str, startLine: int = 1, endLine: int = 9999) -> str:
        """Read a file via VS Code."""
        return _record_tool_call("vscode_read_file", {"filePath": filePath, "startLine": startLine, "endLine": endLine})

    def _vscode_edit_file(filePath: str, oldString: str, newString: str) -> str:
        """Edit a file via VS Code."""
        return _record_tool_call("vscode_edit_file", {"filePath": filePath, "oldString": oldString, "newString": newString})

    def _vscode_search(query: str, path: str = "") -> str:
        """Search workspace via VS Code."""
        return _record_tool_call("vscode_search", {"query": query, "path": path})

    def _vscode_ask_questions(questions: list[dict]) -> str:
        """Ask the user questions via VS Code interactive dialog.

        Args:
            questions: List of question dicts, each with:
                - header (str): Short identifier for the question (max 50 chars)
                - question (str): The question text (max 200 chars)
                - options (list[dict], optional): List of {label, description, recommended}
                - multiSelect (bool, optional): Allow multiple selections
                - allowFreeformInput (bool, optional): Allow free text (default True)

        Returns:
            Placeholder string. The actual tool call is deferred to VS Code.
        """
        # Validate and clean questions
        cleaned = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            item = {
                "header": str(q.get("header", "Question"))[:50],
                "question": str(q.get("question", ""))[:200],
            }
            if "options" in q:
                item["options"] = q["options"]
            if "multiSelect" in q:
                item["multiSelect"] = bool(q["multiSelect"])
            if "allowFreeformInput" in q:
                item["allowFreeformInput"] = bool(q["allowFreeformInput"])
            cleaned.append(item)

        if not cleaned:
            return "ERROR: No valid questions provided"

        return _record_tool_call("vscode_askQuestions", {"questions": cleaned})

    def get_vscode_tool_calls() -> list[dict]:
        """Return pending VS Code tool calls."""
        return list(_vscode_tool_calls)

    def get_vscode_id_map() -> dict[str, dict]:
        """Return the tool_call_id → {name, args} mapping."""
        return dict(_vscode_id_map)

    def clear_vscode_tool_calls():
        """Clear pending tool calls and ID map (call before each RLM run)."""
        _vscode_tool_calls.clear()
        _vscode_id_map.clear()

    result = {}

    # Local tools (execute on server) — only when RLM has filesystem access
    if include_local_tools:
        result.update({
            "read_file": {
                "tool": _read_file,
                "description": "Read file contents from workspace. Path relative to workspace root. Usage: read_file('src/main.py')",
            },
            "write_file": {
                "tool": _write_file,
                "description": "Write content to file in workspace. Creates parent dirs. Usage: write_file('output/result.txt', data)",
            },
            "run_command": {
                "tool": _run_command,
                "description": "Run shell command in workspace (30s timeout). CWD=workspace root. Usage: run_command('pytest tests/')",
            },
            "search_code": {
                "tool": _search_code,
                "description": "Regex search in codebase directory. Path MUST be a directory. Usage: search_code('def main', 'src/')",
            },
        })

    # VS Code tools (deferred — executed by VS Code, not server)
    result.update({
        "vscode_run_terminal": {
            "tool": _vscode_run_terminal,
            "description": "Execute a shell command in VS Code terminal. Use for git, npm, pip, or any system command that needs VS Code execution context. Usage: vscode_run_terminal('git push origin main', 'Push changes to remote')",
        },
        "vscode_read_file": {
            "tool": _vscode_read_file,
            "description": "Read a file via VS Code (absolute path). Usage: vscode_read_file('/home/ros2/project/src/main.py', startLine=1, endLine=50)",
        },
        "vscode_edit_file": {
            "tool": _vscode_edit_file,
            "description": "Edit a file via VS Code. Provide exact old and new strings. Usage: vscode_edit_file('/path/to/file.py', 'old code', 'new code')",
        },
        "vscode_search": {
            "tool": _vscode_search,
            "description": "Search workspace via VS Code search. Usage: vscode_search('functionName', 'src/')",
        },
        "vscode_askQuestions": {
            "tool": _vscode_ask_questions,
            "description": (
                "Ask the user questions via VS Code interactive dialog. "
                "Use this to gather user input, confirm decisions, or present choices. "
                "Each question has a header (short ID), question text, and optional options list. "
                "Returns immediately — the actual dialog is shown by VS Code. "
                "Usage: vscode_askQuestions([{'header': 'Approach', 'question': 'Which approach?', "
                "'options': [{'label': 'Option A'}, {'label': 'Option B'}]}])"
            ),
        },
    })

    # Tracker accessors (for server)
    result.update({
        "_get_vscode_tool_calls": {"tool": get_vscode_tool_calls, "description": "Internal: get pending VS Code tool calls"},
        "_get_vscode_id_map": {"tool": get_vscode_id_map, "description": "Internal: get tool_call_id to {name,args} mapping"},
        "_clear_vscode_tool_calls": {"tool": clear_vscode_tool_calls, "description": "Internal: clear pending tool calls and id map"},
    })

    # Mimo consultant tool (direct API call, not through RLM backends)
    mimo_api_key = os.getenv("MIMO_API_KEY", "")
    mimo_base_url = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

    if mimo_api_key:
        import httpx

        def _consult_mimo(query: str) -> str:
            """Consult Mimo (Xiaomi) for a second opinion or alternative approach."""
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{mimo_base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {mimo_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": mimo_model,
                            "messages": [{"role": "user", "content": query}],
                            "max_tokens": 2048,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                return f"ERROR consulting Mimo: {e}"

        result["consult_mimo"] = {
            "tool": _consult_mimo,
            "description": "Consult Mimo (Xiaomi) for a second opinion, alternative approach, or cross-check. Usage: consult_mimo('What do you think about this architecture?')",
        }

    return result
