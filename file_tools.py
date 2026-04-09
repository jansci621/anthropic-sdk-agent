"""File tools: read, write, glob (find files), grep (search content)."""

import glob as _glob
import json
import os
import re

# ── Path Safety ──────────────────────────────────────────────────────────────

# Default sandbox root (used in CLI mode). Web mode overrides per-session.
_DEFAULT_WORKSPACE = os.path.realpath(os.getcwd())


def _safe_path(path: str, workspace_root: str | None = None) -> str:
    """Resolve path and verify it stays within the workspace root.

    Args:
        path: The path to resolve.
        workspace_root: Sandbox root. Defaults to CWD if None.
    Raises PermissionError if the resolved path escapes the workspace.
    """
    root = os.path.realpath(workspace_root or _DEFAULT_WORKSPACE)
    # Resolve relative paths against workspace_root (not CWD)
    if workspace_root and not os.path.isabs(path):
        resolved = os.path.realpath(os.path.join(root, path))
    else:
        resolved = os.path.realpath(os.path.abspath(path))
    if not resolved.startswith(root + os.sep) and resolved != root:
        raise PermissionError(
            f"Access denied: path '{path}' resolves outside workspace "
            f"({root})"
        )
    return resolved


# ── Tool Definitions (Anthropic API format) ─────────────────────────────────

FILE_TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a file and return its content. Supports line range. "
            "Auto-detects text encoding. Returns up to 20000 characters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based, default 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default 200)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates parent directories if needed. "
            "Overwrites existing files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "Find files matching a glob pattern. "
            "e.g. '**/*.py' finds all Python files recursively. "
            "Returns matching file paths sorted by modification time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts', '*.json')",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current directory)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_content",
        "description": (
            "Search for a regex pattern in file contents. "
            "Returns matching lines with file path and line numbers. "
            "Supports file type filtering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in",
                },
                "glob": {
                    "type": "string",
                    "description": "File pattern filter (e.g. '*.py', '*.{js,ts}')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 50)",
                },
            },
            "required": ["pattern"],
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

_MAX_READ = 20000


def _read_file(path: str, offset: int = 1, limit: int = 200, workspace_root: str | None = None) -> str:
    """Read file with line range support."""
    try:
        abs_path = _safe_path(path, workspace_root)
    except PermissionError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    if not os.path.isfile(abs_path):
        return json.dumps({"error": f"File not found: {abs_path}"}, ensure_ascii=False)

    # Check line count for large files
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()

    total = len(all_lines)
    start = max(1, offset) - 1
    end = min(total, start + limit)
    lines = all_lines[start:end]

    output = ""
    for i, line in enumerate(lines, start=start + 1):
        output += f"{i}\t{line}"

    output = output[:_MAX_READ]
    return json.dumps(
        {"path": abs_path, "total_lines": total, "showing": f"{start+1}-{end}", "content": output},
        ensure_ascii=False,
    )


def _write_file(path: str, content: str, workspace_root: str | None = None) -> str:
    """Write content to file, creating dirs as needed."""
    try:
        abs_path = _safe_path(path, workspace_root)
    except PermissionError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    size = os.path.getsize(abs_path)
    return json.dumps(
        {"status": "ok", "path": abs_path, "bytes_written": size},
        ensure_ascii=False,
    )


def _list_files(pattern: str, path: str = ".", workspace_root: str | None = None) -> str:
    """Glob files and return sorted by mtime."""
    root = os.path.realpath(workspace_root or _DEFAULT_WORKSPACE)
    # Resolve relative paths against workspace root
    if workspace_root and not os.path.isabs(path):
        abs_base = os.path.join(root, path)
    else:
        abs_base = os.path.abspath(path)
    matches = _glob.glob(os.path.join(abs_base, pattern), recursive=True)
    # Sort by modification time
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    # Return relative paths
    results = [os.path.relpath(m, abs_base) for m in matches[:100]]
    return json.dumps(
        {"base": abs_base, "pattern": pattern, "count": len(results), "files": results},
        ensure_ascii=False,
    )


def _search_content(pattern: str, path: str = ".", glob_pattern: str = None, max_results: int = 50, workspace_root: str | None = None) -> str:
    """Grep-like content search."""
    root = os.path.realpath(workspace_root or _DEFAULT_WORKSPACE)
    # Resolve relative paths against workspace root
    if workspace_root and not os.path.isabs(path):
        abs_path = os.path.join(root, path)
    else:
        abs_path = os.path.abspath(path)
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex: {e}"}, ensure_ascii=False)

    results = []
    targets = []

    if os.path.isfile(abs_path):
        targets = [abs_path]
    elif os.path.isdir(abs_path):
        g = glob_pattern or "*"
        targets = _glob.glob(os.path.join(abs_path, "**", g), recursive=True)
        targets = [t for t in targets if os.path.isfile(t)]

    for filepath in targets:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append({
                            "file": os.path.relpath(filepath, abs_path) if os.path.isdir(abs_path) else os.path.basename(filepath),
                            "line": i,
                            "text": line.strip()[:200],
                        })
                        if len(results) >= max_results:
                            break
        except (OSError, PermissionError):
            continue
        if len(results) >= max_results:
            break

    return json.dumps(
        {"pattern": pattern, "path": abs_path, "matches": len(results), "results": results},
        ensure_ascii=False,
    )


# ── Tool Dispatch ────────────────────────────────────────────────────────────

def handle_file_tool(name: str, tool_input: dict, workspace_root: str | None = None) -> str:
    """Dispatch a file tool call and return the JSON result string."""
    if name == "read_file":
        return _read_file(
            path=tool_input["path"],
            offset=tool_input.get("offset", 1),
            limit=tool_input.get("limit", 200),
            workspace_root=workspace_root,
        )

    if name == "write_file":
        return _write_file(
            path=tool_input["path"],
            content=tool_input["content"],
            workspace_root=workspace_root,
        )

    if name == "list_files":
        return _list_files(
            pattern=tool_input["pattern"],
            path=tool_input.get("path", "."),
            workspace_root=workspace_root,
        )

    if name == "search_content":
        return _search_content(
            pattern=tool_input["pattern"],
            path=tool_input.get("path", "."),
            glob_pattern=tool_input.get("glob"),
            max_results=tool_input.get("max_results", 50),
            workspace_root=workspace_root,
        )

    return json.dumps({"error": f"Unknown file tool: {name}"})
