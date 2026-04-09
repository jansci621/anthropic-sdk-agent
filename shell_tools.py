"""Shell tools: execute local system commands."""

import json
import shlex
import subprocess


# ── Tool Definitions (Anthropic API format) ─────────────────────────────────

SHELL_TOOLS = [
    {
        "name": "run_command",
        "description": (
            "Execute a local shell command and return its output. "
            "Use this to check system info (CPU, memory, disk, network), "
            "run scripts, list files, or any other local operation. "
            "Output is limited to 5000 characters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
            },
            "required": ["command"],
        },
    },
]


# ── Tool Dispatch ────────────────────────────────────────────────────────────

def handle_shell_tool(name: str, tool_input: dict) -> str:
    """Dispatch a shell tool call and return the JSON result string."""
    command = tool_input["command"]
    try:
        # Use shell=False to prevent command injection.
        # shlex.split handles quoted args safely on POSIX systems.
        cmd_list = shlex.split(command) if isinstance(command, str) else command
        result = subprocess.run(
            cmd_list,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 and result.stderr:
            output = f"[exit code {result.returncode}]\n{output}"
        return json.dumps(
            {"status": "ok", "command": command, "output": output[:5000]},
            ensure_ascii=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"status": "error", "command": command, "error": "Command timed out (30s)"},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "command": command, "error": str(e)},
            ensure_ascii=False,
        )
