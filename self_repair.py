"""L7: Self-repairing code for skill scripts.

When a skill script fails at runtime, this module:
1. Reads the original script source
2. Sends it to the LLM with the error context
3. Receives a fixed version
4. Backs up the original and writes the fix
5. Retries execution

Safety measures:
- Original file is backed up with .bak extension (never deleted)
- Repair attempts are limited per session (MAX_REPAIRS_PER_SCRIPT)
- Only repairs scripts under the skills/ directory
- User is notified of every repair
"""

import json
import os
import shutil
import time

import anthropic

import config

REPAIR_CODE_PROMPT = """You are a Python code repair specialist. A skill script failed during execution.

## Original Script
```python
{source}
```

## Error
```
{error}
```

## Tool Input
```json
{tool_input}
```

## Task
Fix the script so it handles this error correctly. Rules:
1. Keep the same `run(tool_input: dict)` interface
2. Preserve the docstring with tool schema (if any)
3. Add proper error handling for the failure case
4. Do NOT change the script's core logic — only fix the bug
5. Return the COMPLETE fixed script (not a diff, not a patch)

Output ONLY the fixed Python code, no explanations."""


# ── Config ────────────────────────────────────────────────────────────────────

MAX_REPAIRS_PER_SCRIPT = 3  # per session


class ScriptRepairer:
    """Repairs failing skill scripts using LLM-generated fixes."""

    def __init__(self, client: anthropic.Anthropic, model: str | None = None):
        self.client = client
        self.model = model or config.MODEL
        self._repair_count: dict[str, int] = {}  # script_path → count

    def can_repair(self, script_path: str) -> bool:
        """Check if this script is eligible for self-repair."""
        key = str(script_path)
        return self._repair_count.get(key, 0) < MAX_REPAIRS_PER_SCRIPT

    def repair(
        self,
        script_path: str,
        source: str,
        error: str,
        tool_input: dict,
    ) -> str | None:
        """Attempt to repair a script. Returns fixed source code or None."""
        key = str(script_path)

        if not self.can_repair(script_path):
            return None

        prompt = REPAIR_CODE_PROMPT.format(
            source=source,
            error=error,
            tool_input=json.dumps(tool_input, ensure_ascii=False, indent=2),
        )

        try:
            print(
                f"  {config.COLOR_SYSTEM}[Self-Repair] Asking LLM to fix {os.path.basename(script_path)}...{config.COLOR_RESET}",
                flush=True,
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system="You are a Python code repair specialist. Return ONLY valid Python code.",
                messages=[{"role": "user", "content": prompt}],
            )

            fixed = response.content[0].text.strip()
            # Strip markdown code fences if present
            if fixed.startswith("```python"):
                fixed = fixed[len("```python"):]
            elif fixed.startswith("```"):
                fixed = fixed[3:]
            if fixed.endswith("```"):
                fixed = fixed[:-3]
            fixed = fixed.strip()

            # Basic validation: must contain a `run` function
            if "def run(" not in fixed:
                print(
                    f"  {config.COLOR_SYSTEM}[Self-Repair] Rejected: fixed code has no run() function{config.COLOR_RESET}",
                    flush=True,
                )
                return None

            # Backup original
            backup_path = script_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(script_path, backup_path)

            # Write fixed version
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(fixed)

            self._repair_count[key] = self._repair_count.get(key, 0) + 1

            print(
                f"  {config.COLOR_SYSTEM}[Self-Repair] Fixed {os.path.basename(script_path)} "
                f"(attempt {self._repair_count[key]}/{MAX_REPAIRS_PER_SCRIPT}){config.COLOR_RESET}",
                flush=True,
            )

            return fixed

        except Exception as e:
            print(
                f"  {config.COLOR_SYSTEM}[Self-Repair] Failed: {e}{config.COLOR_RESET}",
                flush=True,
            )
            return None
