"""Skill system: standard-format skill loading, script registration, and dispatch."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Skill Protocol ──────────────────────────────────────────────────────────


class Skill:
    """A loaded skill with prompt, tools, and optional scripts."""

    # Class-level module cache: path_str → (mtime, module)
    _module_cache: dict[str, tuple[float, Any]] = {}

    def __init__(self, skill_dir: str):
        self.dir = Path(skill_dir)
        self.name: str = ""
        self.description: str = ""
        self.triggers: list[str] = []
        self.global_tools: list[str] = []  # names of global tools this skill uses
        self.depends_on: list[str] = []    # other skills this skill depends on
        self.prompt: str = ""
        self.tools: list[dict] = []  # Anthropic tool schemas (from scripts)
        self._scripts: dict[str, Path] = {}  # tool_name → script path
        self._state: dict = {}
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load(self):
        """Parse SKILL.md frontmatter + body, discover scripts."""
        skill_md = self.dir / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"No SKILL.md in {self.dir}")

        raw = skill_md.read_text(encoding="utf-8")
        metadata, body = self._parse_frontmatter(raw)

        self.name = metadata.get("name", self.dir.name)
        self.description = metadata.get("description", "")
        self.triggers = metadata.get("trigger", [])
        if isinstance(self.triggers, str):
            self.triggers = [self.triggers]
        self.global_tools = metadata.get("tools", [])
        self.depends_on = metadata.get("depends_on", [])
        if isinstance(self.depends_on, str):
            self.depends_on = [self.depends_on]
        self.prompt = body.strip()

        # Discover and register scripts
        script_names = metadata.get("scripts", [])
        if isinstance(script_names, str):
            script_names = [script_names]
        self._discover_scripts(script_names)

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from SKILL.md. Returns (metadata, body)."""
        metadata = {}
        if not text.startswith("---"):
            return metadata, text

        parts = text.split("---", 2)
        if len(parts) < 3:
            return metadata, text

        yaml_text = parts[1].strip()
        body = parts[2]

        # Minimal YAML parser (no dependency)
        for line in yaml_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if not key or not value:
                continue

            # Parse list values: [a, b, c]
            if value.startswith("[") and value.endswith("]"):
                items = [item.strip().strip("'\"") for item in value[1:-1].split(",")]
                metadata[key] = [item for item in items if item]
            else:
                metadata[key] = value.strip("'\"")

        return metadata, body

    # ── Script Discovery ──────────────────────────────────────────────────

    def _discover_scripts(self, script_names: list[str]):
        """Load scripts from script/ directory and register as tools."""
        script_dir = self.dir / "script"
        if not script_dir.exists():
            return

        # If specific scripts listed, only load those; otherwise load all .py
        if script_names:
            candidates = [script_dir / s for s in script_names]
        else:
            candidates = sorted(script_dir.glob("*.py"))

        for script_path in candidates:
            if not script_path.exists() or not script_path.name.endswith(".py"):
                continue
            tool_name, tool_schema = self._load_script(script_path)
            if tool_schema:
                self.tools.append(tool_schema)
                self._scripts[tool_name] = script_path

    @staticmethod
    def _load_script(script_path: Path) -> tuple[str, dict | None]:
        """Extract tool name and schema from a script file.

        Convention: script starts with a docstring containing JSON tool schema:
            \"\"\"Tool description.
            ---
            {"type":"object","properties":{...},"required":[...]}
            \"\"\"

        Or auto-generate a minimal schema from the docstring first line.
        """
        tool_name = script_path.stem
        source = script_path.read_text(encoding="utf-8")

        # Extract docstring
        docstring = ""
        schema_json = None
        if '"""' in source:
            start = source.index('"""') + 3
            end = source.index('"""', start)
            docstring = source[start:end].strip()
            if "---" in docstring:
                doc_part, schema_part = docstring.split("---", 1)
                docstring = doc_part.strip()
                try:
                    schema_json = json.loads(schema_part.strip())
                except json.JSONDecodeError:
                    pass

        description = docstring or f"Execute {tool_name} script"

        tool_schema = {
            "name": tool_name,
            "description": description,
            "input_schema": schema_json if schema_json else {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Arguments to pass to the script",
                    },
                },
                "required": [],
            },
        }

        return tool_name, tool_schema

    # ── Execution ─────────────────────────────────────────────────────────

    def handle(self, tool_name: str, tool_input: dict) -> str:
        """Execute a script tool and return the JSON result string."""
        script_path = self._scripts.get(tool_name)
        if not script_path:
            return json.dumps({"error": f"Script not found: {tool_name}"})

        try:
            mod = self._load_module(script_path)
            if not hasattr(mod, "run"):
                return json.dumps(
                    {"error": f"Script {tool_name} has no run() function"},
                    ensure_ascii=False,
                )

            result = mod.run(tool_input)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {"error": f"Script {tool_name} failed: {e}"},
                ensure_ascii=False,
            )

    def _load_module(self, script_path: Path) -> Any:
        """Load (or return cached) script module, reloading if the file changed."""
        key = str(script_path)
        mtime = script_path.stat().st_mtime
        cached = Skill._module_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        spec = importlib.util.spec_from_file_location(
            f"skill_{self.name}_{script_path.stem}", script_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Skill._module_cache[key] = (mtime, mod)
        return mod

    # ── Reference docs ────────────────────────────────────────────────────

    def get_reference(self) -> str:
        """Concatenate all reference docs for context injection."""
        ref_dir = self.dir / "reference"
        if not ref_dir.exists():
            return ""
        parts = []
        for f in sorted(ref_dir.glob("*")):
            if f.suffix in (".md", ".txt", ".json"):
                content = f.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"## {f.name}\n\n{content}")
        return "\n\n".join(parts)

    @property
    def tool_names(self) -> set[str]:
        return set(self._scripts.keys())

    # ── State Management ──────────────────────────────────────────────────

    @property
    def _state_file(self) -> Path:
        """Path to skill state file: data/skills/{name}_state.json"""
        import config
        return Path(config.BASE_DIR) / "data" / "skills" / f"{self.name}_state.json"

    def load_state(self) -> dict:
        """Load persisted skill state from disk."""
        if self._state_file.exists():
            try:
                self._state = json.loads(self._state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._state = {}
        return self._state

    def save_state(self):
        """Persist skill state to disk."""
        if not self._state:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self._state_file) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._state_file)

    def set_state(self, key: str, value):
        """Set a state value."""
        self._state[key] = value

    def get_state(self, key: str, default=None):
        """Get a state value."""
        return self._state.get(key, default)

    def clear_state(self):
        """Clear all state."""
        self._state = {}
        if self._state_file.exists():
            self._state_file.unlink()
