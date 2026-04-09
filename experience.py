"""E1: Experience Memory — records repair outcomes and retrieves relevant past fixes.

Each experience entry captures:
  task_summary    — what the agent was trying to do
  error_pattern   — the error message / type (normalised)
  fix_strategy    — what action resolved it
  tool_name       — which tool was involved
  success         — whether the fix worked
  success_rate    — rolling average across all entries for this pattern
  use_count       — how many times this experience has been retrieved
  last_used       — ISO timestamp
"""

import json
import os
import re
from datetime import datetime, timezone

import config

_EXPERIENCE_FILE = os.path.join(config.BASE_DIR, "data", "experiences.json")

# Max entries kept in memory / on disk (evict oldest on overflow)
_MAX_ENTRIES = 500
# How many experiences to surface per turn
_TOP_K = 3


class ExperienceStore:
    """Persistent store for repair experiences with keyword-based retrieval."""

    def __init__(self, filepath: str = _EXPERIENCE_FILE):
        self.filepath = filepath
        self._data = self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def record(
        self,
        task_summary: str,
        error_pattern: str,
        fix_strategy: str,
        tool_name: str,
        success: bool,
    ) -> dict:
        """Add or update an experience entry."""
        # Normalise the error pattern for deduplication
        norm_key = _normalise(error_pattern)

        # Update existing entry if same pattern + tool
        for entry in self._data["experiences"]:
            if entry["norm_key"] == norm_key and entry["tool_name"] == tool_name:
                n = entry["use_count"]
                entry["success_rate"] = (entry["success_rate"] * n + int(success)) / (n + 1)
                entry["use_count"] += 1
                entry["last_used"] = _now()
                if success:
                    entry["fix_strategy"] = fix_strategy  # promote better strategy
                self._flush()
                return entry

        entry = {
            "task_summary": task_summary[:200],
            "error_pattern": error_pattern[:300],
            "norm_key": norm_key,
            "fix_strategy": fix_strategy[:400],
            "tool_name": tool_name,
            "success": success,
            "success_rate": 1.0 if success else 0.0,
            "use_count": 1,
            "last_used": _now(),
        }
        self._data["experiences"].append(entry)

        # Evict oldest entries if over limit
        if len(self._data["experiences"]) > _MAX_ENTRIES:
            self._data["experiences"].sort(key=lambda e: e["last_used"])
            self._data["experiences"] = self._data["experiences"][-_MAX_ENTRIES:]

        self._flush()
        return entry

    def retrieve(self, query: str, top_k: int = _TOP_K) -> list[dict]:
        """Return the most relevant experiences for a given query."""
        if not self._data["experiences"]:
            return []

        query_tokens = set(_tokenise(query))
        scored = []
        for entry in self._data["experiences"]:
            candidate = f"{entry['error_pattern']} {entry['tool_name']} {entry['task_summary']}"
            overlap = len(query_tokens & set(_tokenise(candidate)))
            if overlap > 0:
                # Boost high success-rate entries
                score = overlap + entry["success_rate"] * 0.5
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def format_for_prompt(self, experiences: list[dict]) -> str:
        """Format retrieved experiences as a system-prompt snippet."""
        if not experiences:
            return ""
        lines = ["## Relevant Past Experiences\n"]
        for i, e in enumerate(experiences, 1):
            sr = f"{e['success_rate'] * 100:.0f}%"
            lines.append(
                f"{i}. **Tool:** {e['tool_name']} | **Success rate:** {sr}\n"
                f"   **Error pattern:** {e['error_pattern']}\n"
                f"   **Fix strategy:** {e['fix_strategy']}\n"
            )
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"experiences": []}

    def _flush(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        tmp = self.filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.filepath)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise(text: str) -> str:
    """Lower-case, strip numbers/paths, keep meaningful tokens."""
    text = text.lower()
    text = re.sub(r"[\\/][^\s]+", " PATH ", text)   # file paths
    text = re.sub(r"\d+", "N", text)                 # numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]\w*", text.lower())
