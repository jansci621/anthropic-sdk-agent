"""E3: Tool Reliability Tracking — records per-tool success/failure counts.

Persists to data/tool_stats.json. At session start the reliability ranking is
injected into the system prompt so the agent prefers reliable tools and is
cautious with unreliable ones.
"""

import json
import os
import threading
from datetime import datetime, timezone

import config

_STATS_FILE = os.path.join(config.BASE_DIR, "data", "tool_stats.json")

# Only surface tools with enough samples to be meaningful
_MIN_SAMPLES = 3


class ToolStats:
    """Tracks per-tool call success/failure rates."""

    def __init__(self, filepath: str = _STATS_FILE):
        self.filepath = filepath
        self._data = self._load()
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def record(self, tool_name: str, success: bool):
        """Record one tool call outcome (thread-safe)."""
        with self._lock:
            stats = self._data["tools"].setdefault(tool_name, {
                "success": 0, "failure": 0, "last_used": ""
            })
            if success:
                stats["success"] += 1
            else:
                stats["failure"] += 1
            stats["last_used"] = datetime.now(timezone.utc).isoformat()
            try:
                self._flush()
            except OSError:
                pass  # non-critical; will retry on next call

    def reliability(self, tool_name: str) -> float | None:
        """Return success rate [0..1] or None if not enough data."""
        stats = self._data["tools"].get(tool_name)
        if not stats:
            return None
        total = stats["success"] + stats["failure"]
        if total < _MIN_SAMPLES:
            return None
        return stats["success"] / total

    def format_for_prompt(self) -> str:
        """Return a compact reliability table for system prompt injection."""
        rows = []
        for name, stats in self._data["tools"].items():
            total = stats["success"] + stats["failure"]
            if total < _MIN_SAMPLES:
                continue
            rate = stats["success"] / total
            label = _label(rate)
            rows.append((rate, name, total, label))

        if not rows:
            return ""

        rows.sort(reverse=True)
        lines = ["## Tool Reliability (from past sessions)\n",
                 "| Tool | Reliability | Calls |",
                 "|------|-------------|-------|"]
        for rate, name, total, label in rows:
            lines.append(f"| {name} | {label} ({rate*100:.0f}%) | {total} |")
        lines.append(
            "\nUse high-reliability tools when possible. "
            "Be extra careful with low-reliability tools — verify parameters before calling."
        )
        return "\n".join(lines)

    # ── Internal ─────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"tools": {}}

    def _flush(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        tmp = self.filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.filepath)


def _label(rate: float) -> str:
    if rate >= 0.9:
        return "✅ high"
    if rate >= 0.7:
        return "⚠️ medium"
    return "❌ low"
