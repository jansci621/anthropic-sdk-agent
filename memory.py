"""File-based persistent memory system."""

import json
import os
import threading
import uuid
from datetime import datetime, timezone

import config


# ── Tool Definitions (Anthropic API format) ─────────────────────────────────

MEMORY_TOOLS = [
    {
        "name": "save_memory",
        "description": (
            "Save a piece of information to persistent memory. "
            "Use this to store user preferences, facts, project details, "
            "decisions, or any important context the user shares. "
            "Saved memories persist across conversations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "A short label or title for this memory (e.g. 'user_favorite_language')",
                },
                "value": {
                    "type": "string",
                    "description": "The information to store",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "fact", "project", "decision", "person", "summary", "instruction", "other"],
                    "description": "Category for organizing this memory",
                },
            },
            "required": ["key", "value", "category"],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Search stored memories by keywords or category. "
            "Use this to recall previously saved information, "
            "such as user preferences, past decisions, or project context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for in memory keys and values",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "fact", "project", "decision", "person", "summary", "instruction", "other"],
                    "description": "Filter results to a specific category",
                },
            },
            "required": [],
        },
    },
]


# ── Memory Store ────────────────────────────────────────────────────────────

class MemoryStore:
    """JSON-file-backed persistent memory."""

    def __init__(self, filepath: str = config.MEMORY_FILE):
        self.filepath = filepath
        self._data = self._load()
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────

    def save(self, key: str, value: str, category: str) -> dict:
        """Save a memory entry and return the stored record (thread-safe)."""
        with self._lock:
            entry = {
                "id": uuid.uuid4().hex[:12],
                "key": key,
                "value": value,
                "category": category,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._data["memories"].append(entry)
            self._flush()
            return entry

    def search(self, query: str = "", category: str = "") -> list[dict]:
        """Search memories by keyword and/or category."""
        results = self._data["memories"]

        if category:
            results = [m for m in results if m.get("category") == category]

        if query:
            q = query.lower()
            results = [
                m for m in results
                if q in m.get("key", "").lower() or q in m.get("value", "").lower()
            ]

        # Cap at 50 to avoid context overflow
        return results[:50]

    # ── Internal ─────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load memories from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"memories": []}

    def _flush(self):
        """Write memories to disk atomically."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        tmp = self.filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.filepath)


# ── Tool Dispatch ───────────────────────────────────────────────────────────

def handle_memory_tool(name: str, tool_input: dict, store: MemoryStore) -> str:
    """Dispatch a memory tool call and return the JSON result string."""
    if name == "save_memory":
        result = store.save(
            key=tool_input["key"],
            value=tool_input["value"],
            category=tool_input["category"],
        )
        return json.dumps({"status": "saved", "memory": result}, ensure_ascii=False)

    if name == "search_memory":
        results = store.search(
            query=tool_input.get("query", ""),
            category=tool_input.get("category", ""),
        )
        return json.dumps({"count": len(results), "memories": results}, ensure_ascii=False)

    return json.dumps({"error": f"Unknown memory tool: {name}"})
