"""E5: Fast/Slow Path Evolution — two-tier error recovery inspired by eBPF rule learning.

Fast Path  — Pattern → Rule → Direct fix (no LLM, <1ms)
Slow Path  — ErrorClassifier → RepairAgent → Record → Distil into Fast Rule

Rules are stored in data/fast_rules.json.
A rule graduates from Slow → Fast after `PROMOTE_THRESHOLD` successful fixes.

Rule schema:
  pattern      — normalised error substring to match (regex)
  fix_template — hint string injected into the error report
  tool_name    — optional: only apply for this tool ("*" = any)
  hit_count    — how many times this rule has matched
  success_rate — rolling average of post-fix success
  promoted     — bool: True = Fast Path active
"""

import json
import os
import re
import threading
from datetime import datetime, timezone

import config

_RULES_FILE = os.path.join(config.BASE_DIR, "data", "fast_rules.json")

# Slow Path hits before a rule is promoted to Fast Path
PROMOTE_THRESHOLD = 3
# Max rules to keep
_MAX_RULES = 200


class FastRuleEngine:
    """Manages Fast/Slow Path rules for common error patterns."""

    def __init__(self, filepath: str = _RULES_FILE):
        self.filepath = filepath
        self._data = self._load()
        self._lock = threading.Lock()

    # ── Fast Path ─────────────────────────────────────────────────────────

    def fast_match(self, error_msg: str, tool_name: str) -> str | None:
        """Return an instant fix hint if a promoted rule matches. No LLM needed.

        Both the stored pattern and the incoming error are normalised consistently
        so that path names, numbers, and casing don't prevent matches.
        """
        with self._lock:
            norm_error = _normalise(error_msg)   # match on normalised form, same as stored
            for rule in self._data["rules"]:
                if not rule.get("promoted"):
                    continue
                if rule.get("tool_name") not in ("*", tool_name):
                    continue
                try:
                    if re.search(rule["pattern"], norm_error, re.IGNORECASE):
                        rule["hit_count"] = rule.get("hit_count", 0) + 1
                        rule["last_matched"] = _now()
                        self._flush()
                        return f"[FastRule] {rule['fix_template']}"
                except re.error:
                    continue
            return None

    # ── Slow Path ─────────────────────────────────────────────────────────

    def record_repair(
        self,
        error_msg: str,
        tool_name: str,
        fix_strategy: str,
        success: bool,
    ):
        """Record a Slow Path repair outcome and potentially promote to Fast Path (thread-safe)."""
        with self._lock:
            norm = _normalise(error_msg)

            # Find or create a rule for this pattern
            for rule in self._data["rules"]:
                if rule["norm_pattern"] == norm and rule.get("tool_name") in ("*", tool_name):
                    n = rule.get("hit_count", 0)
                    rule["success_rate"] = (rule.get("success_rate", 0) * n + int(success)) / (n + 1)
                    rule["hit_count"] = n + 1
                    rule["last_matched"] = _now()
                    if success and not rule.get("promoted") and rule["hit_count"] >= PROMOTE_THRESHOLD:
                        rule["promoted"] = True
                        print(
                            f"  {config.COLOR_SYSTEM}[FastRule] Promoted rule for: "
                            f"{norm[:60]}...{config.COLOR_RESET}",
                            flush=True,
                        )
                    self._flush()
                    return

            # New rule — start in Slow Path
            rule = {
                "pattern": re.escape(norm[:100]),
                "norm_pattern": norm,
                "fix_template": fix_strategy[:300],
                "tool_name": tool_name or "*",
                "hit_count": 1,
                "success_rate": 1.0 if success else 0.0,
                "promoted": False,
                "last_matched": _now(),
            }
            self._data["rules"].append(rule)

            # Evict oldest unpromoted rules if over limit
            if len(self._data["rules"]) > _MAX_RULES:
                unpromoted = [r for r in self._data["rules"] if not r.get("promoted")]
                unpromoted.sort(key=lambda r: r["last_matched"])
                to_remove = set(id(r) for r in unpromoted[:len(self._data["rules"]) - _MAX_RULES])
                self._data["rules"] = [r for r in self._data["rules"] if id(r) not in to_remove]

            self._flush()

    def stats_summary(self) -> str:
        """Short stats string for display."""
        total = len(self._data["rules"])
        promoted = sum(1 for r in self._data["rules"] if r.get("promoted"))
        return f"FastRules: {promoted} active / {total} total"

    # ── Internal ─────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"rules": []}

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
    text = text.lower()
    text = re.sub(r"[\\/][^\s]+", " PATH ", text)
    text = re.sub(r"\d+", "N", text)
    return re.sub(r"\s+", " ", text).strip()[:150]
