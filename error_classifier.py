"""Error classifier for self-healing tool execution.

Classifies tool errors into categories and determines recovery strategy:
- network    → auto-retry without consuming an LLM turn
- bad_input  → feed back to LLM to fix parameters
- resource   → feed back to LLM to create or find the resource
- permission → feed back to LLM to try an alternative
- unknown    → feed back to LLM for general diagnosis
"""

import re


class ErrorClassification:
    """Result of classifying a tool error."""

    __slots__ = ("category", "retryable", "strategy", "detail")

    def __init__(self, category: str, retryable: bool, strategy: str, detail: str = ""):
        self.category = category
        self.retryable = retryable
        self.strategy = strategy
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "retryable": self.retryable,
            "strategy": self.strategy,
            "detail": self.detail,
        }


# ── Keyword patterns ──────────────────────────────────────────────────────────

_NETWORK_KEYWORDS = re.compile(
    r"timeout|timed.out|connection.{0,10}(refused|reset|error)|network.{0,10}unreachable"
    r"|dns.{0,10}(fail|error)|socket.{0,10}(error|closed)|name.{0,10}not.{0,10}resolved"
    r"|503|502|429|overloaded",
    re.IGNORECASE,
)

_RESOURCE_KEYWORDS = re.compile(
    r"not\s+found|no\s+such\s+file|does\s+not\s+exist|no\s+.*\s+matching"
    r"|enoent|missing|not\s+exist",
    re.IGNORECASE,
)

_INPUT_KEYWORDS = re.compile(
    r"invalid|bad\s+request|syntax\s+error|parse\s+error|regex|value\s*error"
    r"|type\s*error|key\s*error|index\s*error|argument",
    re.IGNORECASE,
)

_PERMISSION_KEYWORDS = re.compile(
    r"permission|denied|forbidden|unauthorized|access\s+denied|not\s+allowed"
    r"|eacces|ep erm",
    re.IGNORECASE,
)


def classify_error(error: Exception, tool_name: str = "", tool_input: dict | None = None) -> ErrorClassification:
    """Classify a tool execution error into a recovery strategy."""
    msg = str(error)

    if _NETWORK_KEYWORDS.search(msg):
        return ErrorClassification(
            category="network",
            retryable=True,
            strategy="auto_retry",
            detail="Network/transport error — will auto-retry once",
        )

    if _PERMISSION_KEYWORDS.search(msg):
        return ErrorClassification(
            category="permission",
            retryable=False,
            strategy="alternative",
            detail="Permission denied — try a different approach or inform the user",
        )

    if _RESOURCE_KEYWORDS.search(msg):
        return ErrorClassification(
            category="resource_missing",
            retryable=True,
            strategy="fix_params",
            detail="Resource not found — check path or create resource first",
        )

    if _INPUT_KEYWORDS.search(msg):
        return ErrorClassification(
            category="bad_input",
            retryable=True,
            strategy="fix_params",
            detail="Invalid input — fix parameters and retry",
        )

    return ErrorClassification(
        category="unknown",
        retryable=True,
        strategy="ask_llm",
        detail="Unclassified error — LLM should diagnose and fix",
    )
