"""Automatic query router: selects between ReAct and general agent mode.

Decision hierarchy
──────────────────
1. Hard-exclude rules  — short/conversational messages → always "agent"
2. Fast ReAct rules    — regex patterns that strongly suggest multi-hop reasoning
3. Fast Agent rules    — regex patterns that strongly suggest direct action
4. LLM classifier      — single cheap call (Haiku) for genuinely ambiguous queries
5. Default fallback    — "agent"

The router is stateless except for the optional Anthropic client used for
step 4. All callers receive a RouteDecision with a mode, a reason string
(useful for debug logging), and a confidence score (0–1).

Disable auto-routing entirely by setting AI_AUTO_ROUTE=false.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import anthropic

import config


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    """Routing outcome for a single query."""

    mode: Literal["react", "agent"]
    reason: str          # human-readable explanation for logging/debug
    confidence: float    # 0.0 = random guess, 1.0 = certain


# ── Rule Tables ──────────────────────────────────────────────────────────────

# Signals → ReAct (multi-hop reasoning + sequential tool use)
_REACT_SIGNALS: list[tuple[str, str]] = [
    # Explicit reasoning requests
    (r"\bstep.by.step\b",                   "step-by-step request"),
    (r"\bwalk me through\b",                "walkthrough request"),
    (r"\bhow does .{3,60} work\b",          "how-does-X-work question"),
    (r"\bwhy (did|does|is|are|isn'?t|aren'?t)\b", "why-question"),
    (r"\bexplain (the|how|why|what)\b",     "explain request"),

    # Investigation / diagnosis
    (r"\b(trace|debug|diagnose|troubleshoot)\b",   "debugging request"),
    (r"\b(investigate|research|analyze|examine)\b", "investigation request"),
    (r"\bfind out\b|\bfigure out\b|\bdetermine\b",  "discovery request"),

    # Comparison / multi-target
    (r"\bcompare .{2,40}\b(and|vs\.?|with)\b",     "comparison request"),
    (r"\bdifference between\b",                     "difference question"),

    # Sequential reasoning cues
    (r"\b(first|initially).{0,60}(then|next|after)\b", "sequential reasoning"),
    (r"\bgive me a breakdown\b|\bbreakdown of\b",   "breakdown request"),

    # Codebase understanding
    (r"\bhow (is|are) .{2,40} (implemented|structured|organised|organized)\b",
     "architecture question"),
    (r"\bwhat does .{2,40} do\b",                   "functionality question"),
]

# Signals → General mode (direct action, memory, or continuation)
_AGENT_SIGNALS: list[tuple[str, str]] = [
    # Continuations / short acknowledgements
    (r"^(ok|okay|sure|yes|no|thanks|thank you|done|continue|go ahead|please|do it|got it|proceed)\.?$",
     "short acknowledgement"),

    # Memory operations
    (r"\b(remember|save|store|recall|forget|memorize)\b", "memory operation"),

    # Direct code / file actions
    (r"\b(write|create|generate|build|implement|scaffold)\b", "code-generation request"),
    (r"\b(fix|repair|refactor|rewrite|update|edit|modify|patch|change)\b", "edit request"),
    (r"\b(delete|remove|rename|move|copy)\b",        "file operation"),
    (r"\b(run|execute|install|deploy|start|stop|restart)\b", "execution request"),

    # Single-word or very direct commands
    (r"^/?\w{1,15}$",                                "single-word command"),

    # Conversational follow-ups with pronoun references
    (r"^(it|that|this|those|these|them)\b",          "pronoun-reference continuation"),
]


# ── Router ───────────────────────────────────────────────────────────────────

class QueryRouter:
    """Routes a user query to either 'react' or 'agent' mode.

    Args:
        client: Optional Anthropic client for LLM-based fallback classification.
                If ``None``, ambiguous queries fall back to "agent".
    """

    # Queries shorter than this are almost always conversational
    _SHORT_THRESHOLD = 25

    # Use LLM classifier only if neither rule set fires and query is long enough
    _LLM_MIN_LENGTH = 40

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client
        # Pre-compile all patterns for speed
        self._react_rules = [
            (re.compile(pat, re.IGNORECASE), reason)
            for pat, reason in _REACT_SIGNALS
        ]
        self._agent_rules = [
            (re.compile(pat, re.IGNORECASE), reason)
            for pat, reason in _AGENT_SIGNALS
        ]

    def route(self, query: str, conversation_len: int = 0) -> RouteDecision:
        """Return a :class:`RouteDecision` for *query*.

        Args:
            query:            Raw user input (not a slash command).
            conversation_len: Number of messages in the current conversation.
                              Used to detect follow-up messages that should
                              stay in general mode.
        """
        stripped = query.strip()

        # ── 1. Hard exclude: too short or a follow-up ──────────────────────
        if len(stripped) < self._SHORT_THRESHOLD:
            return RouteDecision("agent", "short message", 0.95)

        if conversation_len > 0 and len(stripped.split()) <= 6:
            return RouteDecision("agent", "short follow-up", 0.90)

        # ── 2. Fast ReAct rules ────────────────────────────────────────────
        for pattern, reason in self._react_rules:
            if pattern.search(stripped):
                return RouteDecision("react", f"react rule: {reason}", 0.85)

        # ── 3. Fast Agent rules ────────────────────────────────────────────
        for pattern, reason in self._agent_rules:
            if pattern.search(stripped):
                return RouteDecision("agent", f"agent rule: {reason}", 0.85)

        # ── 4. LLM classifier for genuinely ambiguous queries ──────────────
        if self._client and len(stripped) >= self._LLM_MIN_LENGTH:
            return self._llm_classify(stripped)

        # ── 5. Default ─────────────────────────────────────────────────────
        return RouteDecision("agent", "default fallback", 0.55)

    # ── LLM Classifier ───────────────────────────────────────────────────────

    _CLASSIFIER_SYSTEM = (
        "You are a query classifier. Decide whether the user's message needs "
        "multi-step reasoning where each step DEPENDS on the result of the previous "
        "step (tool lookup → reason → next tool). "
        "If yes, reply with exactly: react\n"
        "If no (direct action, code generation, conversational), reply: agent\n"
        "Reply with ONLY one of those two words."
    )

    def _llm_classify(self, query: str) -> RouteDecision:
        """Single cheap Haiku call to classify an ambiguous query.

        Results are cached so identical queries never pay twice.
        """
        mode = self._cached_classify(query)
        return RouteDecision(mode, "llm_classifier", 0.75)

    @lru_cache(maxsize=256)
    def _cached_classify(self, query: str) -> Literal["react", "agent"]:
        """LRU-cached classification — same query string always reuses the result."""
        try:
            resp = self._client.messages.create(
                model=config.ROUTER_MODEL,
                max_tokens=5,
                system=self._CLASSIFIER_SYSTEM,
                messages=[{"role": "user", "content": query}],
            )
            text = next(
                (b.text.strip().lower() for b in resp.content if b.type == "text"),
                "agent",
            )
            return "react" if text.startswith("react") else "agent"
        except Exception:
            # Never let the classifier block the main flow
            return "agent"
