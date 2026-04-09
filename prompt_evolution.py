"""E2: Prompt Evolution — LLM reflects on each session and distils learnings.

At session end the agent calls `reflect()`. A lightweight LLM call analyses the
conversation and produces:
  - strategies that worked well
  - patterns to avoid
  - one-line rules to add to future system prompts

The output is appended to data/evolved_prompt.md and auto-loaded next session.
"""

import os
from datetime import datetime, timezone

import anthropic

import config

_EVOLVED_PROMPT_FILE = os.path.join(config.BASE_DIR, "data", "evolved_prompt.md")

_REFLECTION_SYSTEM = """You are a meta-learning assistant. Your job is to analyse an AI agent's
conversation and extract durable, reusable lessons.

Given the conversation history (tool calls, errors, outcomes), produce a brief markdown section:

## Session Learnings — {date}

### What worked
- (bullet list of effective strategies observed)

### What to avoid
- (bullet list of failure patterns to prevent)

### New rules
- (1–3 short imperative rules to add to the agent's system prompt)

Be concise. Max 200 words total. Skip sections that have nothing useful.
Only output the markdown — no preamble, no explanation."""


class PromptEvolution:
    """Manages session reflection and evolved prompt persistence."""

    def __init__(self, client: anthropic.Anthropic, model: str | None = None):
        self.client = client
        self.model = model or config.MODEL
        self._evolved_prompt_file = _EVOLVED_PROMPT_FILE

    # ── Public API ────────────────────────────────────────────────────────

    def reflect(self, conversation: list[dict], min_turns: int = 3) -> str | None:
        """Analyse the session and persist learnings. Returns the new snippet or None."""
        # Only reflect on substantive sessions
        if len(conversation) < min_turns * 2:
            return None

        summary = _summarise_conversation(conversation)
        if not summary:
            return None

        try:
            print(
                f"  {config.COLOR_SYSTEM}[PromptEvolution] Reflecting on session...{config.COLOR_RESET}",
                flush=True,
            )
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_REFLECTION_SYSTEM.format(date=_today()),
                messages=[{"role": "user", "content": summary}],
            )
            snippet = resp.content[0].text.strip()
            if snippet:
                self._append(snippet)
                print(
                    f"  {config.COLOR_SYSTEM}[PromptEvolution] Learnings saved to {_EVOLVED_PROMPT_FILE}{config.COLOR_RESET}",
                    flush=True,
                )
            return snippet
        except Exception as e:
            print(
                f"  {config.COLOR_SYSTEM}[PromptEvolution] Reflection failed: {e}{config.COLOR_RESET}",
                flush=True,
            )
            return None

    def load(self) -> str:
        """Return persisted evolved prompt content, or empty string."""
        if os.path.exists(self._evolved_prompt_file):
            try:
                with open(self._evolved_prompt_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except OSError:
                pass
        return ""

    # ── Internal ─────────────────────────────────────────────────────────

    _MAX_SESSIONS = 10  # max learning sessions to retain

    def _append(self, snippet: str):
        """Add a new reflection snippet, keeping only the latest _MAX_SESSIONS entries."""
        os.makedirs(os.path.dirname(self._evolved_prompt_file), exist_ok=True)

        # Load existing sections (split on the marker line)
        existing = self.load()
        _MARKER = "\n\n---session---\n\n"
        sections = [s.strip() for s in existing.split(_MARKER) if s.strip()]

        # Retain only the most recent (MAX-1) sessions, then add the new one
        sections = sections[-(self._MAX_SESSIONS - 1):]
        sections.append(snippet.strip())

        content = _MARKER.join(sections)
        tmp = self._evolved_prompt_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        os.replace(tmp, self._evolved_prompt_file)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _summarise_conversation(conversation: list[dict]) -> str:
    """Produce a statistical summary of the conversation for the reflection prompt.

    Deliberately excludes raw user/assistant text to prevent prompt injection
    via crafted user messages influencing the persisted evolved_prompt.md.
    """
    turns = sum(1 for m in conversation if m.get("role") == "user")
    tool_calls: dict[str, int] = {}
    errors: list[str] = []

    for msg in conversation:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = block.get("name", "unknown")
                tool_calls[name] = tool_calls.get(name, 0) + 1
            elif block.get("type") == "tool_result":
                result_text = str(block.get("content", ""))
                if "[Tool Error]" in result_text or '"error"' in result_text[:80].lower():
                    # Extract only the error type, not user-controlled content
                    errors.append(result_text[:80])

    lines = [
        f"Session stats:",
        f"  user_turns: {turns}",
        f"  tool_calls: {dict(list(tool_calls.items())[:20])}",
        f"  errors_count: {len(errors)}",
        f"  error_samples (first 5): {errors[:5]}",
    ]
    return "\n".join(lines)
