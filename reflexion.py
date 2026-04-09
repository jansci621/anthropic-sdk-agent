"""E4: Reflexion — post-task self-evaluation and improvement suggestions.

After a complex task (≥ N tool calls or ≥ 1 repair), the Reflexion module:
1. Counts turns used, repairs attempted, consecutive errors
2. Asks the LLM to evaluate strategy effectiveness
3. Generates concrete improvement suggestions
4. Stores them into the ExperienceStore for future retrieval

Design based on Shinn et al. "Reflexion" (2023) adapted for tool-use agents.
"""

import json

import anthropic

import config

_REFLEXION_SYSTEM = """You are a self-evaluation assistant for an AI agent.

Given a task summary and execution statistics, write a short reflexion:

1. **What went well** — effective strategies, smooth tool use
2. **What went wrong** — errors, wasted turns, bad approaches
3. **Improvement rule** — ONE actionable rule for next time (imperative, ≤ 20 words)

Output JSON:
{
  "went_well": ["..."],
  "went_wrong": ["..."],
  "improvement_rule": "..."
}
Only output valid JSON, no other text."""

# Trigger reflexion when a task exceeds these thresholds
_MIN_TURNS_FOR_REFLEXION = 4
_MIN_REPAIRS_FOR_REFLEXION = 1


class ReflexionAgent:
    """Post-task self-evaluation agent."""

    def __init__(self, client: anthropic.Anthropic, model: str | None = None):
        self.client = client
        self.model = model or config.MODEL

    def should_reflect(self, turns: int, repairs: int, consecutive_errors: int) -> bool:
        """Return True if this task warrants a reflexion pass."""
        return (
            turns >= _MIN_TURNS_FOR_REFLEXION
            or repairs >= _MIN_REPAIRS_FOR_REFLEXION
            or consecutive_errors >= 2
        )

    def reflect(
        self,
        task_summary: str,
        turns: int,
        repairs: int,
        consecutive_errors: int,
        error_log: list[dict],
    ) -> dict | None:
        """Run reflexion and return the structured result dict."""
        stats = {
            "turns_used": turns,
            "repairs_attempted": repairs,
            "max_consecutive_errors": consecutive_errors,
            "error_types": list({e.get("error_type", "?") for e in error_log}),
        }

        prompt = (
            f"Task: {task_summary[:300]}\n\n"
            f"Execution stats:\n{json.dumps(stats, indent=2)}\n\n"
            f"Error log (last 5):\n"
            + "\n".join(
                f"  [{e.get('tool')}] {e.get('error', '')[:150]}"
                for e in error_log[-5:]
            )
        )

        try:
            print(
                f"  {config.COLOR_SYSTEM}[Reflexion] Evaluating task...{config.COLOR_RESET}",
                flush=True,
            )
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_REFLEXION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            self._print_reflexion(result)
            return result
        except Exception as e:
            print(
                f"  {config.COLOR_SYSTEM}[Reflexion] Failed: {e}{config.COLOR_RESET}",
                flush=True,
            )
            return None

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _print_reflexion(result: dict):
        well = result.get("went_well", [])
        wrong = result.get("went_wrong", [])
        rule = result.get("improvement_rule", "")
        print(f"  {config.COLOR_SYSTEM}[Reflexion] went_well: {well}{config.COLOR_RESET}")
        print(f"  {config.COLOR_SYSTEM}[Reflexion] went_wrong: {wrong}{config.COLOR_RESET}")
        if rule:
            print(f"  {config.COLOR_SYSTEM}[Reflexion] new rule: {rule}{config.COLOR_RESET}")
