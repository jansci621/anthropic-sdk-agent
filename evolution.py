"""E6: Multi-Agent Co-Evolution — three specialised agents that each learn
independently and share a common experience pool.

┌─────────────────────────────────────────────────────────┐
│  Shared Experience Pool  (experience.py ExperienceStore) │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
       ┌───────┴──┐   ┌───────┴──┐   ┌──────┴───┐
       │ Executor │   │  Repair  │   │  Judge   │
       │  Agent   │   │  Agent   │   │  Agent   │
       └──────────┘   └──────────┘   └──────────┘

• ExecutorAgent  — learns which tool sequences achieve goals efficiently
• RepairAgent    — learns which fix strategies work per error type
• JudgeAgent     — learns to classify errors and pick the right recovery path

Each agent:
  - reads from the shared pool before acting
  - writes outcomes back to the pool after acting
  - maintains its own specialised model (can be overridden)
"""

import json

import anthropic

import config
from experience import ExperienceStore

# ── Base ─────────────────────────────────────────────────────────────────────

class _EvolvingAgent:
    """Base class for co-evolving agents."""

    role: str = "generic"

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str | None = None,
        pool: ExperienceStore | None = None,
    ):
        self.client = client
        self.model = model or config.MODEL
        # Pool injected by caller; no global singleton
        self.pool: ExperienceStore = pool or ExperienceStore()

    def _relevant_experiences(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant past experiences and format as context."""
        experiences = self.pool.retrieve(query, top_k=top_k)
        return self.pool.format_for_prompt(experiences) if experiences else ""

    def _record(self, task_summary: str, error_pattern: str, fix_strategy: str,
                tool_name: str, success: bool):
        """Write outcome back to the shared pool."""
        self.pool.record(
            task_summary=f"[{self.role}] {task_summary}",
            error_pattern=error_pattern,
            fix_strategy=fix_strategy,
            tool_name=tool_name,
            success=success,
        )


# ── Executor Agent ────────────────────────────────────────────────────────────

_EXECUTOR_SYSTEM = """You are an Executor Agent — specialised in planning efficient tool sequences.

{experience_context}

Given a task description, output a JSON execution plan:
{{
  "steps": [
    {{"tool": "tool_name", "rationale": "why this step", "expected_output": "..."}}
  ],
  "fallback": "what to do if the plan fails"
}}

Prefer tools with high reliability. Minimise total steps.
Output only valid JSON."""


class ExecutorAgent(_EvolvingAgent):
    """Plans tool sequences, learns from past execution outcomes."""

    role = "executor"

    def __init__(self, client, model=None, pool=None):
        super().__init__(client, model, pool)

    def plan(self, task: str) -> dict | None:
        """Generate an execution plan for a task."""
        context = self._relevant_experiences(task)
        system = _EXECUTOR_SYSTEM.format(experience_context=context or "")
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": f"Task: {task}"}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            print(
                f"  {config.COLOR_SYSTEM}[ExecutorAgent] plan failed: {e}{config.COLOR_RESET}",
                flush=True,
            )
            return None

    def record_outcome(self, task: str, tool_name: str, success: bool, error: str = ""):
        self._record(
            task_summary=task,
            error_pattern=error or "(success)",
            fix_strategy="executor_plan",
            tool_name=tool_name,
            success=success,
        )


# ── Repair Agent (evolving) ───────────────────────────────────────────────────

_EVOLVING_REPAIR_SYSTEM = """You are a Repair Agent — specialised in diagnosing and fixing tool failures.

{experience_context}

Given a tool error, respond with JSON:
{{
  "root_cause": "...",
  "fix_strategy": "...",
  "confidence": 0.0-1.0
}}

Learn from past repairs: prefer strategies with proven success rates.
Output only valid JSON."""


class EvolvingRepairAgent(_EvolvingAgent):
    """Repair agent that learns which fix strategies work for each error type."""

    role = "repair"

    def __init__(self, client, model=None, pool=None):
        super().__init__(client, model, pool)

    def diagnose(self, tool_name: str, error_msg: str, tool_input: dict) -> dict | None:
        """Diagnose a tool error using accumulated repair experience."""
        context = self._relevant_experiences(f"{tool_name} {error_msg}")
        system = _EVOLVING_REPAIR_SYSTEM.format(experience_context=context or "")
        prompt = (
            f"Tool: {tool_name}\n"
            f"Error: {error_msg}\n"
            f"Input: {json.dumps(tool_input, ensure_ascii=False)[:300]}"
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            print(
                f"  {config.COLOR_SYSTEM}[EvolvingRepairAgent] diagnose failed: {e}{config.COLOR_RESET}",
                flush=True,
            )
            return None

    def record_repair(self, tool_name: str, error_msg: str, fix_strategy: str, success: bool):
        self._record(
            task_summary=f"repair {tool_name}",
            error_pattern=error_msg,
            fix_strategy=fix_strategy,
            tool_name=tool_name,
            success=success,
        )


# ── Judge Agent ───────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are a Judge Agent — specialised in error classification and recovery routing.

{experience_context}

Given an error, classify it and choose the recovery path. Respond with JSON:
{{
  "category": "network|bad_input|resource_missing|permission|logic|unknown",
  "recovery_path": "auto_retry|fix_params|use_alternative|ask_user|abort",
  "confidence": 0.0-1.0,
  "rationale": "..."
}}

Use past experience to improve classification accuracy.
Output only valid JSON."""


class JudgeAgent(_EvolvingAgent):
    """Error classification agent that improves over time."""

    role = "judge"

    def __init__(self, client, model=None, pool=None):
        super().__init__(client, model, pool)

    def classify(self, tool_name: str, error_msg: str) -> dict | None:
        """Classify an error and recommend a recovery path."""
        context = self._relevant_experiences(f"error {error_msg}")
        system = _JUDGE_SYSTEM.format(experience_context=context or "")
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": f"Tool: {tool_name}\nError: {error_msg}"}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            print(
                f"  {config.COLOR_SYSTEM}[JudgeAgent] classify failed: {e}{config.COLOR_RESET}",
                flush=True,
            )
            return None

    def record_classification(self, tool_name: str, error_msg: str,
                               predicted_category: str, actual_outcome: str, correct: bool):
        self._record(
            task_summary=f"classify {tool_name} error",
            error_pattern=error_msg,
            fix_strategy=f"predicted={predicted_category} actual={actual_outcome}",
            tool_name=tool_name,
            success=correct,
        )


# ── Co-Evolution Coordinator ─────────────────────────────────────────────────

class CoEvolutionCoordinator:
    """Orchestrates the three evolving agents and routes errors through them."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str | None = None,
        pool: ExperienceStore | None = None,
    ):
        # All three agents share the same pool instance (injected, not global)
        shared = pool or ExperienceStore()
        self.executor = ExecutorAgent(client, model, pool=shared)
        self.repair = EvolvingRepairAgent(client, model, pool=shared)
        self.judge = JudgeAgent(client, model, pool=shared)

    def handle_error(
        self,
        tool_name: str,
        error_msg: str,
        tool_input: dict,
    ) -> str:
        """Full co-evolution pipeline: Judge → Repair → record outcomes."""
        # Step 1: Judge classifies the error
        classification = self.judge.classify(tool_name, error_msg)
        category = (classification or {}).get("category", "unknown")
        recovery = (classification or {}).get("recovery_path", "ask_llm")

        # Step 2: Repair agent diagnoses
        diagnosis = self.repair.diagnose(tool_name, error_msg, tool_input)
        fix_strategy = (diagnosis or {}).get("fix_strategy", "")
        confidence = (diagnosis or {}).get("confidence", 0.0)

        # Build hint for main agent
        parts = [f"[CoEvolution] category={category} recovery={recovery}"]
        if fix_strategy:
            # Guard: LLM may return confidence as string instead of float
            try:
                conf_str = f"{float(confidence):.0%}"
            except (TypeError, ValueError):
                conf_str = str(confidence)
            parts.append(f"Fix strategy (confidence={conf_str}): {fix_strategy}")

        return "\n".join(parts)

    def record_outcome(
        self,
        tool_name: str,
        error_msg: str,
        fix_strategy: str,
        success: bool,
        predicted_category: str = "unknown",
    ):
        """Record outcomes back to each agent's learning pool."""
        self.repair.record_repair(tool_name, error_msg, fix_strategy, success)
        self.judge.record_classification(
            tool_name, error_msg, predicted_category, "success" if success else "failure", success
        )
