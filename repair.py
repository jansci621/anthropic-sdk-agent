"""Repair Agent: a lightweight LLM specialized in diagnosing and fixing tool errors.

When the main agent's tool call fails, the Repair Agent can:
1. Analyze the error and classify the root cause
2. Suggest fixed parameters or an alternative approach
3. Return a structured repair plan for the main agent to act on
"""

import json

import anthropic

import config


REPAIR_SYSTEM_PROMPT = """You are a debugging specialist. Your job is to diagnose tool execution failures and suggest fixes.

Given:
- The tool name and its JSON schema
- The input that was provided
- The error message

You must respond with a JSON object:
{
  "root_cause": "<one-line explanation of what went wrong>",
  "fixable": true/false,
  "suggested_action": "retry_with_fix" | "use_alternative" | "ask_user" | "abort",
  "fixed_input": { ... },       // only if fixable and action is retry_with_fix
  "alternative_tool": "...",    // only if action is use_alternative
  "alternative_input": { ... }, // only if action is use_alternative
  "user_message": "..."         // only if action is ask_user
}

Rules:
- Be precise about what parameter was wrong and why
- If the path doesn't exist, suggest checking with list_files first
- If a command failed, suggest a corrected command
- If the error is unrecoverable (permission denied, etc.), set fixable=false
- Always explain your reasoning in root_cause
"""


class RepairAgent:
    """A lightweight agent that diagnoses tool errors and suggests fixes."""

    def __init__(self, client: anthropic.Anthropic, model: str | None = None):
        self.client = client
        self.model = model or config.MODEL

    def diagnose(
        self,
        tool_name: str,
        tool_input: dict,
        error_message: str,
        tool_schema: dict | None = None,
    ) -> dict:
        """Ask the repair agent to analyze a tool error and suggest a fix.

        Returns a dict with keys: root_cause, fixable, suggested_action, and
        optionally fixed_input / alternative_tool / user_message.
        """
        user_message = (
            f"Tool: {tool_name}\n"
            f"Input: {json.dumps(tool_input, ensure_ascii=False)}\n"
            f"Error: {error_message}"
        )
        if tool_schema:
            user_message += f"\nSchema: {json.dumps(tool_schema, ensure_ascii=False)}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=REPAIR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except Exception as e:
            return {
                "root_cause": f"Repair agent failed to respond: {e}",
                "fixable": False,
                "suggested_action": "ask_llm",
            }

    def format_repair_hint(self, diagnosis: dict) -> str:
        """Format a repair diagnosis into a hint string for the main agent."""
        parts = [f"[Repair Diagnosis] {diagnosis.get('root_cause', 'Unknown')}"]

        action = diagnosis.get("suggested_action", "")
        if action == "retry_with_fix" and diagnosis.get("fixed_input"):
            parts.append(f"Suggested fix: use input {json.dumps(diagnosis['fixed_input'], ensure_ascii=False)}")
        elif action == "use_alternative":
            alt_tool = diagnosis.get("alternative_tool", "")
            alt_input = diagnosis.get("alternative_input", {})
            parts.append(f"Suggested alternative: {alt_tool}({json.dumps(alt_input, ensure_ascii=False)})")
        elif action == "ask_user":
            parts.append(f"User input needed: {diagnosis.get('user_message', '')}")
        elif not diagnosis.get("fixable", True):
            parts.append("This error is NOT fixable. Try a completely different approach.")

        return "\n".join(parts)
