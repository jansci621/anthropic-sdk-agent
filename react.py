"""ReAct pattern: explicit Thought → Action → Observation loop.

ReAct (Reasoning + Acting) makes every step of agent reasoning visible:

    Step N
    ├── Thought:      model reasons about what to do next
    ├── Action:       tool call (name + inputs)
    └── Observation:  tool result fed back into context

When the model calls no tools the loop terminates with a Final Answer.

Usage (standalone):
    from react import ReActLoop
    loop = ReActLoop(client, model, tools, dispatcher, thinking_config)
    answer, steps = loop.run(query="...", system=[...])

Integration with Agent:
    agent._run_react("explain the project structure")
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import anthropic

import config
from event_bus import (
    EVENT_THINKING_START, EVENT_THINKING_DELTA, EVENT_THINKING_END,
    EVENT_TEXT_START, EVENT_TEXT_DELTA, EVENT_TEXT_END,
    EVENT_TOOL_CALL_START, EVENT_TOOL_CALL_DELTA, EVENT_TOOL_CALL_END,
    EVENT_TOOL_EXECUTE, EVENT_TOOL_RESULT,
    EVENT_REACT_HEADER, EVENT_REACT_STEP_START, EVENT_REACT_FINAL,
)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    """A single Thought → Action → Observation step."""

    step_num: int
    thought: str = ""           # reasoning extracted from thinking blocks
    action: str | None = None   # primary tool name (first action)
    action_input: dict = field(default_factory=dict)
    observation: str | None = None  # last observation (for display)
    # All actions and observations when the model calls multiple tools in one turn
    actions: list[tuple[str, dict]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    is_final: bool = False
    final_answer: str = ""


@dataclass
class ReActTrace:
    """Full execution trace of a ReAct run."""

    query: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    final_content: list = field(default_factory=list)  # raw content blocks from last message
    total_turns: int = 0
    success: bool = False

    def summary(self) -> str:
        lines = [f"Query: {self.query}", f"Steps: {len(self.steps)}"]
        for s in self.steps:
            prefix = f"  [{s.step_num}]"
            if s.thought:
                t = s.thought[:120]
                lines.append(f"{prefix} Thought: {t}{'...' if len(s.thought) > 120 else ''}")
            for name, inp in s.actions:
                lines.append(f"{prefix} Action: {name}({json.dumps(inp, ensure_ascii=False)[:80]})")
            for obs in s.observations:
                lines.append(f"{prefix} Obs:    {obs[:120]}")
        if self.final_answer:
            lines.append(f"Answer: {self.final_answer[:200]}")
        return "\n".join(lines)


# ── ReAct Loop ───────────────────────────────────────────────────────────────

class ReActLoop:
    """Implements the Thought → Action → Observation loop using the Anthropic API.

    Architecture
    ────────────
    Each iteration of the loop does exactly one model call (streaming).
    The model response is decomposed into:

      • thinking blocks  → Thought
      • tool_use blocks  → Action (one or more; all executed before next turn)
      • text blocks      → accumulated as Final Answer (when stop_reason=end_turn)

    Tool results are returned as ``tool_result`` content in the next user turn,
    forming the Observation for that step.

    Multi-action steps
    ──────────────────
    The Anthropic API may return several tool_use blocks in one response.
    Each is treated as its own Action inside the current step, all executed
    before the next model call.
    """

    MAX_STEPS = 30
    _MAX_RETRIES = 5

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        tools: list[dict],
        tool_dispatcher: Callable[[str, dict], str],
        thinking_config: dict | None = None,
        max_tokens: int = 16000,
        event_bus=None,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.tool_dispatcher = tool_dispatcher
        self.thinking_config = thinking_config
        self.max_tokens = max_tokens
        self.event_bus = event_bus
        self._current_block_type: str | None = None

    def _emit(self, event_type: str, data: dict | None = None):
        """Emit an event through the event bus (if configured)."""
        if self.event_bus:
            self.event_bus.emit(event_type, data or {})

    # ── Public API ───────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        system: list[dict] | str,
    ) -> tuple[str, ReActTrace]:
        """Run the ReAct loop for *query* and return ``(final_answer, trace)``.

        Args:
            query:  The user's question or task.
            system: Anthropic system prompt (string or list of content blocks).

        Returns:
            A 2-tuple of the final answer string and the full :class:`ReActTrace`.
        """
        trace = ReActTrace(query=query)
        messages: list[dict] = [{"role": "user", "content": query}]

        self._print_header(query)

        for step_num in range(1, self.MAX_STEPS + 1):
            step = ReActStep(step_num=step_num)

            # ── Model call (streaming) ─────────────────────────────────
            try:
                message = self._stream_step(messages, system, step)
            except Exception as e:
                print(
                    f"\n{config.COLOR_SYSTEM}[ReAct] API error at step {step_num}: {e}{config.COLOR_RESET}"
                )
                trace.final_answer = f"[ReAct terminated: API error — {e}]"
                trace.total_turns = step_num
                return trace.final_answer, trace

            # ── Append assistant turn ──────────────────────────────────
            messages.append({"role": "assistant", "content": message.content})

            # ── Final Answer ──────────────────────────────────────────
            if message.stop_reason == "end_turn":
                for block in message.content:
                    if block.type == "text":
                        step.final_answer += block.text

                # Empty response: model returned end_turn with no text.
                # This can happen when a non-Claude model writes pseudo tool calls
                # inside thinking blocks instead of making actual tool_use blocks.
                # Nudge the model to respond properly and retry this step.
                if not step.final_answer.strip():
                    print(
                        f"\n{config.COLOR_SYSTEM}[ReAct] Step {step_num}: empty response, "
                        f"nudging model to retry...{config.COLOR_RESET}"
                    )
                    # Remove the empty assistant turn we just appended
                    messages.pop()
                    # Add a nudge and let the loop continue
                    messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM] Your previous response was empty. "
                            "You MUST respond with actual text content or use a tool. "
                            "Do NOT write tool calls inside thinking blocks — "
                            "use actual tool invocations instead. "
                            "Now please answer the original question."
                        ),
                    })
                    continue

                step.is_final = True
                trace.steps.append(step)
                trace.final_answer = step.final_answer
                trace.final_content = list(message.content)  # preserve all blocks including thinking
                trace.total_turns = step_num
                trace.success = True
                self._print_final_answer(step.final_answer)
                return step.final_answer, trace

            # ── Actions + Observations ────────────────────────────────
            if message.stop_reason == "tool_use":
                tool_blocks = [b for b in message.content if b.type == "tool_use"]

                # Record metadata first
                for i, block in enumerate(tool_blocks):
                    if i == 0:
                        step.action = block.name
                        step.action_input = block.input
                    step.actions.append((block.name, block.input))
                    self._print_action(step_num, block.name, block.input, i)

                # Execute tools in parallel
                obs_map: dict[int, str] = {}
                if len(tool_blocks) <= 1:
                    obs = self._execute_action(tool_blocks[0].name, tool_blocks[0].input)
                    obs_map[0] = obs
                else:
                    with ThreadPoolExecutor(max_workers=min(len(tool_blocks), 4)) as pool:
                        futures = {
                            pool.submit(self._execute_action, b.name, b.input): i
                            for i, b in enumerate(tool_blocks)
                        }
                        for future in as_completed(futures):
                            obs_map[futures[future]] = future.result()

                # Collect results in order
                tool_results: list[dict] = []
                for i in range(len(tool_blocks)):
                    obs = obs_map[i]
                    step.observations.append(obs)
                    step.observation = obs
                    self._print_observation(step_num, obs, i)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_blocks[i].id,
                        "content": obs,
                    })

                messages.append({"role": "user", "content": tool_results})
                trace.steps.append(step)
                continue

            # Unexpected stop reason — end gracefully
            trace.total_turns = step_num
            trace.final_answer = "[ReAct terminated: unexpected stop reason]"
            return trace.final_answer, trace

        # Max steps exhausted
        trace.total_turns = self.MAX_STEPS
        trace.final_answer = (
            f"[ReAct: reached {self.MAX_STEPS}-step limit without a final answer]"
        )
        print(
            f"\n{config.COLOR_SYSTEM}[ReAct] Max steps ({self.MAX_STEPS}) reached.{config.COLOR_RESET}"
        )
        return trace.final_answer, trace

    # ── Streaming ────────────────────────────────────────────────────────

    def _stream_step(
        self,
        messages: list[dict],
        system: list[dict] | str,
        step: ReActStep,
    ) -> anthropic.types.Message:
        """Stream one model call, updating *step.thought* in real time."""
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "tools": self.tools,
            "messages": messages,
        }
        if self.thinking_config:
            kwargs["thinking"] = self.thinking_config

        for attempt in range(self._MAX_RETRIES + 1):
            # Reset per attempt so a partial stream from a failed attempt
            # does not corrupt the thought trace on retry.
            thought_parts: list[str] = []
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        self._handle_stream_event(event, thought_parts, step)

                    message = stream.get_final_message()
                    step.thought = "\n".join(thought_parts).strip()
                    return message

            except anthropic.APIStatusError as exc:
                retryable = exc.status_code in (429, 503, 529)
                if not retryable or attempt == self._MAX_RETRIES:
                    raise
                wait = 2 ** attempt * 1.5
                print(
                    f"\n{config.COLOR_SYSTEM}[ReAct] API busy, retry {attempt + 1} in {wait:.0f}s...{config.COLOR_RESET}",
                    flush=True,
                )
                time.sleep(wait)

            except (anthropic.APIConnectionError, ConnectionError, OSError) as exc:
                if attempt == self._MAX_RETRIES:
                    raise
                wait = 2 ** attempt * 1.5
                print(
                    f"\n{config.COLOR_SYSTEM}[ReAct] Network error ({type(exc).__name__}), "
                    f"retry {attempt + 1} in {wait:.0f}s...{config.COLOR_RESET}",
                    flush=True,
                )
                time.sleep(wait)

    def _handle_stream_event(
        self,
        event,
        thought_parts: list[str],
        step: ReActStep,
    ):
        """Route stream events to their correct ReAct phase output."""
        if event.type == "content_block_start":
            block = event.content_block
            self._current_block_type = block.type
            if block.type == "thinking":
                self._emit(EVENT_REACT_STEP_START, {"step_num": step.step_num})
                self._emit(EVENT_THINKING_START)
            elif block.type == "text":
                self._emit(EVENT_TEXT_START)
            elif block.type == "tool_use":
                self._emit(EVENT_TOOL_CALL_START, {
                    "name": block.name,
                    "tool_use_id": block.id,
                })

        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "thinking_delta":
                thought_parts.append(delta.thinking)
                self._emit(EVENT_THINKING_DELTA, {"text": delta.thinking})
            elif delta.type == "text_delta":
                self._emit(EVENT_TEXT_DELTA, {"text": delta.text})
            elif delta.type == "input_json_delta":
                self._emit(EVENT_TOOL_CALL_DELTA, {"partial_json": delta.partial_json})

        elif event.type == "content_block_stop":
            if self._current_block_type == "thinking":
                self._emit(EVENT_THINKING_END)
            elif self._current_block_type == "text":
                self._emit(EVENT_TEXT_END)
            elif self._current_block_type == "tool_use":
                self._emit(EVENT_TOOL_CALL_END)
            self._current_block_type = None

        elif event.type == "content_block_stop":
            pass  # _end events emitted by CLIPrintSink or handled by caller

    # ── Tool Execution ────────────────────────────────────────────────────

    def _execute_action(self, name: str, tool_input: dict) -> str:
        """Execute a tool and return its result as a string."""
        try:
            return self.tool_dispatcher(name, tool_input)
        except Exception as exc:
            return json.dumps({
                "error": f"{type(exc).__name__}: {exc}",
                "tool": name,
                "input": tool_input,
            }, ensure_ascii=False)

    # ── Display ───────────────────────────────────────────────────────────

    _SEPARATOR = "─" * 60

    def _print_header(self, query: str):
        self._emit(EVENT_REACT_HEADER, {
            "query": query[:120],
            "separator": self._SEPARATOR,
            "message": f"[ReAct] Query: {query[:120]}",
        })

    def _print_action(self, step_num: int, name: str, tool_input: dict, idx: int):
        label = f"Action {step_num}" if idx == 0 else f"Action {step_num}.{idx + 1}"
        compact = json.dumps(tool_input, ensure_ascii=False)
        if len(compact) > 200:
            compact = compact[:200] + "…"
        self._emit(EVENT_TOOL_EXECUTE, {
            "label": label,
            "name": name,
            "input": compact,
        })

    def _print_observation(self, step_num: int, obs: str, idx: int):
        label = f"Obs {step_num}" if idx == 0 else f"Obs {step_num}.{idx + 1}"
        self._emit(EVENT_TOOL_RESULT, {
            "label": label,
            "result": obs if len(obs) <= 300 else obs[:300] + "…",
        })

    def _print_final_answer(self, answer: str):
        self._emit(EVENT_REACT_FINAL, {
            "separator": self._SEPARATOR,
        })
        # Answer text already streamed above; just close the section
