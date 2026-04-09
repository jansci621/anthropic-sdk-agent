"""Event bus: decouple agent output from rendering.

Every piece of output the agent produces becomes a typed event.
Sinks subscribe and decide how to render (CLI terminal, WebSocket, etc.).
"""

import asyncio
import json
import threading

import config

# ── Event Type Constants ─────────────────────────────────────────────────────

# Thinking block
EVENT_THINKING_START = "thinking_start"
EVENT_THINKING_DELTA = "thinking_delta"
EVENT_THINKING_END = "thinking_end"

# Text block
EVENT_TEXT_START = "text_start"
EVENT_TEXT_DELTA = "text_delta"
EVENT_TEXT_END = "text_end"

# Tool call block
EVENT_TOOL_CALL_START = "tool_call_start"
EVENT_TOOL_CALL_DELTA = "tool_call_delta"
EVENT_TOOL_CALL_END = "tool_call_end"

# Tool execution
EVENT_TOOL_EXECUTE = "tool_execute"
EVENT_TOOL_RESULT = "tool_result"

# Meta events
EVENT_SYSTEM = "system"
EVENT_ERROR = "error"
EVENT_STATUS = "status"

# ReAct-specific
EVENT_REACT_HEADER = "react_header"
EVENT_REACT_STEP_START = "react_step_start"
EVENT_REACT_FINAL = "react_final"


# ── EventBus ─────────────────────────────────────────────────────────────────

class EventBus:
    """Synchronous, thread-safe event multiplexer.

    Agent code calls emit() from any thread. Each registered sink
    receives every event.
    """

    def __init__(self):
        self._sinks: list = []
        self._lock = threading.Lock()

    def add_sink(self, sink):
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink):
        with self._lock:
            try:
                self._sinks.remove(sink)
            except ValueError:
                pass

    def emit(self, event_type: str, data: dict | None = None):
        """Fire an event to all sinks. Safe to call from any thread."""
        payload = data or {}
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(event_type, payload)
            except Exception:
                pass  # don't let a broken sink crash the agent


# ── Sink: CLI Terminal ───────────────────────────────────────────────────────

class CLIPrintSink:
    """Reproduces the original print() behavior for CLI mode."""

    def __call__(self, event_type: str, data: dict):
        if event_type == EVENT_THINKING_START:
            print(f"\n{config.COLOR_THINKING}[Thinking]{config.COLOR_RESET} ", end="", flush=True)
        elif event_type == EVENT_THINKING_DELTA:
            print(f"{config.COLOR_THINKING}{data['text']}{config.COLOR_RESET}", end="", flush=True)
        elif event_type == EVENT_THINKING_END:
            print()
        elif event_type == EVENT_TEXT_START:
            print(f"\n{config.COLOR_TOOL}[Assistant]{config.COLOR_RESET} ", end="", flush=True)
        elif event_type == EVENT_TEXT_DELTA:
            print(data["text"], end="", flush=True)
        elif event_type == EVENT_TEXT_END:
            print()
        elif event_type == EVENT_TOOL_CALL_START:
            print(f"\n{config.COLOR_TOOL}[Tool: {data['name']}]{config.COLOR_RESET}", end="", flush=True)
        elif event_type == EVENT_TOOL_CALL_DELTA:
            print(f"{config.COLOR_SYSTEM}{data['partial_json']}{config.COLOR_RESET}", end="", flush=True)
        elif event_type == EVENT_TOOL_CALL_END:
            pass  # newline handled by content_block_stop logic
        elif event_type == EVENT_TOOL_EXECUTE:
            label = data.get("label", "")
            if label:
                # ReAct mode: show as [Action N] name(input)
                print(
                    f"{config.COLOR_TOOL}[{label}]{config.COLOR_RESET} "
                    f"{data['name']}({data['input']})",
                    flush=True,
                )
            else:
                # Agent mode
                print(f"{config.COLOR_SYSTEM}  Executing: {data['name']}({data['input']}){config.COLOR_RESET}")
        elif event_type == EVENT_TOOL_RESULT:
            label = data.get("label", "")
            if label:
                print(
                    f"{config.COLOR_SYSTEM}[{label}]{config.COLOR_RESET} {data['result']}",
                    flush=True,
                )
        elif event_type == EVENT_SYSTEM:
            print(f"{config.COLOR_SYSTEM}{data['message']}{config.COLOR_RESET}")
        elif event_type == EVENT_ERROR:
            print(f"\n{config.COLOR_SYSTEM}[Error] {data['message']}{config.COLOR_RESET}")
        elif event_type == EVENT_STATUS:
            print(f"{config.COLOR_SYSTEM}{data['message']}{config.COLOR_RESET}", flush=True)
        elif event_type == EVENT_REACT_HEADER:
            print(f"\n{config.COLOR_SYSTEM}{data.get('separator', '')}\n{data.get('message', '')}\n{data.get('separator', '')}{config.COLOR_RESET}")
        elif event_type == EVENT_REACT_STEP_START:
            print(f"\n{config.COLOR_THINKING}[Thought {data['step_num']}]{config.COLOR_RESET} ", end="", flush=True)
        elif event_type == EVENT_REACT_FINAL:
            print(f"\n{config.COLOR_SYSTEM}{data.get('separator', '')}\n[ReAct] Final Answer\n{data.get('separator', '')}{config.COLOR_RESET}")


# ── Sink: Async Queue (for WebSocket) ────────────────────────────────────────

class AsyncQueueSink:
    """Pushes events as JSON strings into an asyncio.Queue.

    Used by web_server to forward agent events to a WebSocket.
    Thread-safe: asyncio.Queue.put_nowait() is safe from any thread.
    """

    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    def __call__(self, event_type: str, data: dict):
        message = json.dumps({"type": event_type, **data}, ensure_ascii=False)
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            pass  # drop event if queue is full rather than blocking
