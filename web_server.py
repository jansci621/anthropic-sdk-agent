"""FastAPI web server: WebSocket endpoint + static file serving for the agent UI."""

import asyncio
import json
import os
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from agent import Agent
from scheduler import (
    task_scheduler, ScheduledTask, ScheduleConfig, ActionConfig,
    get_scheduler_events,
)

import config
from event_bus import EventBus, AsyncQueueSink, CLIPrintSink, EVENT_SYSTEM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")
WORKSPACES_DIR = os.path.join(BASE_DIR, "data", "workspaces")


def _session_updater(session_id: str, task_name: str, events: list[dict]):
    """Append scheduler task result to a DEDICATED session for this task.

    Each scheduled task gets its own session (id = "sched_<task_hash>") so
    results never contaminate the session where the task was created.
    """
    from scheduler import dedicated_session_id as _ded_id
    text = "".join(ev.get("text", "") for ev in events if ev.get("type") == "text_delta")
    if not text.strip():
        return

    # Stable dedicated session id based on task name (same formula as scheduler.py)
    dedicated_id = _ded_id(task_name)

    # Ensure the session exists in memory
    if dedicated_id not in _sessions:
        agent, bus = _get_shared_agent()
        saved = _load_session(dedicated_id)
        snapshot = saved["messages"] if saved and saved.get("messages") else []
        _sessions[dedicated_id] = {"agent": agent, "bus": bus, "_conv_snapshot": list(snapshot)}

    session = _sessions[dedicated_id]
    synthetic = {
        "role": "assistant",
        "content": [{"type": "text", "text": text.strip()}],
    }
    snapshot = list(session.get("_conv_snapshot", []))

    # Add a user-role "header" message if this is the first result
    if not snapshot:
        snapshot.append({"role": "user", "content": f"⏰ {task_name}"})

    snapshot.append(synthetic)
    session["_conv_snapshot"] = snapshot
    _save_session(dedicated_id, snapshot)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    task_scheduler.set_agent_getter(_get_shared_agent)
    task_scheduler.set_session_updater(_session_updater)
    task_scheduler.start()
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    task_scheduler.stop()


app = FastAPI(title="AI Agent Web UI", lifespan=lifespan)

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")


# ── Session Persistence ───────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}  # session_id -> {"agent": Agent, "bus": EventBus}

# Shared agent — one instance for all sessions (avoids reloading RAG model)
_shared_agent: Agent | None = None
_shared_bus: EventBus | None = None
_agent_lock = threading.Lock()
_agent_run_lock = asyncio.Lock()  # prevents concurrent executor runs on shared agent


def _get_shared_agent() -> tuple[Agent, EventBus]:
    """Get or create the shared Agent and EventBus (singleton)."""
    global _shared_agent, _shared_bus
    with _agent_lock:
        if _shared_agent is None:
            _shared_bus = EventBus()
            _shared_bus.add_sink(CLIPrintSink())
            _shared_agent = Agent(event_bus=_shared_bus)
        return _shared_agent, _shared_bus


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _serialize_block(block) -> dict:
    """Convert a single Anthropic content block to a JSON-safe dict."""
    if isinstance(block, dict):
        return block
    if not hasattr(block, "type"):
        return {"type": "text", "text": str(block)}
    b: dict = {"type": block.type}
    if block.type == "text":
        b["text"] = getattr(block, "text", "")
    elif block.type == "thinking":
        b["thinking"] = getattr(block, "thinking", "")
    elif block.type == "tool_use":
        b["id"] = getattr(block, "id", "")
        b["name"] = getattr(block, "name", "")
        b["input"] = getattr(block, "input", {})
    elif block.type == "tool_result":
        b["tool_use_id"] = getattr(block, "tool_use_id", "")
        raw_content = getattr(block, "content", "")
        # tool_result.content can be a string or a list of content blocks
        if isinstance(raw_content, list):
            b["content"] = [_serialize_block(c) for c in raw_content]
        else:
            b["content"] = str(raw_content)
    else:
        # Fallback: extract all non-private attributes
        for k, v in vars(block).items():
            if not k.startswith("_"):
                try:
                    json.dumps(v)  # test serializability
                    b[k] = v
                except (TypeError, ValueError):
                    b[k] = str(v)
    return b


def _serialize_conversation(conversation: list) -> list:
    """Convert conversation to JSON-serializable format."""
    result = []
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            result.append({"role": role, "content": [_serialize_block(b) for b in content]})
        elif isinstance(content, dict):
            result.append({"role": role, "content": content})
        else:
            result.append({"role": role, "content": str(content)})
    return result


def _save_session(session_id: str, conversation: list):
    """Save a session's conversation to disk. Creates file immediately on first message."""
    if not conversation:
        return

    os.makedirs(SESSIONS_DIR, exist_ok=True)

    messages = _serialize_conversation(conversation)

    # Auto-title from first user message
    title = "New Chat"
    for msg in conversation:
        text = ""
        c = msg.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    text = b.get("text", "")
                    break
        if msg.get("role") == "user" and text:
            if text.startswith("[SYSTEM"):
                continue
            # For skill-triggered messages, extract only the user-visible request
            req_marker = "\nUser request: "
            idx = text.find(req_marker)
            if idx != -1:
                text = text[idx + len(req_marker):]
            elif text.startswith("[Using skill:"):
                continue  # skill message with no user request — skip
            title = text.strip()[:40].replace("\n", " ")
            if len(text.strip()) > 40:
                title += "..."
            break

    # Read existing to preserve created timestamp
    path = _session_path(session_id)
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    data = {
        "id": session_id,
        "title": title,
        "created": existing.get("created", datetime.now(timezone.utc).isoformat()),
        "updated": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_session(session_id: str) -> dict | None:
    """Load a session's data from disk."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _clean_title_from_messages(messages: list) -> str:
    """Re-derive a clean title from stored messages, skipping injected system messages."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "")
        if not isinstance(text, str):
            continue
        if text.startswith("[SYSTEM"):
            continue
        req_marker = "\nUser request: "
        idx = text.find(req_marker)
        if idx != -1:
            text = text[idx + len(req_marker):]
        elif text.startswith("[Using skill:"):
            continue  # skill message with no user request — skip
        text = text.strip()
        if text:
            title = text[:40].replace("\n", " ")
            return title + ("..." if len(text) > 40 else "")
    return "New Chat"


def _list_sessions() -> list[dict]:
    """List all sessions sorted by most recent first."""
    if not os.path.exists(SESSIONS_DIR):
        return []
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("title", "Untitled")
            # Retroactively clean stale titles from system-injected messages
            if title.startswith("[SYSTEM") or title.startswith("[Using skill:"):
                title = _clean_title_from_messages(data.get("messages", []))
            sessions.append({
                "id": data.get("id", fname[:-5]),
                "title": title,
                "created": data.get("created", ""),
                "updated": data.get("updated", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    sessions.sort(key=lambda s: s.get("updated", ""), reverse=True)
    return sessions


def _get_or_create_session(session_id: str) -> dict:
    """Get or create a session entry. Conversation is stored per-session in _conv_snapshot."""
    agent, bus = _get_shared_agent()
    if session_id not in _sessions:
        # Save any previous session's snapshot before creating new entry
        for prev_id, prev_session in _sessions.items():
            if prev_session["agent"] is agent:
                _save_session(prev_id, list(prev_session.get("_conv_snapshot", [])))
                break

        # Load conversation snapshot from disk if this session has history
        saved = _load_session(session_id)
        if saved and saved.get("messages"):
            snapshot = saved["messages"]
        else:
            snapshot = []

        _sessions[session_id] = {"agent": agent, "bus": bus, "_conv_snapshot": list(snapshot)}
    return _sessions[session_id]


# ── Workspace Management ─────────────────────────────────────────────────────

def _get_workspace(session_id: str) -> str:
    """Get or create the workspace directory for a session."""
    ws = os.path.join(WORKSPACES_DIR, session_id)
    os.makedirs(ws, exist_ok=True)
    return ws


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(BASE_DIR, "web", "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        response = HTMLResponse(f.read())
        # Prevent browser caching so frontend updates take effect immediately
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response


@app.get("/api/info")
async def agent_info():
    return JSONResponse({
        "model": config.MODEL,
        "provider": config.PROVIDER,
        "thinking": config.THINKING_ENABLED,
        "auto_route": config.AUTO_ROUTE,
    })


@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse(_list_sessions())


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    data = _load_session(session_id)
    if not data:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse(data)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    path = _session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
    # Clean up workspace
    ws = os.path.join(WORKSPACES_DIR, session_id)
    if os.path.isdir(ws):
        shutil.rmtree(ws, ignore_errors=True)
    _sessions.pop(session_id, None)
    return JSONResponse({"status": "ok"})


# ── File Browser API (scoped to session workspace) ─────────────────────────────

def _workspace_safe_path(session_id: str, rel_path: str) -> str | None:
    """Resolve a relative path within a session's workspace directory."""
    ws = _get_workspace(session_id)
    resolved = os.path.normpath(os.path.join(ws, rel_path))
    if not resolved.startswith(ws):
        return None
    return resolved


@app.get("/api/files/list")
async def list_files(session_id: str, path: str = ""):
    """List files and directories under the given path in session workspace."""
    real = _workspace_safe_path(session_id, path)
    if real is None or not os.path.isdir(real):
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    ws_root = _get_workspace(session_id)
    entries = []
    try:
        for name in sorted(os.listdir(real)):
            full = os.path.join(real, name)
            rel = os.path.relpath(full, ws_root).replace("\\", "/")
            if name.startswith(".") or name == "__pycache__":
                continue
            is_dir = os.path.isdir(full)
            size = 0 if is_dir else os.path.getsize(full)
            entries.append({
                "name": name,
                "path": rel,
                "is_dir": is_dir,
                "size": size,
            })
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    return JSONResponse({"path": path, "entries": entries})


@app.get("/api/files/view")
async def view_file(session_id: str, path: str):
    """View a file's content as text from session workspace."""
    real = _workspace_safe_path(session_id, path)
    if real is None or not os.path.isfile(real):
        return JSONResponse({"error": "File not found"}, status_code=404)
    # Limit to files < 2MB
    if os.path.getsize(real) > 2 * 1024 * 1024:
        return JSONResponse({"error": "File too large (>2MB)"}, status_code=400)
    try:
        with open(real, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return PlainTextResponse(content)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/files/download")
async def download_file(session_id: str, path: str):
    """Download a file from session workspace."""
    real = _workspace_safe_path(session_id, path)
    if real is None or not os.path.isfile(real):
        return JSONResponse({"error": "File not found"}, status_code=404)
    filename = os.path.basename(real)
    return FileResponse(real, filename=filename)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        await websocket.accept()
    except Exception:
        return

    # Switch to new session (also saves any previous session)
    session = _get_or_create_session(session_id)
    agent: Agent = session["agent"]
    bus: EventBus = session["bus"]

    # Set per-session workspace for file tool isolation
    agent._workspace_dir = _get_workspace(session_id)

    queue: asyncio.Queue = asyncio.Queue()
    sink = AsyncQueueSink(queue)
    # Remove stale AsyncQueueSinks from previous WS connections
    # (prevents ghost events from leaking across sessions)
    with bus._lock:
        bus._sinks = [s for s in bus._sinks if not isinstance(s, AsyncQueueSink)]
    bus.add_sink(sink)

    try:
        async def send_events():
            while True:
                message = await queue.get()
                try:
                    await websocket.send_text(message)
                except Exception:
                    break

        sender_task = asyncio.create_task(send_events())

        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Handle reset
            if data.get("type") == "reset":
                async with _agent_run_lock:
                    sink.active = True
                    try:
                        agent.conversation.clear()
                        agent._consecutive_errors = 0
                        agent._error_budget.clear()
                        bus.emit(EVENT_SYSTEM, {"message": "Conversation reset."})
                        _save_session(session_id, agent.conversation)
                    finally:
                        sink.active = False
                continue

            user_message = data.get("message", "").strip()
            mode = data.get("mode", "auto")

            if not user_message:
                continue

            # Pre-save immediately — before acquiring any lock — so the session
            # appears in the sidebar right away (frontend polls 300ms after send)
            try:
                _save_session(
                    session_id,
                    list(session.get("_conv_snapshot", [])) + [{"role": "user", "content": user_message}],
                )
            except Exception:
                pass

            loop = asyncio.get_running_loop()

            # ── Run agent under lock (prevents concurrent executor on shared agent) ──
            async with _agent_run_lock:
                # Activate only THIS session's sink — prevents event leaking
                sink.active = True

                try:
                    # Restore this session's conversation into the shared agent
                    agent.conversation = list(session.get("_conv_snapshot", []))
                    agent._workspace_dir = _get_workspace(session_id)
                    agent._current_session_id = session_id  # for scheduler tool to bind task to session

                    def _locked(fn, *a):
                        """Run agent method under execution lock to prevent scheduler races."""
                        with agent._execution_lock:
                            fn(*a)

                    # Natural language skill matching
                    skill_match = agent._match_skill_natural(user_message)
                    if skill_match:
                        skill_name, args = skill_match
                        bus.emit(EVENT_SYSTEM, {"message": f"[Skill → {skill_name}]"})
                        # Pre-save with user message so the session appears in sidebar immediately
                        try:
                            _save_session(session_id,
                                          list(agent.conversation) + [{"role": "user", "content": user_message}])
                        except Exception:
                            pass
                        await loop.run_in_executor(None, _locked, agent._trigger_skill, skill_name, args)
                        bus.emit(EVENT_SYSTEM, {"message": ""})
                        try:
                            conv_copy = list(agent.conversation)
                            _save_session(session_id, conv_copy)
                            session["_conv_snapshot"] = conv_copy
                        except Exception:
                            pass
                        continue

                    # Append user message to conversation (no save yet)
                    agent.conversation.append({"role": "user", "content": user_message})

                    use_react = mode == "react"
                    if mode == "react":
                        agent.conversation.pop()
                        bus.emit(EVENT_SYSTEM, {"message": f"[Mode: ReAct]"})
                    elif mode == "agent":
                        pass
                    else:
                        if agent.router and config.AUTO_ROUTE:
                            from router import RouteDecision
                            decision = agent.router.route(user_message, len(agent.conversation))
                            if decision.mode == "react":
                                use_react = True
                                agent.conversation.pop()
                                bus.emit(EVENT_SYSTEM, {
                                    "message": f"[Router -> ReAct] {decision.reason}"
                                })

                    # Agent is about to start — create session file now
                    try:
                        conv_copy = list(agent.conversation)
                        _save_session(session_id, conv_copy)
                        session["_conv_snapshot"] = conv_copy
                    except Exception:
                        pass

                    # Run agent (only one at a time — execution_lock prevents scheduler races)
                    if use_react:
                        await loop.run_in_executor(None, _locked, agent._run_react, user_message)
                    else:
                        await loop.run_in_executor(None, _locked, agent._agent_loop)
                    bus.emit(EVENT_SYSTEM, {"message": ""})

                    # Save after agent turn
                    try:
                        conv_copy = list(agent.conversation)
                        _save_session(session_id, conv_copy)
                        session["_conv_snapshot"] = conv_copy
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                finally:
                    sink.active = False

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        bus.remove_sink(sink)
        sender_task.cancel()
        # Final save on disconnect — use snapshot to avoid saving
        # another session's conversation if agent was already switched
        try:
            snapshot = session.get("_conv_snapshot", agent.conversation)
            _save_session(session_id, list(snapshot))
        except Exception:
            pass


# ── Schedule API ──────────────────────────────────────────────────────────────

@app.get("/api/schedule/events")
async def schedule_events(since: str | None = None):
    """
    Return scheduler execution events (independent of sessions).
    Optionally filter by ?since=<ISO-8601> to get only newer events.
    Poll this endpoint every few seconds to show scheduler activity in the UI.
    """
    return JSONResponse(get_scheduler_events(since))


@app.get("/api/schedule/tasks")
async def schedule_list():
    return JSONResponse([t.to_dict() for t in task_scheduler.list_tasks()])


@app.post("/api/schedule/tasks")
async def schedule_add(body: dict):
    """
    Body example:
    {
      "name": "Daily weather",
      "schedule": {"type": "cron", "expr": "0 8 * * *"},
      "action":   {"type": "skill", "skill": "weather", "args": "Beijing"}
    }
    """
    try:
        task = ScheduledTask(
            id="",
            name=body.get("name", "Unnamed task"),
            schedule=ScheduleConfig(**body["schedule"]),
            action=ActionConfig(**body["action"]),
            enabled=body.get("enabled", True),
        )
    except (KeyError, TypeError) as e:
        return JSONResponse({"error": f"Invalid body: {e}"}, status_code=400)
    task = task_scheduler.add_task(task)
    return JSONResponse(task.to_dict(), status_code=201)


@app.get("/api/schedule/tasks/{task_id}")
async def schedule_get(task_id: str):
    task = task_scheduler.get_task(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(task.to_dict())


@app.patch("/api/schedule/tasks/{task_id}")
async def schedule_update(task_id: str, body: dict):
    task = task_scheduler.update_task(task_id, body)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(task.to_dict())


@app.delete("/api/schedule/tasks/{task_id}")
async def schedule_delete(task_id: str):
    if not task_scheduler.remove_task(task_id):
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"status": "deleted"})


@app.post("/api/schedule/tasks/{task_id}/run")
async def schedule_run_now(task_id: str):
    if not task_scheduler.run_now(task_id):
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"status": "triggered"})


@app.post("/api/schedule/tasks/{task_id}/pause")
async def schedule_pause(task_id: str):
    if not task_scheduler.pause_task(task_id):
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"status": "paused"})


@app.post("/api/schedule/tasks/{task_id}/resume")
async def schedule_resume(task_id: str):
    if not task_scheduler.resume_task(task_id):
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse({"status": "resumed"})
