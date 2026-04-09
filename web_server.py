"""FastAPI web server: WebSocket endpoint + static file serving for the agent UI."""

import asyncio
import json
import os
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent import Agent

import config
from event_bus import EventBus, AsyncQueueSink, CLIPrintSink, EVENT_SYSTEM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="AI Agent Web UI")

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")


# ── Session Store ──────────────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}  # session_id -> {"agent": Agent, "bus": EventBus}


def _get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        bus = EventBus()
        bus.add_sink(CLIPrintSink())  # server console output
        agent = Agent(event_bus=bus)
        _sessions[session_id] = {"agent": agent, "bus": bus}
    return _sessions[session_id]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(BASE_DIR, "web", "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/info")
async def agent_info():
    import config
    return JSONResponse({
        "model": config.MODEL,
        "provider": config.PROVIDER,
        "thinking": config.THINKING_ENABLED,
        "auto_route": config.AUTO_ROUTE,
    })


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = _get_or_create_session(session_id)
    agent: Agent = session["agent"]
    bus: EventBus = session["bus"]

    # Create async queue for this WebSocket connection
    queue: asyncio.Queue = asyncio.Queue()
    sink = AsyncQueueSink(queue)
    bus.add_sink(sink)

    try:
        # Task A: forward events from queue to WebSocket
        async def send_events():
            while True:
                message = await queue.get()
                try:
                    await websocket.send_text(message)
                except Exception:
                    break

        sender_task = asyncio.create_task(send_events())

        # Task B: receive user messages and run the agent
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Handle reset
            if data.get("type") == "reset":
                agent.conversation.clear()
                agent._consecutive_errors = 0
                agent._error_budget.clear()
                bus.emit(EVENT_SYSTEM, {"message": "Conversation reset."})
                continue

            user_message = data.get("message", "").strip()
            mode = data.get("mode", "auto")

            if not user_message:
                continue

            # Route the message
            loop = asyncio.get_running_loop()

            # Natural language skill matching (before routing)
            skill_match = agent._match_skill_natural(user_message)
            if skill_match:
                skill_name, args = skill_match
                bus.emit(EVENT_SYSTEM, {"message": f"[Skill → {skill_name}]"})
                await loop.run_in_executor(None, agent._trigger_skill, skill_name, args)
                bus.emit(EVENT_SYSTEM, {"message": ""})
                continue

            if mode == "react":
                bus.emit(EVENT_SYSTEM, {"message": f"[Mode: ReAct]"})
                await loop.run_in_executor(None, agent._run_react, user_message)
                bus.emit(EVENT_SYSTEM, {"message": ""})
            elif mode == "agent":
                agent.conversation.append({"role": "user", "content": user_message})
                await loop.run_in_executor(None, agent._agent_loop)
                bus.emit(EVENT_SYSTEM, {"message": ""})
            else:
                # Auto mode — check router
                if agent.router and config.AUTO_ROUTE:
                    from router import RouteDecision
                    decision = agent.router.route(user_message, len(agent.conversation))
                    if decision.mode == "react":
                        bus.emit(EVENT_SYSTEM, {
                            "message": f"[Router -> ReAct] {decision.reason}"
                        })
                        await loop.run_in_executor(None, agent._run_react, user_message)
                        bus.emit(EVENT_SYSTEM, {"message": ""})
                        continue

                agent.conversation.append({"role": "user", "content": user_message})
                await loop.run_in_executor(None, agent._agent_loop)
                bus.emit(EVENT_SYSTEM, {"message": ""})

    except WebSocketDisconnect:
        pass
    finally:
        bus.remove_sink(sink)
        sender_task.cancel()
