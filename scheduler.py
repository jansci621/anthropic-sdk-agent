"""Task scheduler: APScheduler-backed with JSON persistence.

Schedule types:
  cron      — standard cron expression  e.g. "0 8 * * 1-5"
  interval  — run every N seconds       e.g. interval_seconds=300
  once      — run once at a datetime    e.g. run_at="2024-06-01T09:00:00+08:00"

Action types:
  skill     — trigger a loaded skill    (agent._trigger_skill)
  query     — send a query to the agent (agent._agent_loop)
"""

import json
import logging
import os
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config

log = logging.getLogger(__name__)

TASKS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "scheduled_tasks.json"
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ScheduleConfig:
    type: str                  # "cron" | "interval" | "once"
    expr: str = ""             # cron expression
    interval_seconds: int = 0  # seconds between runs (interval type)
    run_at: str = ""           # ISO-8601 datetime (once type)


@dataclass
class ActionConfig:
    type: str        # "skill" | "query"
    skill: str = ""  # skill name  (skill type)
    args: str = ""   # skill args  (skill type)
    query: str = ""  # query text  (query type)


@dataclass
class ScheduledTask:
    id: str
    name: str
    schedule: ScheduleConfig
    action: ActionConfig
    enabled: bool = True
    session_id: str = ""   # 创建该任务的 session，执行结果回写到这个 session
    last_run: str = ""
    next_run: str = ""
    last_result: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledTask":
        d = dict(d)
        d["schedule"] = ScheduleConfig(**d["schedule"])
        d["action"] = ActionConfig(**d["action"])
        return cls(**d)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class TaskScheduler:
    """APScheduler wrapper with JSON persistence and agent integration."""

    def __init__(self):
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._tasks: dict[str, ScheduledTask] = {}
        self._exec_lock = threading.Lock()   # prevents concurrent task execution
        self._agent_getter: Callable | None = None
        self._session_updater: Callable | None = None  # (session_id, task_name, events) -> None
        self._load_tasks()

    def set_agent_getter(self, getter: Callable):
        """Provide a callable that returns (Agent, EventBus) — called by web_server / main."""
        self._agent_getter = getter

    def set_session_updater(self, fn: Callable):
        """Register callback: fn(session_id, task_name, events) → None.
        Called after each task execution to append results to the originating session."""
        self._session_updater = fn

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        self._scheduler.start()
        enabled = 0
        for task in self._tasks.values():
            if task.enabled:
                self._register(task)
                enabled += 1
        print(
            f"{config.COLOR_SYSTEM}[Scheduler] Started — "
            f"{enabled}/{len(self._tasks)} task(s) active.{config.COLOR_RESET}"
        )

    def stop(self):
        self._scheduler.shutdown(wait=False)

    # ── Public API ────────────────────────────────────────────────────────

    def list_tasks(self) -> list[ScheduledTask]:
        self._refresh_next_run()
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> ScheduledTask | None:
        return self._tasks.get(task_id)

    def add_task(self, task: ScheduledTask) -> ScheduledTask:
        if not task.id:
            task.id = str(uuid.uuid4())[:8]
        self._tasks[task.id] = task
        if task.enabled:
            self._register(task)
        self._refresh_next_run()
        self._save_tasks()
        return task

    def update_task(self, task_id: str, patch: dict) -> ScheduledTask | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for k, v in patch.items():
            if k == "schedule" and isinstance(v, dict):
                for sk, sv in v.items():
                    setattr(task.schedule, sk, sv)
            elif k == "action" and isinstance(v, dict):
                for ak, av in v.items():
                    setattr(task.action, ak, av)
            elif hasattr(task, k) and k not in ("id", "created_at"):
                setattr(task, k, v)
        self._unregister(task_id)
        if task.enabled:
            self._register(task)
        self._refresh_next_run()
        self._save_tasks()
        return task

    def remove_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        self._unregister(task_id)
        task_name = task.name
        del self._tasks[task_id]
        self._save_tasks()
        # 清除历史执行记录，并追加一条 deleted 事件通知前端清理 DOM
        with _events_lock:
            for ev in list(_scheduler_events):
                if ev.get("task") == task_name:
                    _scheduler_events.remove(ev)
            _scheduler_events.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": "deleted",
                "task": task_name,
                "msg": "Task deleted",
                "events": [],
            })
        return True

    def pause_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.enabled = False
        self._unregister(task_id)
        self._save_tasks()
        return True

    def resume_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.enabled = True
        self._register(task)
        self._refresh_next_run()
        self._save_tasks()
        return True

    def run_now(self, task_id: str) -> bool:
        """Trigger a task immediately in a background thread."""
        if task_id not in self._tasks:
            return False
        t = threading.Thread(target=self._execute, args=(task_id,), daemon=True)
        t.start()
        return True

    # ── Internal ─────────────────────────────────────────────────────────

    def _job_id(self, task_id: str) -> str:
        return f"task_{task_id}"

    def _register(self, task: ScheduledTask):
        try:
            trigger = self._make_trigger(task.schedule)
        except Exception as e:
            log.error("[Scheduler] Invalid schedule for '%s': %s", task.name, e)
            print(f"{config.COLOR_SYSTEM}[Scheduler] Warning: invalid schedule for '{task.name}': {e}{config.COLOR_RESET}")
            return
        self._scheduler.add_job(
            self._execute,
            trigger=trigger,
            id=self._job_id(task.id),
            args=[task.id],
            replace_existing=True,
            max_instances=1,    # 只允许一个实例同时运行
            coalesce=True,      # 执行期间积压的多次触发合并为一次，完成后补跑一次
            misfire_grace_time=None,  # 不因超时丢弃，等上一次完成后继续
        )

    def _unregister(self, task_id: str):
        job_id = self._job_id(task_id)
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

    def _make_trigger(self, s: ScheduleConfig):
        if s.type == "cron":
            return CronTrigger.from_crontab(s.expr, timezone="UTC")
        if s.type == "interval":
            if s.interval_seconds <= 0:
                raise ValueError("interval_seconds must be > 0")
            from datetime import timedelta
            # 延迟一个周期后首次运行，避免任务创建时与用户 agent_loop 并发
            start_date = datetime.now(timezone.utc) + timedelta(seconds=s.interval_seconds)
            return IntervalTrigger(seconds=s.interval_seconds, start_date=start_date)
        if s.type == "once":
            return DateTrigger(run_date=datetime.fromisoformat(s.run_at))
        raise ValueError(f"Unknown schedule type: {s.type!r}")

    def _execute(self, task_id: str):
        """Execute a task — runs in APScheduler background thread."""
        task = self._tasks.get(task_id)
        if not task or not task.enabled:
            return
        if self._agent_getter is None:
            log.warning("[Scheduler] No agent getter configured, skipping '%s'", task.name)
            return

        task.last_run = datetime.now(timezone.utc).isoformat()
        print(f"\n{config.COLOR_SYSTEM}[Scheduler] ▶ Running: {task.name}{config.COLOR_RESET}")

        agent, bus = self._agent_getter()
        # 获取 agent 执行锁，与用户会话串行，防止并发修改 agent.conversation
        with self._exec_lock, agent._execution_lock:
            capture = _CaptureSink()
            bus.add_sink(capture)           # 临时挂载，仅捕获本次任务事件
            try:
                append_scheduler_event("info", task.name, "started")

                # 保存当前 conversation，执行完后无条件恢复
                # 防止调度任务携带/污染共享 agent 的对话历史
                saved_conv = list(agent.conversation)
                try:
                    action = task.action
                    if action.type == "skill":
                        # skill 也需要隔离：_trigger_skill 会 append 到 conversation
                        # 清空后让它在空白上下文中独立执行
                        agent.conversation = []
                        agent._trigger_skill(action.skill, action.args)
                        task.last_result = f"OK — skill '{action.skill}' executed"

                    elif action.type == "query":
                        # 独立单轮对话，不混入用户 session
                        agent.conversation = [{"role": "user", "content": action.query}]
                        agent._agent_loop()
                        task.last_result = "OK — query executed"

                    else:
                        task.last_result = f"Error: unknown action type '{action.type}'"
                finally:
                    agent.conversation = saved_conv  # 无论成功/失败都还原

                captured = capture.get_events()
                append_scheduler_event("ok", task.name, task.last_result, captured)
                # 把结果写回原 session
                if task.session_id and self._session_updater:
                    try:
                        self._session_updater(task.session_id, task.name, captured)
                    except Exception as cb_err:
                        log.warning("[Scheduler] session_updater failed: %s", cb_err)
                print(
                    f"\n{config.COLOR_SYSTEM}[Scheduler] ✓ Done: {task.name}{config.COLOR_RESET}\n"
                )

            except Exception as e:
                task.last_result = f"Error: {e}"
                log.exception("[Scheduler] Task '%s' failed", task.name)
                captured = capture.get_events()
                append_scheduler_event("error", task.name, str(e), captured)
                if task.session_id and self._session_updater:
                    try:
                        self._session_updater(task.session_id, task.name, captured)
                    except Exception:
                        pass
                print(
                    f"\n{config.COLOR_SYSTEM}[Scheduler] ✗ Failed: {task.name} — {e}{config.COLOR_RESET}\n"
                )
            finally:
                bus.remove_sink(capture)    # 执行完立即卸载，不影响后续 session
                self._save_tasks()

    def _refresh_next_run(self):
        for task in self._tasks.values():
            job = self._scheduler.get_job(self._job_id(task.id))
            if job and job.next_run_time:
                task.next_run = job.next_run_time.isoformat()
            elif not task.enabled:
                task.next_run = ""

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_tasks(self):
        self._refresh_next_run()
        os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
        tmp = TASKS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self._tasks.values()], f,
                      ensure_ascii=False, indent=2)
        os.replace(tmp, TASKS_FILE)

    def _load_tasks(self):
        if not os.path.exists(TASKS_FILE):
            return
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                task = ScheduledTask.from_dict(d)
                self._tasks[task.id] = task
            log.info("[Scheduler] Loaded %d task(s).", len(self._tasks))
        except Exception as e:
            log.error("[Scheduler] Failed to load tasks: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

task_scheduler = TaskScheduler()

# ── Scheduler event log (独立于 session，供 Web UI 轮询）────────────────────
# 保留最近 200 条事件，线程安全
_scheduler_events: deque[dict] = deque(maxlen=200)
_events_lock = threading.Lock()


class _CaptureSink:
    """临时挂载到 bus，捕获调度任务执行期间的事件流。
    只保留 Web 渲染所需的事件，跳过大量 thinking_delta 避免 payload 过大。
    无 active 属性 — 永远接收事件。
    """

    # 只保留这些类型，thinking_delta/tool_call_delta 等跳过
    _KEEP = frozenset({
        "text_delta", "tool_execute", "system", "status", "error",
    })

    def __init__(self):
        self._events: list[dict] = []

    def __call__(self, event_type: str, data: dict):
        if event_type in self._KEEP:
            self._events.append({"type": event_type, **data})

    def get_events(self) -> list[dict]:
        return list(self._events)


def dedicated_session_id(task_name: str) -> str:
    """Stable dedicated session id for a scheduled task (sched_ + md5[:12])."""
    import hashlib as _hashlib
    return "sched_" + _hashlib.md5(task_name.encode()).hexdigest()[:12]


def append_scheduler_event(level: str, task_name: str, message: str,
                           events: list[dict] | None = None):
    with _events_lock:
        _scheduler_events.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "task": task_name,
            "msg": message,
            "events": events or [],
            "session_id": dedicated_session_id(task_name),  # 前端用于定向刷新
        })


def get_scheduler_events(since: str | None = None) -> list[dict]:
    """Return events newer than `since` (ISO-8601). Returns all if since=None."""
    with _events_lock:
        events = list(_scheduler_events)
    if since:
        events = [e for e in events if e["ts"] > since]
    return events


# ── Agent Tool Definitions ────────────────────────────────────────────────────

SCHEDULER_TOOLS = [
    {
        "name": "create_scheduled_task",
        "description": (
            "Create a scheduled or recurring task. Use this when the user asks to "
            "run something periodically (every N minutes/hours), at a specific time, "
            "or set up automated recurring actions like weather checks, reminders, "
            "data collection, or reports.\n\n"
            "schedule_type options:\n"
            "  cron     — standard cron expression, e.g. '* * * * *' (every minute), "
            "'0 8 * * *' (daily 8am UTC)\n"
            "  interval — run every N seconds, e.g. interval_seconds=60 for every minute\n"
            "  once     — run once at a specific datetime (ISO-8601)\n\n"
            "action_type options:\n"
            "  skill — trigger a named skill (provide skill_name and skill_args)\n"
            "  query — send a natural language query through the agent"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable task name",
                },
                "schedule_type": {
                    "type": "string",
                    "enum": ["cron", "interval", "once"],
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression (required when schedule_type=cron)",
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "Seconds between runs (required when schedule_type=interval)",
                },
                "run_at": {
                    "type": "string",
                    "description": "ISO-8601 datetime (required when schedule_type=once)",
                },
                "action_type": {
                    "type": "string",
                    "enum": ["skill", "query"],
                },
                "skill_name": {
                    "type": "string",
                    "description": "Skill to trigger (required when action_type=skill)",
                },
                "skill_args": {
                    "type": "string",
                    "description": "Arguments passed to the skill",
                },
                "query": {
                    "type": "string",
                    "description": "Query text (required when action_type=query)",
                },
            },
            "required": ["name", "schedule_type", "action_type"],
        },
    },
    {
        "name": "list_scheduled_tasks",
        "description": "List all scheduled tasks with their status, next run time, and last result.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_scheduled_task",
        "description": "Pause or permanently delete a scheduled task by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to cancel"},
                "permanent": {
                    "type": "boolean",
                    "description": "If true, delete permanently; if false (default), just pause",
                },
            },
            "required": ["task_id"],
        },
    },
]


def handle_scheduler_tool(name: str, tool_input: dict, session_id: str = "") -> str:
    """Dispatch scheduler tool calls from the agent."""
    import json as _json

    if name == "create_scheduled_task":
        stype = tool_input.get("schedule_type", "")
        schedule = ScheduleConfig(
            type=stype,
            expr=tool_input.get("cron_expr", ""),
            interval_seconds=int(tool_input.get("interval_seconds", 0)),
            run_at=tool_input.get("run_at", ""),
        )
        atype = tool_input.get("action_type", "query")
        action = ActionConfig(
            type=atype,
            skill=tool_input.get("skill_name", ""),
            args=tool_input.get("skill_args", ""),
            query=tool_input.get("query", ""),
        )
        try:
            task = task_scheduler.add_task(
                ScheduledTask(id="", name=tool_input.get("name", "Task"),
                              schedule=schedule, action=action,
                              session_id=session_id)
            )
            return _json.dumps({
                "status": "created",
                "task_id": task.id,
                "name": task.name,
                "next_run": task.next_run,
            }, ensure_ascii=False)
        except Exception as e:
            return _json.dumps({"error": str(e)})

    if name == "list_scheduled_tasks":
        tasks = task_scheduler.list_tasks()
        return _json.dumps([{
            "id": t.id,
            "name": t.name,
            "enabled": t.enabled,
            "schedule": t.schedule.type,
            "next_run": t.next_run,
            "last_run": t.last_run,
            "last_result": t.last_result,
        } for t in tasks], ensure_ascii=False)

    if name == "cancel_scheduled_task":
        task_id = tool_input.get("task_id", "")
        permanent = tool_input.get("permanent", False)
        if permanent:
            ok = task_scheduler.remove_task(task_id)
            action = "deleted"
        else:
            ok = task_scheduler.pause_task(task_id)
            action = "paused"
        if ok:
            return _json.dumps({"status": action, "task_id": task_id})
        return _json.dumps({"error": f"Task {task_id!r} not found"})

    return _json.dumps({"error": f"Unknown scheduler tool: {name}"})
