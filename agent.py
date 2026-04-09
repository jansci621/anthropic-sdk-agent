"""Core agent: manual agentic loop with streaming, thinking output, and tool dispatch."""

import json
import os
import sys
import time

import anthropic

import config
from error_classifier import classify_error
from memory import MemoryStore, MEMORY_TOOLS, handle_memory_tool
from repair import RepairAgent
from self_repair import ScriptRepairer
from experience import ExperienceStore
from prompt_evolution import PromptEvolution
from tool_stats import ToolStats
from reflexion import ReflexionAgent
from fast_rules import FastRuleEngine
from evolution import CoEvolutionCoordinator
from rag import RAGSystem, RAG_TOOL, handle_rag_tool
from web_tools import WEB_TOOLS, handle_web_tool
from shell_tools import SHELL_TOOLS, handle_shell_tool
from file_tools import FILE_TOOLS, handle_file_tool
from skills import discover_skills
from react import ReActLoop
from router import QueryRouter


class Agent:
    """AI Agent with memory, RAG, and streaming thinking output."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = config.SYSTEM_PROMPT,
    ):
        self.model = model or config.MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS

        # Resolve API key
        resolved_key = api_key or config.API_KEY
        if not resolved_key:
            print(f"\033[1;31mError: No API key configured.\033[0m")
            print(f"Set it via one of:")
            print(f"  export ANTHROPIC_API_KEY='sk-ant-xxx'    # for Anthropic")
            print(f"  export AI_API_KEY='xxx'                  # generic")
            print(f"  # or create a .env file (see .env.example)")
            sys.exit(1)

        # Build client with optional custom endpoint
        client_kwargs = {"api_key": resolved_key, "timeout": anthropic.Timeout(connect=10.0, read=120.0, write=120.0, pool=120.0)}
        if base_url or config.BASE_URL:
            client_kwargs["base_url"] = base_url or config.BASE_URL
        self.client = anthropic.Anthropic(**client_kwargs)

        self.thinking_config = config.THINKING_CONFIG
        self.provider = config.PROVIDER

        # Sub-systems
        self.memory = MemoryStore()
        self.rag = RAGSystem()
        self.repair_agent = RepairAgent(self.client)
        self.script_repairer = ScriptRepairer(self.client)
        self.experience = ExperienceStore()
        self.prompt_evolution = PromptEvolution(self.client)
        self.tool_stats = ToolStats()
        self.reflexion = ReflexionAgent(self.client)
        self.fast_rules = FastRuleEngine()
        # E6: share the same ExperienceStore instance across all co-evolution agents
        self.co_evolution = CoEvolutionCoordinator(self.client, pool=self.experience)

        # Skills
        self.skills = discover_skills(config.SKILLS_DIR)

        # ReAct loop (lazy-init, shares client / tools / dispatcher)
        self._react_loop: ReActLoop | None = None

        # Query router (auto-selects react vs agent mode)
        self.router = QueryRouter(client=self.client if config.AUTO_ROUTE else None)

        # Conversation state
        self.conversation: list[dict] = []

        # Self-healing state
        self._error_budget: dict[str, int] = {}  # "tool:input_hash" → retry count
        self._consecutive_errors: int = 0
        self._turn_checkpoint: int = 0  # conversation length at user turn start
        self._error_log: list[dict] = []

        # Tools: base tools + skill scripts
        skill_tools = [t for s in self.skills.values() for t in s.tools]
        self.tools = [*MEMORY_TOOLS, RAG_TOOL, *FILE_TOOLS, *WEB_TOOLS, *SHELL_TOOLS, *skill_tools]

        # System prompt: base + CLAUDE.md + skill prompts + references
        self.system = self._build_system_prompt(system_prompt)
        # E1: keep a clean copy for experience injection (never mutate this)
        self._clean_system = self.system

    # ── Main Loop ────────────────────────────────────────────────────────

    def run(self):
        """Run the interactive agent loop."""
        print(f"\n{config.COLOR_SYSTEM}═══ AI Agent with Memory & RAG ═══")
        print(f"Provider: {self.provider} | Model: {self.model}")
        print(f"Thinking: {'adaptive' if self.thinking_config else 'disabled'} | Tools: {len(self.tools)}")
        if self.skills:
            print(f"Skills: {', '.join(self.skills.keys())}")
        if config.BASE_URL:
            print(f"Endpoint: {config.BASE_URL}")
        if os.path.isfile(config.CLAUDE_MD_FILE):
            print(f"CLAUDE.md: {config.CLAUDE_MD_FILE}")
        if os.path.isfile(config.CWD_CLAUDE_MD) and os.path.abspath(config.CWD_CLAUDE_MD) != os.path.abspath(config.CLAUDE_MD_FILE):
            print(f"CWD CLAUDE.md: {config.CWD_CLAUDE_MD}")
        route_status = "auto-routing ON" if config.AUTO_ROUTE else "auto-routing OFF (set AI_AUTO_ROUTE=true)"
        print(f"Routing: {route_status}")
        print(f"Type /quit to exit, /clear to reset, /react <query> to force ReAct, /skills to list{config.COLOR_RESET}\n")

        while True:
            try:
                user_input = input(f"{config.COLOR_USER}You: {config.COLOR_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{config.COLOR_SYSTEM}Goodbye!{config.COLOR_RESET}")
                self.prompt_evolution.reflect(self.conversation)
                break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print(f"{config.COLOR_SYSTEM}Goodbye!{config.COLOR_RESET}")
                self.prompt_evolution.reflect(self.conversation)
                break
            if user_input.lower() == "/clear":
                self.conversation.clear()
                print(f"{config.COLOR_SYSTEM}Conversation cleared.{config.COLOR_RESET}\n")
                continue

            # ── ReAct mode (/react <query>) ─────────────────────────────────
            if user_input.lower().startswith("/react"):
                query = user_input[6:].strip()
                if not query:
                    print(f"{config.COLOR_SYSTEM}Usage: /react <your question or task>{config.COLOR_RESET}\n")
                else:
                    self._run_react(query)
                continue

            # ── Skill commands ──────────────────────────────────────────────

            if user_input.lower() == "/skills":
                self._cmd_list_skills()
                continue

            # /<skill_name> [args] — direct skill trigger
            skill_match = self._match_skill_command(user_input)
            if skill_match:
                skill_name, args = skill_match
                self._trigger_skill(skill_name, args)
                continue

            # +skill <name> — hot load
            if user_input.startswith("+skill "):
                self._cmd_load_skill(user_input[7:].strip())
                continue

            # -skill <name> — hot unload
            if user_input.startswith("-skill "):
                self._cmd_unload_skill(user_input[7:].strip())
                continue

            if config.AUTO_ROUTE:
                decision = self.router.route(user_input, len(self.conversation))
                if decision.mode == "react":
                    print(
                        f"{config.COLOR_SYSTEM}[Router → ReAct] "
                        f"{decision.reason} (conf={decision.confidence:.0%}){config.COLOR_RESET}"
                    )
                    self._run_react(user_input)
                    continue

            self.conversation.append({"role": "user", "content": user_input})
            self._agent_loop()

    # ── Skill Commands ────────────────────────────────────────────────────

    def _cmd_list_skills(self):
        """List all loaded skills."""
        if not self.skills:
            print(f"{config.COLOR_SYSTEM}No skills loaded.{config.COLOR_RESET}\n")
            return
        print(f"{config.COLOR_SYSTEM}Loaded Skills:{config.COLOR_RESET}")
        for skill in self.skills.values():
            tools_str = ", ".join(t["name"] for t in skill.tools) if skill.tools else "none"
            triggers_str = ", ".join(skill.triggers[:5]) if skill.triggers else "none"
            print(f"  {config.COLOR_TOOL}{skill.name}{config.COLOR_RESET} — {skill.description}")
            print(f"    triggers: {triggers_str} | scripts: {tools_str}")
        print()

    def _match_skill_command(self, user_input: str) -> tuple[str, str] | None:
        """Check if input matches /<skill_name> or a trigger keyword."""
        if not user_input.startswith("/"):
            return None

        parts = user_input[1:].split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Exact match by skill name
        for skill in self.skills.values():
            if cmd == skill.name.lower():
                return (skill.name, args)

        # Match by trigger
        for skill in self.skills.values():
            if cmd in [t.lower() for t in skill.triggers]:
                return (skill.name, args)

        return None

    def _trigger_skill(self, skill_name: str, args: str):
        """Directly trigger a skill, inject context into conversation."""
        skill = self.skills.get(skill_name)
        if not skill:
            print(f"{config.COLOR_SYSTEM}Skill '{skill_name}' not loaded.{config.COLOR_RESET}\n")
            return

        # Auto-load dependencies (orchestration)
        self._ensure_dependencies(skill)

        print(f"{config.COLOR_TOOL}[Skill: {skill_name}]{config.COLOR_RESET} {args}")

        # Build a contextual user message that includes skill prompt + args
        context = args
        if skill.prompt:
            context = f"[Using skill: {skill_name}]\n{skill.prompt}\n\nUser request: {args or skill.description}"
        self.conversation.append({"role": "user", "content": context})

        # Load skill state if exists
        state = skill.load_state()
        if state:
            print(f"{config.COLOR_SYSTEM}  Restored skill state ({len(state)} keys){config.COLOR_RESET}")

        self._agent_loop()

        # Save skill state after execution
        skill.save_state()

    def _ensure_dependencies(self, skill):
        """Auto-load any skills that this skill depends on."""
        for dep_name in skill.depends_on:
            if dep_name not in self.skills:
                dep_dir = os.path.join(config.SKILLS_DIR, dep_name)
                if os.path.isdir(dep_dir) and os.path.isfile(os.path.join(dep_dir, "SKILL.md")):
                    self._cmd_load_skill(dep_name)
                else:
                    print(f"{config.COLOR_SYSTEM}Warning: dependency '{dep_name}' not found{config.COLOR_RESET}")

    def _cmd_load_skill(self, skill_name: str):
        """Hot-load a skill at runtime."""
        if skill_name in self.skills:
            print(f"{config.COLOR_SYSTEM}Skill '{skill_name}' already loaded.{config.COLOR_RESET}\n")
            return

        skill_dir = os.path.join(config.SKILLS_DIR, skill_name)
        if not os.path.isdir(skill_dir) or not os.path.isfile(os.path.join(skill_dir, "SKILL.md")):
            print(f"{config.COLOR_SYSTEM}Skill '{skill_name}' not found in {config.SKILLS_DIR}{config.COLOR_RESET}\n")
            return

        from skills.base import Skill
        try:
            skill = Skill(skill_dir)
        except Exception as e:
            print(f"{config.COLOR_SYSTEM}Failed to load skill '{skill_name}': {e}{config.COLOR_RESET}\n")
            return

        # Register
        self.skills[skill.name] = skill
        self.tools.extend(skill.tools)

        # Rebuild system prompt
        self._rebuild_system_prompt()

        print(f"{config.COLOR_TOOL}+ Skill '{skill.name}' loaded.{config.COLOR_RESET}")
        print(f"  tools: {', '.join(t['name'] for t in skill.tools) or 'none'}")
        print()

    def _cmd_unload_skill(self, skill_name: str):
        """Hot-unload a skill at runtime."""
        skill = self.skills.pop(skill_name, None)
        if not skill:
            print(f"{config.COLOR_SYSTEM}Skill '{skill_name}' not loaded.{config.COLOR_RESET}\n")
            return

        # Remove its tools
        skill_tool_names = skill.tool_names
        self.tools = [t for t in self.tools if t.get("name") not in skill_tool_names]

        # Rebuild system prompt
        self._rebuild_system_prompt()

        print(f"{config.COLOR_TOOL}- Skill '{skill_name}' unloaded.{config.COLOR_RESET}\n")

    # ── E1: Experience Injection ──────────────────────────────────────────

    def _inject_experiences(self, query: str):
        """Prepend relevant past experiences to the system prompt for this turn.

        Always starts from _clean_system to prevent unbounded growth across turns.
        """
        experiences = self.experience.retrieve(query)
        if not experiences:
            self.system = self._clean_system
            return
        snippet = self.experience.format_for_prompt(experiences)
        base = self._clean_system[0]["text"]          # always from the clean copy
        self.system = [
            {
                "type": "text",
                "text": base + f"\n\n{snippet}",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _current_task_summary(self) -> str:
        """Extract a short summary of the current task from conversation."""
        if not self.conversation:
            return ""
        for msg in reversed(self.conversation):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"][:100]
        return ""

    def _rebuild_system_prompt(self):
        """Rebuild system prompt from base + CLAUDE.md + all skill prompts."""
        self.system = self._build_system_prompt()
        self._clean_system = self.system  # keep clean copy in sync

    def _build_system_prompt(self, base: str = config.SYSTEM_PROMPT) -> list[dict]:
        """Assemble the full system prompt and return an Anthropic system block."""
        full_prompt = base

        # E2: inject evolved prompt learnings from past sessions
        evolved = self.prompt_evolution.load()
        if evolved:
            full_prompt += f"\n\n## Evolved Learnings (from past sessions)\n{evolved}"

        # E3: inject tool reliability ranking
        reliability_table = self.tool_stats.format_for_prompt()
        if reliability_table:
            full_prompt += f"\n\n{reliability_table}"

        # Project-level CLAUDE.md
        if os.path.isfile(config.CLAUDE_MD_FILE):
            with open(config.CLAUDE_MD_FILE, encoding="utf-8") as _f:
                claude_md = _f.read().strip()
            if claude_md:
                full_prompt += f"\n\n## Project Instructions (CLAUDE.md)\n{claude_md}"

        # Working-directory CLAUDE.md
        if os.path.isfile(config.CWD_CLAUDE_MD) and os.path.abspath(config.CWD_CLAUDE_MD) != os.path.abspath(config.CLAUDE_MD_FILE):
            with open(config.CWD_CLAUDE_MD, encoding="utf-8") as _f:
                cwd_md = _f.read().strip()
            if cwd_md:
                full_prompt += f"\n\n## Working Directory Context ({config.CWD_CLAUDE_MD})\n{cwd_md}"

        # Skill prompts + references
        skill_prompt_parts = []
        for skill in self.skills.values():
            if skill.prompt:
                skill_prompt_parts.append(f"## Skill: {skill.name}\n{skill.prompt}")
            ref = skill.get_reference()
            if ref:
                skill_prompt_parts.append(f"### {skill.name} Reference\n{ref}")

        if skill_prompt_parts:
            full_prompt += "\n\n## Active Skills\n\n" + "\n\n".join(skill_prompt_parts)

        return [
            {
                "type": "text",
                "text": full_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    # ── ReAct ────────────────────────────────────────────────────────────

    def _get_react_loop(self) -> ReActLoop:
        """Lazy-init the ReActLoop, sharing this agent's client and dispatcher."""
        if self._react_loop is None:
            self._react_loop = ReActLoop(
                client=self.client,
                model=self.model,
                tools=self.tools,
                tool_dispatcher=self._dispatch_tool,
                thinking_config=self.thinking_config,
                max_tokens=self.max_tokens,
            )
        return self._react_loop

    def _run_react(self, query: str):
        """Run one query through the explicit ReAct Thought→Action→Observation loop."""
        loop = self._get_react_loop()
        # Refresh tools in case skills were hot-loaded since last call
        loop.tools = self.tools

        answer, trace = loop.run(query=query, system=self.system)

        # Append to shared conversation so the user can follow up naturally
        self.conversation.append({"role": "user", "content": query})
        self.conversation.append({
            "role": "assistant",
            "content": answer or "[no answer]",
        })

        print(
            f"\n{config.COLOR_SYSTEM}[ReAct] Completed in {trace.total_turns} step(s).{config.COLOR_RESET}\n"
        )

    # ── Agentic Loop ─────────────────────────────────────────────────────

    _MAX_CONSECUTIVE_ERRORS = 5   # L5: rollback threshold
    _MAX_SAME_CALL_RETRIES = 3    # L4: same-call retry budget

    def _agent_loop(self):
        """Run the tool-use loop with self-healing (retry budget + checkpoint rollback)."""
        max_iterations = 30

        # L5: checkpoint at user turn start
        self._turn_checkpoint = len(self.conversation)
        self._turn_start_time: float = time.time()   # E4: used by _maybe_reflect
        self._consecutive_errors = 0
        self._error_budget.clear()

        # E1: inject relevant past experiences into system prompt for this turn
        user_msg = ""
        if self.conversation:
            last = self.conversation[-1]
            if isinstance(last.get("content"), str):
                user_msg = last["content"]
        self._inject_experiences(user_msg)

        _rolled_back = False

        for iteration in range(max_iterations):
            try:
                message = self._stream_response()
            except Exception as e:
                # Prevent silent failure — feed error back and break gracefully
                print(f"\n{config.COLOR_SYSTEM}[Error] API call failed: {e}{config.COLOR_RESET}")
                self.conversation.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM ERROR] The last API call failed: {e}\n\n"
                        "Please analyze the error and try a different approach."
                    ),
                })
                break

            # Append full assistant response (preserves thinking blocks for multi-turn)
            self.conversation.append({"role": "assistant", "content": message.content})

            if message.stop_reason == "end_turn":
                break

            if message.stop_reason == "tool_use":
                tool_use_blocks = [b for b in message.content if b.type == "tool_use"]
                tool_results = self._execute_tools(tool_use_blocks)

                # L4: retry budget — detect repeated failures on same call
                tool_results = self._check_retry_budget(tool_use_blocks, tool_results)

                # L5: check consecutive errors for rollback
                has_error = any(
                    "error" in str(r.get("content", "")).lower()[:200]
                    for r in tool_results
                )
                if has_error:
                    self._consecutive_errors += 1
                    if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                        self._rollback_and_replan()
                        _rolled_back = True
                        continue
                else:
                    self._consecutive_errors = 0

                self.conversation.append({"role": "user", "content": tool_results})
                continue

            if message.stop_reason == "max_tokens":
                print(f"\n{config.COLOR_SYSTEM}[Warning: hit max_tokens limit]{config.COLOR_RESET}")
                break

            break

        # E4: post-task reflexion (skip if rolled back — state is already reset)
        if not _rolled_back:
            self._maybe_reflect(iteration + 1)

        print()  # blank line after response

    def _check_retry_budget(self, tool_use_blocks: list, tool_results: list[dict]) -> list[dict]:
        """L4: Detect when the same tool+input combination fails repeatedly."""
        for i, block in enumerate(tool_use_blocks):
            content = str(tool_results[i].get("content", ""))
            if "[Tool Error]" not in content and '"error"' not in content[:100].lower():
                continue

            # Hash the call to detect repetitions (json.dumps handles nested types)
            call_key = f"{block.name}:{hash(json.dumps(block.input, sort_keys=True))}"
            self._error_budget[call_key] = self._error_budget.get(call_key, 0) + 1

            if self._error_budget[call_key] >= self._MAX_SAME_CALL_RETRIES:
                print(
                    f"  {config.COLOR_SYSTEM}[Retry Budget] {block.name} failed "
                    f"{self._error_budget[call_key]} times with similar params{config.COLOR_RESET}"
                )
                tool_results[i]["content"] += (
                    "\n\n[SYSTEM WARNING] This is your 3rd+ failed attempt with the same approach. "
                    "You MUST try a completely different strategy. "
                    "Do NOT call this tool with similar parameters again."
                )
        return tool_results

    def _rollback_and_replan(self):
        """L5: Rollback conversation to checkpoint and force re-planning."""
        print(
            f"\n  {config.COLOR_SYSTEM}[Self-Healing] "
            f"{self._MAX_CONSECUTIVE_ERRORS} consecutive errors — rolling back and re-planning{config.COLOR_RESET}"
        )
        # Rollback conversation
        self.conversation[:] = self.conversation[: self._turn_checkpoint]
        self._consecutive_errors = 0
        self._error_budget.clear()

        # Inject forced re-plan message
        self.conversation.append({
            "role": "user",
            "content": (
                "[SYSTEM] Your recent tool calls have all failed. "
                "I've rolled back the conversation to before these attempts.\n\n"
                "Please RE-PLAN your approach from scratch:\n"
                "1. Analyze WHY the previous attempts failed\n"
                "2. Propose a completely different strategy\n"
                "3. Execute step by step, verifying each step succeeds"
            ),
        })

    def _maybe_reflect(self, turns: int):
        """E4: Trigger reflexion if the task was complex enough."""
        repairs = sum(1 for e in self._error_log
                      if e.get("timestamp", 0) >= self._turn_start_time)
        if not self.reflexion.should_reflect(turns, repairs, self._consecutive_errors):
            return
        task = self._current_task_summary()
        result = self.reflexion.reflect(
            task_summary=task,
            turns=turns,
            repairs=repairs,
            consecutive_errors=self._consecutive_errors,
            error_log=self._error_log,
        )
        if result and result.get("improvement_rule"):
            # Store the improvement rule as a high-value experience
            self.experience.record(
                task_summary=task,
                error_pattern="reflexion_insight",
                fix_strategy=result["improvement_rule"],
                tool_name="(reflexion)",
                success=True,
            )

    # ── Streaming ────────────────────────────────────────────────────────

    _MAX_CONVERSATION_PAIRS = 40  # keep last N user/assistant pairs

    def _stream_response(self, max_retries: int = 10) -> anthropic.Message:
        """Stream a response with exponential backoff on retryable errors."""
        # Trim old conversation to prevent context overflow
        self._trim_conversation()

        stream_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "tools": self.tools,
            "messages": self.conversation,
        }
        if self.thinking_config:
            stream_kwargs["thinking"] = self.thinking_config

        for attempt in range(max_retries + 1):
            try:
                with self.client.messages.stream(**stream_kwargs) as stream:
                    for event in stream:
                        self._handle_event(event)

                    return stream.get_final_message()
            except anthropic.APIStatusError as e:
                retryable = e.status_code in (429, 503, 529) or (
                    isinstance(e.body, dict)
                    and isinstance(e.body.get("error"), dict)
                    and e.body["error"].get("code") in ("1302", "1305", "overloaded")
                )
                if not retryable or attempt == max_retries:
                    raise
                wait = 2 ** attempt * 1.5
                print(
                    f"\n{config.COLOR_SYSTEM}[Retry {attempt + 1}/{max_retries}] "
                    f"API busy, waiting {wait:.0f}s...{config.COLOR_RESET}",
                    flush=True,
                )
                time.sleep(wait)

    def _trim_conversation(self):
        """Trim old conversation pairs to stay within context window limits.

        Keeps the first user message (task context) and the last N pairs.
        """
        if len(self.conversation) <= self._MAX_CONVERSATION_PAIRS * 2:
            return

        # Preserve the first user message for task context
        first_user = None
        if self.conversation and self.conversation[0].get("role") == "user":
            first_user = self.conversation[0]

        # Keep last N pairs (each pair = user + assistant)
        keep_from = max(1, len(self.conversation) - self._MAX_CONVERSATION_PAIRS * 2)
        trimmed = self.conversation[keep_from:]

        if first_user:
            trimmed = [first_user] + trimmed

        removed = len(self.conversation) - len(trimmed)
        if removed > 0:
            print(
                f"  {config.COLOR_SYSTEM}[Context] Trimmed {removed} old messages "
                f"to prevent context overflow{config.COLOR_RESET}",
                flush=True,
            )
            self.conversation[:] = trimmed

    def _handle_event(self, event):
        """Print streaming events with styled output."""
        if event.type == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                print(f"\n{config.COLOR_THINKING}[Thinking]{config.COLOR_RESET} ", end="", flush=True)
            elif block.type == "text":
                print(f"\n{config.COLOR_TOOL}[Assistant]{config.COLOR_RESET} ", end="", flush=True)
            elif block.type == "tool_use":
                print(
                    f"\n{config.COLOR_TOOL}[Tool: {block.name}]{config.COLOR_RESET}",
                    end="",
                    flush=True,
                )

        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "thinking_delta":
                print(
                    f"{config.COLOR_THINKING}{delta.thinking}{config.COLOR_RESET}",
                    end="",
                    flush=True,
                )
            elif delta.type == "text_delta":
                print(delta.text, end="", flush=True)
            elif delta.type == "input_json_delta":
                # Tool input streaming — show partial JSON
                print(
                    f"{config.COLOR_SYSTEM}{delta.partial_json}{config.COLOR_RESET}",
                    end="",
                    flush=True,
                )

        elif event.type == "content_block_stop":
            print()  # newline after each block

    # ── Tool Execution ───────────────────────────────────────────────────

    _TOOL_ERROR_TEMPLATE = (
        "[Tool Error] {tool_name}\n"
        "Error: {error_type}: {message}\n"
        "Input: {input}\n"
        "Hint: Analyze the error carefully. Fix your parameters if needed, "
        "or try an alternative approach. Do NOT repeat the exact same call."
    )

    def _execute_tools(self, tool_use_blocks: list) -> list[dict]:
        """Execute tool calls with error classification and auto-retry for network errors."""
        results = []
        for block in tool_use_blocks:
            print(f"{config.COLOR_SYSTEM}  Executing: {block.name}({block.input}){config.COLOR_RESET}")
            result_str = self._execute_single_tool(block)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        return results

    def _execute_single_tool(self, block) -> str:
        """Execute one tool call with error classification, auto-retry, and RepairAgent diagnosis."""
        try:
            result = self._dispatch_tool(block.name, block.input)
            # E3: track reliability (no experience entry — success adds no repair knowledge)
            self.tool_stats.record(block.name, success=True)
            return result
        except Exception as e:
            classification = classify_error(e, block.name, block.input)
            self._log_tool_error(block.name, e, block.input)
            self.tool_stats.record(block.name, success=False)

            # Auto-retry network errors once (no LLM turn consumed)
            if classification.strategy == "auto_retry":
                wait = 1.5
                print(
                    f"  {config.COLOR_SYSTEM}[Auto-retry] {classification.detail}, "
                    f"waiting {wait}s...{config.COLOR_RESET}",
                    flush=True,
                )
                time.sleep(wait)
                try:
                    result = self._dispatch_tool(block.name, block.input)
                    # E1: record that auto-retry fixed a network error
                    self.experience.record(
                        task_summary=self._current_task_summary(),
                        error_pattern=str(e),
                        fix_strategy="auto_retry_once",
                        tool_name=block.name,
                        success=True,
                    )
                    self.tool_stats.record(block.name, success=True)
                    return result
                except Exception as e2:
                    self.experience.record(
                        task_summary=self._current_task_summary(),
                        error_pattern=str(e2),
                        fix_strategy="auto_retry_failed",
                        tool_name=block.name,
                        success=False,
                    )
                    return self._TOOL_ERROR_TEMPLATE.format(
                        tool_name=block.name,
                        error_type=type(e2).__name__,
                        message=f"{e2} (auto-retry also failed)",
                        input=json.dumps(block.input, ensure_ascii=False),
                    )

            # E5 Fast Path: check for a promoted rule before calling LLM
            fast_hint = self.fast_rules.fast_match(str(e), block.name)

            # L6: Dispatch to RepairAgent for diagnosis on non-network errors
            error_report = self._TOOL_ERROR_TEMPLATE.format(
                tool_name=block.name,
                error_type=type(e).__name__,
                message=str(e),
                input=json.dumps(block.input, ensure_ascii=False),
            )

            if fast_hint:
                error_report += f"\n\n{fast_hint}"

            repair_hint = self._diagnose_with_repair_agent(block, str(e))
            if repair_hint:
                error_report += f"\n\n{repair_hint}"
                fix_strategy = repair_hint[:300]
            else:
                # E6: fall back to co-evolution pipeline only when RepairAgent gives nothing
                co_hint = self.co_evolution.handle_error(block.name, str(e), block.input)
                if co_hint:
                    error_report += f"\n\n{co_hint}"
                fix_strategy = co_hint[:300] if co_hint else "no_fix_found"

            # E1: record repair attempt
            self.experience.record(
                task_summary=self._current_task_summary(),
                error_pattern=str(e),
                fix_strategy=fix_strategy,
                tool_name=block.name,
                success=False,
            )
            # E5 Slow Path: record for potential rule promotion
            self.fast_rules.record_repair(
                error_msg=str(e),
                tool_name=block.name,
                fix_strategy=fix_strategy,
                success=False,
            )

            return error_report

    def _diagnose_with_repair_agent(self, block, error_message: str) -> str | None:
        """L6: Ask the RepairAgent to diagnose a tool error and suggest a fix."""
        # Find tool schema for the repair agent
        tool_schema = None
        for tool_def in self.tools:
            if tool_def.get("name") == block.name:
                tool_schema = tool_def
                break

        try:
            print(
                f"  {config.COLOR_SYSTEM}[RepairAgent] Diagnosing {block.name} error...{config.COLOR_RESET}",
                flush=True,
            )
            diagnosis = self.repair_agent.diagnose(
                tool_name=block.name,
                tool_input=block.input,
                error_message=error_message,
                tool_schema=tool_schema,
            )
            return self.repair_agent.format_repair_hint(diagnosis)
        except Exception as e:
            # RepairAgent itself failed — don't block the main agent
            print(
                f"  {config.COLOR_SYSTEM}[RepairAgent] Diagnosis failed: {e}{config.COLOR_RESET}",
                flush=True,
            )
            return None

    def _log_tool_error(self, tool_name: str, error: Exception, tool_input: dict):
        """Track tool errors for diagnostics."""
        self._error_log.append({
            "tool": tool_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "input_snapshot": {k: v for k, v in tool_input.items()},
            "timestamp": time.time(),
        })

    def _dispatch_tool(self, name: str, tool_input: dict) -> str:
        """Route a tool call to the appropriate handler."""
        # Check skill scripts first (includes orchestrated skills)
        for skill in self.skills.values():
            if name in skill.tool_names:
                result = skill.handle(name, tool_input)
                # L7: If skill script returned an error, attempt self-repair
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict) and parsed.get("error"):
                            repaired = self._try_self_repair(skill, name, tool_input, parsed["error"])
                            if repaired is not None:
                                return repaired
                    except (json.JSONDecodeError, TypeError):
                        pass
                return result
        # Global tools
        if name in ("save_memory", "search_memory"):
            return handle_memory_tool(name, tool_input, self.memory)
        if name == "search_documents":
            return handle_rag_tool(tool_input, self.rag)
        if name in ("read_file", "write_file", "list_files", "search_content"):
            return handle_file_tool(name, tool_input)
        if name in ("fetch_url", "web_search"):
            return handle_web_tool(name, tool_input)
        if name in ("run_command",):
            return handle_shell_tool(name, tool_input)
        return f"Unknown tool: {name}"

    def _try_self_repair(self, skill, tool_name: str, tool_input: dict, error_msg: str) -> str | None:
        """L7: Attempt to self-repair a failing skill script and retry."""
        script_path = skill._scripts.get(tool_name)
        if not script_path or not script_path.exists():
            return None
        if not self.script_repairer.can_repair(str(script_path)):
            return None

        # Read original source
        try:
            source = script_path.read_text(encoding="utf-8")
        except OSError:
            return None

        # Ask LLM to fix the script
        fixed = self.script_repairer.repair(
            script_path=str(script_path),
            source=source,
            error=error_msg,
            tool_input=tool_input,
        )
        if fixed is None:
            return None

        # Retry with the fixed script
        try:
            return skill.handle(tool_name, tool_input)
        except Exception as e:
            # Fixed version also failed — report both errors
            return json.dumps({
                "error": f"Self-repair attempted but still failed: {e}",
                "original_error": error_msg,
                "repair_attempted": True,
            }, ensure_ascii=False)
