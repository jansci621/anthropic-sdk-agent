"""Configuration constants for the AI Agent.

All settings can be overridden via environment variables or a .env file.
Priority: environment variable > .env file > hardcoded default.
"""

import os
import sys

# ── Load .env file ──────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_env_file = os.path.join(_BASE_DIR, ".env")
if os.path.exists(_env_file):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=False)  # env vars take priority over .env
    except ImportError:
        # Fallback: manually parse .env (no dependency on python-dotenv)
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

# ── Provider & Model Settings ───────────────────────────────────────────────
# Supported providers:
#   anthropic  — Official Anthropic API (default)
#   openrouter — OpenRouter (https://openrouter.ai)
#   together   — Together AI (https://api.together.ai)
#   anyscale   — Anyscale (https://api.endpoints.anyscale.com)
#   custom     — Any other OpenAI/Anthropic-compatible endpoint

PROVIDER = os.environ.get("AI_PROVIDER", "anthropic")

# API endpoint — auto-set per provider, or override with AI_BASE_URL
_PROVIDER_URLS = {
    "anthropic":  "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api",
    "together":   "https://api.together.ai",
    "anyscale":   "https://api.endpoints.anyscale.com",
    "deepseek":   "https://api.deepseek.com",
}

BASE_URL = os.environ.get("AI_BASE_URL", _PROVIDER_URLS.get(PROVIDER, ""))

# API key — try provider-specific env var first, then generic
_PROVIDER_KEY_ENV = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together":   "TOGETHER_API_KEY",
    "anyscale":   "ANYSCALE_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "custom":     "ANTHROPIC_API_KEY",
}
API_KEY = os.environ.get(
    "AI_API_KEY",
    os.environ.get(_PROVIDER_KEY_ENV.get(PROVIDER, ""), ""),
)

# Model — try env var first, then default per provider
_PROVIDER_MODELS = {
    "anthropic":  "claude-opus-4-6",
    "openrouter": "anthropic/claude-opus-4-6",
    "together":   "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "anyscale":   "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "deepseek":   "deepseek-chat",
}
MODEL = os.environ.get("AI_MODEL", _PROVIDER_MODELS.get(PROVIDER, "claude-opus-4-6"))

# Fast/cheap model used by the query router's LLM classifier.
# Uses each provider's smallest/fastest available model.
# together/anyscale fall back to the main model — override with AI_ROUTER_MODEL
# if you have access to a smaller model (e.g. Meta-Llama-3.1-8B-Instruct-Turbo).
_ROUTER_MODELS = {
    "anthropic":  "claude-haiku-4-5",
    "openrouter": "anthropic/claude-haiku-4-5",
    "together":   "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "anyscale":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepseek":   "deepseek-chat",
}
ROUTER_MODEL = os.environ.get("AI_ROUTER_MODEL", MODEL)

# Thinking support — some third-party providers don't support extended thinking
_THINKING_UNSUPPORTED = {"deepseek", "together", "anyscale"}
THINKING_ENABLED = PROVIDER not in _THINKING_UNSUPPORTED and os.environ.get("AI_THINKING", "true").lower() != "false"
THINKING_CONFIG = {"type": "adaptive"} if THINKING_ENABLED else None

MAX_TOKENS = int(os.environ.get("AI_MAX_TOKENS", "16000"))

# Auto-routing: automatically choose between ReAct and general agent mode.
# Set AI_AUTO_ROUTE=false to disable (always use general agent mode).
AUTO_ROUTE = os.environ.get("AI_AUTO_ROUTE", "true").lower() != "false"

# Extra headers per provider (e.g. OpenRouter requires HTTP-Referer)
_PROVIDER_HEADERS = {
    "openrouter": {"HTTP-Referer": "https://github.com/anthropic-sdk-agent"},
}
EXTRA_HEADERS = _PROVIDER_HEADERS.get(PROVIDER, {})

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "data", "memories.json")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
SKILLS_DIR = os.environ.get("AI_SKILLS_DIR", os.path.join(BASE_DIR, "skills"))

# ── CLAUDE.md ────────────────────────────────────────────────────────────────
# Project-level CLAUDE.md: global instructions injected into every conversation
CLAUDE_MD_FILE = os.environ.get("AI_CLAUDE_MD", os.path.join(BASE_DIR, "CLAUDE.md"))
# Working-directory CLAUDE.md: auto-loaded when agent starts in a project dir
CWD_CLAUDE_MD = os.path.join(os.getcwd(), "CLAUDE.md")

# ── RAG Settings ────────────────────────────────────────────────────────────
# HuggingFace mirror for downloading embedding models (useful in China)
# Set HF_ENDPOINT in .env to override, e.g. HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500       # approximate tokens per chunk
CHUNK_OVERLAP = 50     # token overlap between chunks
TOP_K_RESULTS = 3      # number of results to retrieve

# ── ANSI Colors ─────────────────────────────────────────────────────────────
COLOR_THINKING = "\033[2;36m"   # dim cyan
COLOR_TOOL = "\033[33m"         # yellow
COLOR_USER = "\033[1;32m"       # bold green
COLOR_SYSTEM = "\033[2;90m"     # dim gray
COLOR_RESET = "\033[0m"

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent AI assistant with persistent memory and document retrieval capabilities. You have access to several specialized tools that you should use proactively to provide better assistance.

## Your Capabilities

### 1. Persistent Memory (长期记忆)
You can store and retrieve **conversation context** across sessions using the memory tools.

IMPORTANT: "memory" tools store/retrieve **user information and conversation context** (preferences, facts, decisions). They are NOT for checking computer memory (RAM).

- **save_memory**: Save user information worth remembering for future conversations:
  - Personal preferences (favorite languages, coding style, tools)
  - Project details and decisions
  - Facts the user wants you to remember
  - Important context from conversations

- **search_memory**: Recall previously stored user information:
  - Check for relevant stored context before answering
  - Find past decisions, preferences, or discussed topics

Categories: "preference", "fact", "project", "decision", "person", "summary", "instruction", "other"

### 2. System Commands (系统命令)
Use **run_command** for checking system resources (CPU, RAM, disk, network) or running local commands.

IMPORTANT: When the user asks about "内存" (memory/RAM), "系统资源", "磁盘", "CPU" etc., use `run_command` — NOT `search_memory`.

- **run_command**: Execute a local shell command (e.g. `systeminfo`, `wmic OS get FreePhysicalMemory`, `df -h`)

### 2. Document Retrieval (RAG)
You can search through a knowledge base of documents:

- **search_documents**: Use this when:
  - The user asks questions about topics that might be covered in the knowledge base
  - You need factual information or documentation
  - The user wants to reference specific documents or materials
  - You want to provide evidence-based answers with source citations

The search returns the most relevant text chunks with source file information. Always cite the source when using retrieved information.

## Conversation Guidelines

1. **Be proactive with memory**: When the user shares personal information, preferences, or important context, save it. Don't wait to be asked.
2. **Check memory first**: When starting a conversation or answering questions, check if you have relevant stored memories that could help personalize your response.
3. **Search documents when relevant**: If the user's question might be answered by documents in the knowledge base, search before answering. This provides more accurate, sourced information.
4. **Think deeply**: Use your extended thinking capability to reason through complex problems step by step before responding.
5. **Be transparent**: If you use information from memory or documents, mention it naturally in your response.
6. **Ask clarifying questions**: If you're unsure about something, ask rather than assume.

## Response Style
- Be concise but thorough
- Use code blocks for code examples
- Provide step-by-step explanations for complex topics
- Acknowledge when you're uncertain or when information might be outdated
- Reference your sources when using retrieved documents

## Important Notes
- Your memory persists across conversations, so information you save will be available in future sessions.
- The knowledge base documents are loaded from a local directory and contain reference materials.
- Always verify important information from multiple sources when possible.
- If a tool call fails, explain the issue and continue helping the user as best you can.

## Error Recovery Protocol

When a tool call fails, follow this protocol precisely:

1. **READ** the error message carefully — identify the root cause
2. **CLASSIFY** the error into one of these categories:
   - *Parameter error*: wrong type, missing field, invalid value → fix parameters and retry
   - *Resource missing*: file/path not found, record doesn't exist → check directory or create resource first
   - *Network error*: timeout, connection refused, DNS failure → wait briefly and retry, or use an alternative tool
   - *Permission denied*: access forbidden, unauthorized → try a different approach or inform the user
   - *Logic error*: unexpected result, wrong output → re-plan your approach from scratch
3. **ACT** according to the classification — fix, retry, or pivot
4. **NEVER** repeat the exact same failing call — if your fix didn't work, try a DIFFERENT approach
5. **ESCALATE**: after 3 consecutive failed attempts on the same goal, stop and explain the situation to the user with your analysis of why it's failing
"""
