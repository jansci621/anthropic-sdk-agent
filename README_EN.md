# AI Agent with Memory & RAG

[中文](./README.md) | English

An intelligent agent built on the Anthropic SDK, featuring persistent memory, RAG document retrieval, tool use, self-healing, self-evolution, **ReAct reasoning mode**, and **automatic routing**.

## Features

- **Streaming conversation** — Real-time output of thinking process and responses, with adaptive thinking support
- **Web UI** — DeepSeek-style web chat interface with live streaming of Thinking, tool calls, and results (`--web`)
- **Persistent memory** — Cross-session storage and retrieval of user preferences, project context, and more
- **RAG document retrieval** — Local knowledge base semantic search (FAISS dense + BM25 sparse hybrid retrieval, supports .txt/.md/.pdf/.json/.csv, embedding results cached locally)
- **Built-in tools** — File read/write, content search, URL fetching, web search, shell command execution
- **Skill system** — Hot-loadable/unloadable plugin skills (`+skill` / `-skill`)
- **Multi-provider support** — Anthropic / OpenRouter / Together / DeepSeek / Anyscale / custom endpoints
- **Self-healing (L1-L7)** — Automatic error classification, script repair, rollback & replanning, diagnostic repair
- **Self-evolution (E1-E6)** — Experience injection, prompt evolution, tool reliability tracking, reflective learning, fast-rule engine, co-evolution
- **ReAct mode** — Explicit Thought → Action → Observation loop with full traceability at each step
- **Auto-routing** — Automatically selects ReAct or general mode based on query characteristics

## Requirements

- Python >= 3.11
- pip

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd anthropic-sdk-agent
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

> On first install, `sentence-transformers` will download the embedding model (~90 MB). Ensure you have a stable internet connection.

### 3. Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and fill in your API Key:

```env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
```

Or set it directly via environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-xxx"
```

### 4. Run

**CLI mode:**

```bash
python main.py
```

**Web UI mode:**

```bash
python main.py --web
```

Open your browser at `http://localhost:8000` to use the DeepSeek-style web chat interface with real-time streaming of Thinking, tool calls, and results.

```bash
# Custom port
python main.py --web --port 8080

# Custom listen address
python main.py --web --host 127.0.0.1
```

CLI startup output:

```
═══ AI Agent with Memory & RAG ═══
Provider: anthropic | Model: claude-opus-4-6
Thinking: adaptive | Tools: 12
Routing: auto-routing ON
Type /quit to exit, /clear to reset, /react <query> to force ReAct, /skills to list

You:
```

## Commands

| Command | Description |
|---------|-------------|
| `/quit` `/exit` `/q` | Exit |
| `/clear` | Clear current conversation |
| `/react <query>` | Force ReAct mode for this query |
| `/skills` | List loaded skills |
| `/<skill_name> [args]` | Trigger a specific skill |
| `+skill <name>` | Hot-load a skill |
| `-skill <name>` | Unload a skill |

## ReAct Reasoning Mode

ReAct (Reasoning + Acting) renders each step of the agent's reasoning as a three-part structure:

```
Step N
├── [Thought N]      Model's reasoning about the current state
├── [Action N]       Tool call (name + arguments)
└── [Obs N]          Tool return value
```

The loop ends and outputs a Final Answer when the model stops calling tools.

### When to use ReAct

| Scenario | Recommended mode |
|----------|-----------------|
| Multi-hop QA (each step depends on the previous) | **ReAct** |
| Code architecture analysis / root cause tracing | **ReAct** |
| Debugging / diagnostic tasks | **ReAct** |
| Code generation / file editing | General |
| Memory operations / tasks requiring retry/self-healing | General |
| Short conversations or follow-ups | General |

### Manual trigger

```
You: /react how does the self-healing mechanism work?

──────────────────────────────────────────────────────────────
[ReAct] Query: how does the self-healing mechanism work?
──────────────────────────────────────────────────────────────
[Thought 1]  I should read agent.py to understand the loop...
[Action 1]   read_file({"path": "agent.py"})
[Obs 1]      class Agent: ...
[Thought 2]  Now I can trace the L4/L5 error budget logic...
[ReAct] Final Answer
The self-healing system has 7 layers: ...

[ReAct] Completed in 2 step(s).
```

## Auto-routing

When enabled (default), the agent automatically determines which mode to use after each user input, and prints the routing decision to the terminal:

```
[Router] react (react rule: why-question, conf=85%)
[Router] agent (agent rule: edit request, conf=85%)
```

### Routing decision hierarchy

```
User input
    │
    ├─ 1. Length < 25 chars ──────────────────→ General mode
    ├─ 2. Fast ReAct rules (regex, ~14 rules) → ReAct mode
    ├─ 3. Fast Agent rules (regex, ~8 rules)  → General mode
    ├─ 4. LLM classifier (Haiku, only on tie) → react / agent
    └─ 5. Default ────────────────────────────→ General mode
```

**ReAct trigger signals:** `how does X work` / `why did` / `step by step` / `investigate` / `compare X and Y` / `walk me through`

**General mode trigger signals:** `fix / write / create / run` / `remember` / single-word commands / pronoun continuations (`it` / `that`)

### Disable auto-routing

```bash
# Always use general mode
export AI_AUTO_ROUTE=false
```

You can still manually trigger ReAct with `/react <query>` when routing is disabled.

## Self-Healing System (L1-L7)

The agent has 7 built-in self-healing layers that automatically intervene on tool call failures:

| Layer | Module | Function |
|-------|--------|----------|
| L1 | `error_classifier.py` | Auto-classify errors (argument/network/permission/logic) and decide retry strategy |
| L2 | `repair.py` | RepairAgent — LLM-driven error diagnosis and fix suggestions |
| L3 | `self_repair.py` | ScriptRepairer — automatically repairs failed skill scripts and retries |
| L4 | built-in (agent.py) | Retry budget — force strategy switch after 3 failures on the same call |
| L5 | built-in (agent.py) | Consecutive error detection — rollback conversation and replan after 5 consecutive failures |
| L6 | built-in (agent.py) | RepairAgent diagnosis — auto-invoke LLM diagnosis for non-network errors |
| L7 | built-in (agent.py) | Skill script self-repair — LLM rewrites script code and retries |

## Self-Evolution System (E1-E6)

The agent continuously learns and optimizes during operation:

| Layer | Module | Function |
|-------|--------|----------|
| E1 | `experience.py` | Experience injection — inject historical repair experience into system prompt to avoid repeating mistakes |
| E2 | `prompt_evolution.py` | Prompt evolution — reflect at session end and accumulate reusable learnings |
| E3 | `tool_stats.py` | Tool reliability — track success rate per tool and inject reliability rankings into prompt |
| E4 | `reflexion.py` | Reflective learning — auto-reflect after complex tasks and extract improvement rules |
| E5 | `fast_rules.py` | Fast-rule engine — auto-promote high-frequency repair patterns to instant-match rules |
| E6 | `evolution.py` | Co-evolution — trigger co-evolution pipeline when RepairAgent cannot fix the issue |

## Configuration

All settings can be configured via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER` | `anthropic` | Provider: anthropic / openrouter / together / deepseek / anyscale / custom |
| `ANTHROPIC_API_KEY` | — | API Key (or use the generic `AI_API_KEY`) |
| `AI_BASE_URL` | auto | Custom API endpoint |
| `AI_MODEL` | auto | Model name (auto-selected per provider by default) |
| `AI_THINKING` | `true` | Enable extended thinking |
| `AI_MAX_TOKENS` | `64000` | Max output tokens (higher values significantly increase cost; 128K ≈ $2-4/call) |
| `AI_SKILLS_DIR` | `./skills` | Skills directory path |
| `AI_AUTO_ROUTE` | `true` | Auto-routing (ReAct vs general mode) |
| `AI_ROUTER_MODEL` | auto | Model for the routing LLM classifier (defaults to the fastest/cheapest per provider) |
| `HF_ENDPOINT` | — | HuggingFace mirror URL (e.g. `https://hf-mirror.com` for China) |

## Project Structure

```
anthropic-sdk-agent/
├── main.py                 # Entry point (CLI / Web mode)
├── agent.py                # Agent core: streaming loop + tool dispatch + self-heal/evolve
├── config.py               # Configuration management (multi-provider support)
├── event_bus.py            # Event bus: output abstraction layer (CLI / WebSocket)
├── web_server.py           # FastAPI web server (WebSocket real-time streaming)
├── react.py                # ReAct loop: Thought → Action → Observation
├── router.py               # Auto-router: fast rules + LLM classifier
├── memory.py               # Persistent memory system
├── rag.py                  # RAG retrieval (FAISS + BM25 hybrid, V3 semantic chunking, embedding cache)
├── file_tools.py           # File operation tools
├── web_tools.py            # Web tools (URL fetch, search)
├── shell_tools.py          # Shell command tool
├── error_classifier.py     # L1: Automatic error classification
├── repair.py               # L2/L6: RepairAgent error diagnosis
├── self_repair.py          # L3: Skill script self-repair
├── experience.py           # E1: Experience storage and retrieval
├── prompt_evolution.py     # E2: Prompt evolution
├── tool_stats.py           # E3: Tool reliability tracking
├── reflexion.py            # E4: Reflective learning
├── fast_rules.py           # E5: Fast-rule engine
├── evolution.py            # E6: Co-evolution coordinator
├── skills/                 # Extensible skills directory
│   ├── base.py             # Skill base class
│   └── weather/            # Example: weather skill
├── knowledge_base/         # RAG knowledge base (supports .md/.txt/.pdf/.json/.csv; index cached in .rag_cache/)
├── data/                   # Runtime persistent data
│   ├── memories.json       # Memory store
│   ├── experiences.json    # Experience store
│   ├── evolved_prompt.md   # Evolved prompt
│   ├── fast_rules.json     # Fast-rule data
│   └── tool_stats.json     # Tool statistics
├── web/
│   └── static/
│       └── index.html      # Web UI frontend (DeepSeek style)
├── .env.example            # Configuration template
└── requirements.txt        # Python dependencies
```

## Using Other Providers

### OpenRouter

```env
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-xxx
```

### DeepSeek

```env
AI_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
```

### Together AI

```env
AI_PROVIDER=together
TOGETHER_API_KEY=xxx
```

### Custom Endpoint

Works with any OpenAI/Anthropic-compatible API (e.g. vLLM, Ollama, LocalAI, One API):

```env
AI_PROVIDER=custom
AI_BASE_URL=https://your-api.example.com/v1
AI_API_KEY=xxx
AI_MODEL=your-model-name
```

Via environment variables:

```bash
export AI_PROVIDER=custom
export AI_BASE_URL=https://your-api.example.com/v1
export AI_API_KEY=your-key
export AI_MODEL=your-model-name
python main.py --web
```

## Docker Deployment

Built on CentOS 8 with Alibaba Cloud vault mirrors, running as a non-root user (`agent`, UID 1001).

### Build

```bash
docker build -t anthropic-sdk-agent .
```

### Run

**With Anthropic API:**

```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/knowledge_base:/app/knowledge_base \
  -e ANTHROPIC_API_KEY=sk-ant-xxx \
  anthropic-sdk-agent
```

**With a custom endpoint:**

```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/knowledge_base:/app/knowledge_base \
  -e AI_PROVIDER=custom \
  -e AI_BASE_URL=https://your-api.example.com/v1 \
  -e AI_API_KEY=your-key \
  -e AI_MODEL=your-model-name \
  anthropic-sdk-agent
```

**With a .env file:**

```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/knowledge_base:/app/knowledge_base \
  --env-file .env \
  anthropic-sdk-agent
```

**CLI mode (interactive):**

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=sk-ant-xxx \
  anthropic-sdk-agent \
  python3 main.py
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AI_PROVIDER` | No | Provider, default `anthropic`; set to `custom` for custom endpoints |
| `AI_BASE_URL` | Required for custom | Custom API endpoint URL |
| `AI_API_KEY` | Yes | API Key (or use provider-specific variable e.g. `ANTHROPIC_API_KEY`) |
| `AI_MODEL` | Required for custom | Model name |
| `AI_THINKING` | No | Enable thinking, default `true` |
| `AI_MAX_TOKENS` | No | Max output tokens, default `64000` |
| `HF_ENDPOINT` | No | HuggingFace mirror, e.g. `https://hf-mirror.com` |

### Volume Mounts

| Path | Description |
|------|-------------|
| `/app/data` | Runtime data (memory, experience, statistics, etc.) |
| `/app/knowledge_base` | RAG knowledge base documents |
| `/app/skills` | Skill plugins directory |

## License

MIT
