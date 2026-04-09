# AI Agent with Memory & RAG

基于 Anthropic SDK 构建的智能 Agent，支持持久化记忆、RAG 文档检索、工具调用、自我修复、自我进化，以及 **ReAct 推理模式**与**自动路由**。

## 特性

- **流式对话** — 实时输出思维过程和回复，支持 adaptive thinking
- **Web UI** — DeepSeek 风格的 Web 聊天界面，实时展示 Thinking、工具调用和结果（`--web`）
- **持久化记忆** — 跨会话存储和检索用户偏好、项目信息等
- **RAG 文档检索** — 本地知识库的语义搜索（基于 FAISS + sentence-transformers）
- **内置工具** — 文件读写、内容搜索、URL 抓取、Web 搜索、Shell 命令执行
- **技能系统** — 可热加载/卸载的插件式技能（`+skill` / `-skill`）
- **多 Provider 支持** — Anthropic / OpenRouter / Together / DeepSeek / Anyscale / 自定义端点
- **自我修复 (L1-L7)** — 自动分类错误、修复脚本、回滚重规划、诊断修复
- **自我进化 (E1-E6)** — 经验注入、提示词进化、工具可靠性追踪、反思学习、快规则引擎、协同进化
- **ReAct 模式** — 显式 Thought → Action → Observation 循环，每步可追溯
- **自动路由** — 根据查询特征自动选择 ReAct 或通用模式

## 环境要求

- Python >= 3.11
- pip

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd anthropic-sdk-agent
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

> 首次安装时 `sentence-transformers` 会下载嵌入模型（约 90MB），请确保网络通畅。

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
```

也可以直接通过环境变量设置：

```bash
export ANTHROPIC_API_KEY="sk-ant-xxx"
```

### 4. 运行

**CLI 模式：**

```bash
python main.py
```

**Web UI 模式：**

```bash
python main.py --web
```

启动后打开浏览器访问 `http://localhost:8000`，即可使用 DeepSeek 风格的 Web 聊天界面，支持实时流式输出 Thinking、工具调用和结果。

```bash
# 指定端口
python main.py --web --port 8080

# 指定监听地址
python main.py --web --host 127.0.0.1
```

CLI 模式启动后会看到如下提示：

```
═══ AI Agent with Memory & RAG ═══
Provider: anthropic | Model: claude-opus-4-6
Thinking: adaptive | Tools: 12
Routing: auto-routing ON
Type /quit to exit, /clear to reset, /react <query> to force ReAct, /skills to list

You:
```

## 交互命令

| 命令 | 说明 |
|------|------|
| `/quit` `/exit` `/q` | 退出 |
| `/clear` | 清除当前对话 |
| `/react <query>` | 强制使用 ReAct 模式执行该查询 |
| `/skills` | 列出已加载的技能 |
| `/<skill_name> [args]` | 触发指定技能 |
| `+skill <name>` | 热加载技能 |
| `-skill <name>` | 卸载技能 |

## ReAct 推理模式

ReAct（Reasoning + Acting）将 Agent 的每一步推理过程显式展示为三段结构：

```
Step N
├── [Thought N]      模型对当前状态的推理
├── [Action N]       工具调用（名称 + 参数）
└── [Obs N]          工具返回结果
```

当模型不再调用工具时，循环结束并输出 Final Answer。

### 适用场景

| 场景 | 推荐模式 |
|------|---------|
| 多跳问答（每步依赖上一步结果） | **ReAct** |
| 代码架构分析 / 原因追溯 | **ReAct** |
| 调试 / 诊断类任务 | **ReAct** |
| 代码生成 / 文件编辑 | 通用 |
| 记忆操作 / 需要重试自愈的任务 | 通用 |
| 简短对话或续接 | 通用 |

### 手动触发

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

## 自动路由

启用自动路由后（默认开启），Agent 会在每次用户输入后自动判断使用哪种模式，并在终端打印路由决策：

```
[Router] react (react rule: why-question, conf=85%)
[Router] agent (agent rule: edit request, conf=85%)
```

### 路由决策层级

```
用户输入
    │
    ├─ 1. 长度 < 25 字符 ──────────────────────→ 通用模式
    ├─ 2. Fast ReAct 规则（正则，约 14 条）──→ ReAct 模式
    ├─ 3. Fast Agent 规则（正则，约 8 条）───→ 通用模式
    ├─ 4. LLM 分类器（Haiku，仅在模糊时触发）→ react / agent
    └─ 5. 默认 ────────────────────────────────→ 通用模式
```

**ReAct 触发信号举例：** `how does X work` / `why did` / `step by step` / `investigate` / `compare X and Y` / `walk me through`

**通用模式触发信号举例：** `fix / write / create / run` / `remember` / 单词命令 / 代词续接（`it` / `that`）

### 禁用自动路由

```bash
# 始终使用通用模式
export AI_AUTO_ROUTE=false
```

禁用后仍可用 `/react <query>` 手动触发 ReAct。

## 自我修复系统 (L1-L7)

Agent 内置了 7 层自我修复机制，在工具调用失败时自动介入：

| 层级 | 模块 | 功能 |
|------|------|------|
| L1 | `error_classifier.py` | 错误自动分类（参数/网络/权限/逻辑），决定重试策略 |
| L2 | `repair.py` | RepairAgent — LLM 驱动的错误诊断与修复建议 |
| L3 | `self_repair.py` | ScriptRepairer — 自动修复失败的技能脚本并重试 |
| L4 | 内置 (agent.py) | 重试预算 — 同一调用失败 3 次后强制切换策略 |
| L5 | 内置 (agent.py) | 连续错误检测 — 5 次连续失败后回滚对话并重新规划 |
| L6 | 内置 (agent.py) | RepairAgent 诊断 — 非 network 错误自动调用 LLM 诊断 |
| L7 | 内置 (agent.py) | 技能脚本自修复 — LLM 重写脚本代码并重试 |

## 自我进化系统 (E1-E6)

Agent 在运行过程中持续学习和优化：

| 层级 | 模块 | 功能 |
|------|------|------|
| E1 | `experience.py` | 经验注入 — 将历史修复经验注入系统提示词，避免重复犯错 |
| E2 | `prompt_evolution.py` | 提示词进化 — 会话结束时反思并积累可复用的学习成果 |
| E3 | `tool_stats.py` | 工具可靠性 — 追踪每个工具的成功率，注入可靠性排名到提示词 |
| E4 | `reflexion.py` | 反思学习 — 复杂任务后自动反思，提取改进规则 |
| E5 | `fast_rules.py` | 快规则引擎 — 高频修复模式自动提升为即时匹配规则 |
| E6 | `evolution.py` | 协同进化 — RepairAgent 无法修复时触发协同进化管线 |

## 配置说明

所有配置项均可通过 `.env` 文件或环境变量设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AI_PROVIDER` | `anthropic` | 服务商：anthropic / openrouter / together / deepseek / anyscale / custom |
| `ANTHROPIC_API_KEY` | — | API Key（或使用 `AI_API_KEY` 通用变量） |
| `AI_BASE_URL` | 自动 | 自定义 API 端点 |
| `AI_MODEL` | 自动 | 模型名称（默认按 provider 自动选择） |
| `AI_THINKING` | `true` | 是否启用 extended thinking |
| `AI_MAX_TOKENS` | `16000` | 最大输出 token 数 |
| `AI_SKILLS_DIR` | `./skills` | 技能目录路径 |
| `AI_AUTO_ROUTE` | `true` | 自动路由（ReAct vs 通用模式） |
| `AI_ROUTER_MODEL` | 自动 | 路由 LLM 分类器使用的模型（默认为各 provider 最快/最便宜的型号） |
| `HF_ENDPOINT` | — | HuggingFace 镜像地址（国内可设为 `https://hf-mirror.com`） |

## 项目结构

```
anthropic-sdk-agent/
├── main.py                 # 入口（CLI / Web 模式）
├── agent.py                # Agent 核心：流式循环 + 工具调度 + 自修复/自进化
├── config.py               # 配置管理（多 Provider 支持）
├── event_bus.py            # 事件总线：输出抽象层（CLI / WebSocket）
├── web_server.py           # FastAPI Web 服务（WebSocket 实时流式推送）
├── react.py                # ReAct 循环：Thought → Action → Observation
├── router.py               # 自动路由：快规则 + LLM 分类器
├── memory.py               # 持久化记忆系统
├── rag.py                  # RAG 文档检索（FAISS）
├── file_tools.py           # 文件操作工具
├── web_tools.py            # Web 工具（URL 抓取、搜索）
├── shell_tools.py          # Shell 命令工具
├── error_classifier.py     # L1: 错误自动分类
├── repair.py               # L2/L6: RepairAgent 错误诊断
├── self_repair.py          # L3: 技能脚本自修复
├── experience.py           # E1: 经验存储与检索
├── prompt_evolution.py     # E2: 提示词进化
├── tool_stats.py           # E3: 工具可靠性追踪
├── reflexion.py            # E4: 反思学习
├── fast_rules.py           # E5: 快规则引擎
├── evolution.py            # E6: 协同进化协调器
├── skills/                 # 可扩展技能目录
│   ├── base.py             # 技能基类
│   └── weather/            # 示例：天气技能
├── knowledge_base/         # RAG 知识库（放入 .md/.txt 文件即可索引）
├── data/                   # 运行时持久化数据
│   ├── memories.json       # 记忆存储
│   ├── experiences.json    # 经验存储
│   ├── evolved_prompt.md   # 进化后的提示词
│   ├── fast_rules.json     # 快规则数据
│   └── tool_stats.json     # 工具统计
├── web/
│   └── static/
│       └── index.html      # Web UI 前端（DeepSeek 风格）
├── .env.example            # 配置模板
└── requirements.txt        # Python 依赖
```

## 使用其他 Provider

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

### 自定义端点

```env
AI_PROVIDER=custom
AI_BASE_URL=https://your-api.example.com
AI_API_KEY=xxx
AI_MODEL=your-model-name
```

## License

MIT
