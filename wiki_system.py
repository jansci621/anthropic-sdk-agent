"""LLM Wiki: compiled Markdown knowledge layer over raw sources.

The wiki layer complements RAG:
  - raw/sources keeps immutable source snapshots
  - wiki/ holds curated source, entity, concept, and synthesis pages
  - graph/ stores a deterministic wikilink graph for traversal and health checks
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import zipfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import config


# ── Tool Definitions (Anthropic API format) ─────────────────────────────────

WIKI_TOOLS = [
    {
        "name": "wiki_init",
        "description": (
            "Initialize the local LLM Wiki directory structure under knowledge_base. "
            "Creates raw/sources, wiki pages, index/log/overview/schema, and graph output directories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "force_rebuild": {
                    "type": "boolean",
                    "description": "Rebuild index, overview, and graph files after initialization.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "wiki_ingest",
        "description": (
            "Ingest a file or pasted content into the LLM Wiki. "
            "Performs a two-stage flow: analyze the source, then write immutable raw source snapshots "
            "plus source/entity/concept wiki pages, index, log, and wikilink graph."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a local source file to ingest. Relative paths resolve within the workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "Raw content to ingest when no path is available.",
                },
                "title": {
                    "type": "string",
                    "description": "Human-readable title for the source. Defaults to file stem or first heading.",
                },
                "source_type": {
                    "type": "string",
                    "description": "Source category such as document, note, code, url, meeting, or pasted.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "wiki_search",
        "description": (
            "Search compiled LLM Wiki pages before falling back to raw document RAG. "
            "Returns page titles, paths, types, excerpts, and scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for concepts, entities, source titles, or decisions.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of wiki pages to return (default 5).",
                },
                "include_system_pages": {
                    "type": "boolean",
                    "description": "Include index/log/schema/overview pages in search results.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "wiki_read",
        "description": (
            "Read a wiki page by title, slug, or relative path. "
            "Use after wiki_search to gather compiled context with citations and links."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "string",
                    "description": "Wiki page title, slug, or relative path such as concepts/rag.md.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 20000).",
                },
            },
            "required": ["page"],
        },
    },
    {
        "name": "wiki_health",
        "description": (
            "Check LLM Wiki health without using an LLM. "
            "Reports initialized files, page counts, broken links, orphan pages, and index coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "init_if_missing": {
                    "type": "boolean",
                    "description": "Create the wiki structure first if it does not exist (default true).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "wiki_lint",
        "description": (
            "Run deterministic LLM Wiki lint checks. "
            "Currently aliases the health audit and adds severity grouping for maintenance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "wiki_graph",
        "description": (
            "Build or query the wikilink graph for compiled wiki pages. "
            "Actions: build, neighbors, report. Build writes knowledge_base/graph/graph.json and graph.md."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["build", "neighbors", "report"],
                    "description": "Graph operation to run.",
                },
                "page": {
                    "type": "string",
                    "description": "Page title, slug, or relative path for neighbors action.",
                },
                "hops": {
                    "type": "integer",
                    "description": "Number of hops for neighbor traversal (default 1, max 3).",
                },
            },
            "required": [],
        },
    },
]

WIKI_TOOL_NAMES = {tool["name"] for tool in WIKI_TOOLS}


# ── Constants ────────────────────────────────────────────────────────────────

_SYSTEM_FILES = {"index.md", "log.md", "overview.md", "SCHEMA.md"}
_PAGE_TYPES = ("source", "entity", "concept", "synthesis")
_TEXT_EXTS = {
    ".md", ".markdown", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".py",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".yaml", ".yml", ".toml",
    ".xml", ".rst",
}
SUPPORTED_SOURCE_EXTS = set(_TEXT_EXTS) | {".pdf", ".docx"}
_COMMON_ENTITY_WORDS = {
    "A", "An", "And", "Are", "As", "At", "By", "Can", "For", "From", "How",
    "If", "In", "Into", "Is", "It", "Of", "On", "Or", "The", "This", "To",
    "Use", "Uses", "When", "Where", "With", "You",
}
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _slugify(text: str, fallback: str = "page") -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", value)
    value = value.strip("-")
    if not value:
        value = fallback
    return value[:90].strip("-") or fallback


def _safe_workspace_path(path: str, workspace_root: str | None = None) -> Path:
    root = Path(workspace_root or os.getcwd()).resolve()
    raw = Path(path).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    root_str = str(root)
    resolved_str = str(resolved)
    if resolved_str != root_str and not resolved_str.startswith(root_str + os.sep):
        raise PermissionError(
            f"Access denied: path '{path}' resolves outside workspace ({root})"
        )
    return resolved


def _read_source_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from rag import _read_pdf

            return _read_pdf(str(path))
        except Exception as exc:
            raise ValueError(f"Could not extract PDF text: {exc}") from exc

    if ext == ".docx":
        return _read_docx(path)

    if ext == ".doc":
        raise ValueError("Legacy .doc files are not supported. Convert to .docx, .pdf, or .md first.")

    if ext and ext not in _TEXT_EXTS:
        raise ValueError(f"Unsupported source extension: {ext}")

    return path.read_text(encoding="utf-8", errors="replace")


def _read_docx(path: Path) -> str:
    """Extract paragraphs and tables from a .docx using only the standard library."""
    try:
        with zipfile.ZipFile(path) as docx:
            xml_bytes = docx.read("word/document.xml")
    except KeyError as exc:
        raise ValueError("Invalid .docx: missing word/document.xml") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid .docx zip container") from exc

    root = ElementTree.fromstring(xml_bytes)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    blocks = []
    for para in root.iter(f"{ns}p"):
        parts = []
        for node in para.iter():
            if node.tag == f"{ns}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{ns}tab":
                parts.append("\t")
            elif node.tag == f"{ns}br":
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            blocks.append(text)
    return "\n\n".join(blocks)


def _first_heading(text: str) -> str:
    match = re.search(r"^\s{0,3}#{1,3}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_.:/-]+|[\u4e00-\u9fff]", text.lower())
    return [t for t in tokens if len(t) > 1 or re.match(r"[\u4e00-\u9fff]", t)]


def _strip_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text

    raw_meta = text[4:end].strip()
    body = text[text.find("\n", end + 4) + 1:]
    meta: dict[str, Any] = {}
    for line in raw_meta.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value.startswith("[") and raw_value.endswith("]"):
            items = []
            for item in raw_value[1:-1].split(","):
                value = item.strip().strip("'\"")
                if value:
                    items.append(value)
            meta[key] = items
        else:
            meta[key] = raw_value.strip("'\"")
    return meta, body


def _format_frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, list):
            rendered = ", ".join(json.dumps(str(v), ensure_ascii=False) for v in value)
            lines.append(f"{key}: [{rendered}]")
        else:
            value_str = str(value)
            if not value_str or any(ch in value_str for ch in [":", "#", "[", "]", "{", "}"]):
                value_str = json.dumps(value_str, ensure_ascii=False)
            lines.append(f"{key}: {value_str}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _excerpt(text: str, max_chars: int = 650) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _quote_block(text: str, max_chars: int = 700) -> str:
    body = _excerpt(text, max_chars=max_chars)
    return "\n".join(f"> {line}" if line.strip() else ">" for line in body.splitlines())


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


# ── Wiki System ──────────────────────────────────────────────────────────────

class WikiSystem:
    """Local file-backed LLM Wiki with deterministic ingest/search/graph."""

    def __init__(
        self,
        knowledge_base_dir: str = config.KNOWLEDGE_BASE_DIR,
        wiki_root: str | None = None,
        raw_sources_root: str | None = None,
        graph_root: str | None = None,
    ):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        default_kb = Path(config.KNOWLEDGE_BASE_DIR)
        use_config_paths = self.knowledge_base_dir == default_kb
        default_wiki_root = getattr(config, "WIKI_ROOT_DIR", default_kb / "wiki")
        default_raw_root = getattr(config, "WIKI_RAW_SOURCES_DIR", default_kb / "raw" / "sources")
        default_graph_root = getattr(config, "WIKI_GRAPH_DIR", default_kb / "graph")
        self.wiki_root = Path(
            wiki_root or (default_wiki_root if use_config_paths else self.knowledge_base_dir / "wiki")
        )
        self.raw_sources_root = Path(
            raw_sources_root or (default_raw_root if use_config_paths else self.knowledge_base_dir / "raw" / "sources")
        )
        self.graph_root = Path(
            graph_root or (default_graph_root if use_config_paths else self.knowledge_base_dir / "graph")
        )
        self._lock = threading.RLock()

    # ── Public API ───────────────────────────────────────────────────────

    def init(self, force_rebuild: bool = False) -> dict[str, Any]:
        with self._lock:
            self._ensure_dirs()
            created = []
            created.extend(self._write_initial_files())
            if force_rebuild:
                self._rebuild_index_and_overview()
                graph = self.build_graph()
            else:
                graph = {}
            return {
                "status": "ok",
                "wiki_root": str(self.wiki_root),
                "raw_sources_root": str(self.raw_sources_root),
                "graph_root": str(self.graph_root),
                "created": created,
                "rebuilt_graph": graph,
            }

    def ingest(
        self,
        path: str | None = None,
        content: str | None = None,
        title: str | None = None,
        source_type: str = "document",
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_initialized()

            source_path: Path | None = None
            source_name = title or "pasted-content"
            ext = ".md"

            if path:
                source_path = _safe_workspace_path(path, workspace_root)
                if not source_path.is_file():
                    raise FileNotFoundError(f"Source file not found: {source_path}")
                content = _read_source_file(source_path)
                source_name = title or _first_heading(content) or source_path.stem
                ext = source_path.suffix.lower() or ".md"
            elif content:
                source_name = title or _first_heading(content) or "Pasted Content"
                ext = ".md"
                source_type = source_type or "pasted"
            else:
                raise ValueError("wiki_ingest requires either 'path' or 'content'")

            text = (content or "").strip()
            if not text:
                raise ValueError("Source content is empty after extraction")

            title = title or source_name
            source_type = source_type or "document"
            source_hash = _sha256(text)
            source_slug = f"{_slugify(title)}-{source_hash[:8]}"
            raw_ext = ext if ext in _TEXT_EXTS else ".md"
            raw_filename = f"{_today()}-{_slugify(source_name)}-{source_hash[:12]}{raw_ext}"
            raw_path = self.raw_sources_root / raw_filename
            if not raw_path.exists():
                raw_path.write_text(text + "\n", encoding="utf-8")

            analysis = self._analyze_source(title, text)
            written = []
            source_rel = self._write_source_page(
                title=title,
                source_slug=source_slug,
                source_type=source_type,
                source_hash=source_hash,
                raw_path=raw_path,
                source_path=source_path,
                analysis=analysis,
            )
            written.append(source_rel)

            concept_links = []
            entity_links = []
            for concept in analysis["concepts"]:
                concept_links.append(self._upsert_topic_page(
                    kind="concept",
                    name=concept["name"],
                    source_slug=source_slug,
                    source_title=title,
                    excerpt_text=concept["excerpt"],
                    related=[source_slug] + [e["slug"] for e in analysis["entities"][:4]],
                ))
            for entity in analysis["entities"]:
                entity_links.append(self._upsert_topic_page(
                    kind="entity",
                    name=entity["name"],
                    source_slug=source_slug,
                    source_title=title,
                    excerpt_text=entity["excerpt"],
                    related=[source_slug] + [c["slug"] for c in analysis["concepts"][:4]],
                ))

            written.extend(concept_links)
            written.extend(entity_links)
            self._rebuild_index_and_overview()
            graph = self.build_graph()
            self._append_log(
                "ingest",
                f"{title} -> [[{source_slug}]], pages={len(written)}, sha={source_hash[:12]}",
            )

            return {
                "status": "ingested",
                "source": {
                    "title": title,
                    "type": source_type,
                    "sha256": source_hash,
                    "raw_path": str(raw_path),
                    "wiki_page": source_rel,
                },
                "stage_1_analysis": {
                    "summary": analysis["summary"],
                    "concepts": [c["name"] for c in analysis["concepts"]],
                    "entities": [e["name"] for e in analysis["entities"]],
                },
                "stage_2_writes": {
                    "pages": written,
                    "index": str(self.wiki_root / "index.md"),
                    "log": str(self.wiki_root / "log.md"),
                    "graph": graph,
                },
            }

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_system_pages: bool = False,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        q = query.strip()
        if not q:
            return {"query": query, "results": []}

        query_tokens = _tokenize(q)
        exact = q.lower()
        results = []
        for page in self._load_pages(include_system=include_system_pages):
            text_l = page["content"].lower()
            title_l = page["title"].lower()
            path_l = page["rel_path"].lower()
            score = 0.0
            if exact in text_l:
                score += 8.0
            if exact in title_l or exact in path_l:
                score += 12.0
            for token in query_tokens:
                if token in title_l:
                    score += 4.0
                if token in path_l:
                    score += 2.0
                score += min(text_l.count(token), 8) * 0.8
            if score <= 0:
                continue
            results.append({
                "title": page["title"],
                "type": page["type"],
                "path": page["rel_path"],
                "slug": page["slug"],
                "score": round(score, 3),
                "excerpt": self._best_excerpt(page["content"], query_tokens, exact),
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return {"query": query, "results": results[: max(1, top_k)]}

    def read(self, page: str, max_chars: int = 20000) -> dict[str, Any]:
        self._ensure_initialized()
        resolved = self._resolve_page(page)
        if not resolved:
            return {"error": f"Wiki page not found: {page}"}
        text = resolved.read_text(encoding="utf-8", errors="replace")
        meta, _body = _strip_frontmatter(text)
        rel = resolved.relative_to(self.wiki_root).as_posix()
        truncated = text[:max_chars]
        return {
            "path": rel,
            "title": meta.get("title") or resolved.stem,
            "type": meta.get("type", "page"),
            "total_chars": len(text),
            "content": truncated,
            "truncated": len(text) > len(truncated),
        }

    def health(self, init_if_missing: bool = True) -> dict[str, Any]:
        with self._lock:
            if init_if_missing:
                self._ensure_initialized()

            required = [
                self.wiki_root,
                self.raw_sources_root,
                self.graph_root,
                self.wiki_root / "index.md",
                self.wiki_root / "log.md",
                self.wiki_root / "overview.md",
                self.wiki_root / "SCHEMA.md",
            ]
            missing = [str(path) for path in required if not path.exists()]
            pages = self._load_pages(include_system=False) if self.wiki_root.exists() else []
            type_counts = Counter(page["type"] for page in pages)
            links = self._collect_links(pages)
            incoming = Counter(edge["target"] for edge in links["edges"])
            orphan_pages = [
                page["rel_path"] for page in pages
                if incoming[page["rel_path"]] == 0 and page["type"] not in {"source"}
            ]
            index_missing = self._index_missing_pages(pages)
            raw_count = len(list(self.raw_sources_root.glob("*"))) if self.raw_sources_root.exists() else 0

            issues = []
            if missing:
                issues.append({"severity": "error", "check": "required_paths", "items": missing})
            if links["broken"]:
                issues.append({"severity": "error", "check": "broken_links", "items": links["broken"]})
            if index_missing:
                issues.append({"severity": "warning", "check": "index_coverage", "items": index_missing})
            if orphan_pages:
                issues.append({"severity": "info", "check": "orphan_pages", "items": orphan_pages})

            return {
                "status": "ok" if not [i for i in issues if i["severity"] == "error"] else "needs_attention",
                "initialized": not missing,
                "wiki_root": str(self.wiki_root),
                "page_count": len(pages),
                "raw_source_count": raw_count,
                "page_types": dict(type_counts),
                "link_count": len(links["edges"]),
                "broken_link_count": len(links["broken"]),
                "orphan_count": len(orphan_pages),
                "index_missing_count": len(index_missing),
                "issues": issues,
            }

    def lint(self) -> dict[str, Any]:
        report = self.health(init_if_missing=True)
        grouped = defaultdict(list)
        for issue in report.get("issues", []):
            grouped[issue["severity"]].append(issue)
        report["lint"] = {
            "errors": grouped.get("error", []),
            "warnings": grouped.get("warning", []),
            "info": grouped.get("info", []),
        }
        return report

    def build_graph(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_initialized()
            pages = self._load_pages(include_system=False)
            links = self._collect_links(pages)
            nodes = [
                {
                    "id": page["rel_path"],
                    "slug": page["slug"],
                    "title": page["title"],
                    "type": page["type"],
                    "path": page["rel_path"],
                }
                for page in pages
            ]
            graph = {
                "generated_at": _now_iso(),
                "nodes": nodes,
                "edges": links["edges"],
                "broken_links": links["broken"],
            }
            self.graph_root.mkdir(parents=True, exist_ok=True)
            graph_json = self.graph_root / "graph.json"
            graph_md = self.graph_root / "graph.md"
            graph_json.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
            graph_md.write_text(self._render_graph_report(graph), encoding="utf-8")
            return {
                "status": "built",
                "nodes": len(nodes),
                "edges": len(links["edges"]),
                "broken_links": len(links["broken"]),
                "graph_json": str(graph_json),
                "graph_report": str(graph_md),
            }

    def graph_neighbors(self, page: str, hops: int = 1) -> dict[str, Any]:
        self._ensure_initialized()
        hops = max(1, min(int(hops or 1), 3))
        graph_path = self.graph_root / "graph.json"
        if not graph_path.exists():
            self.build_graph()
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        start_path = self._resolve_page(page)
        if not start_path:
            return {"error": f"Wiki page not found: {page}"}
        start_id = start_path.relative_to(self.wiki_root).as_posix()

        adjacency = defaultdict(set)
        for edge in graph.get("edges", []):
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])

        nodes_by_id = {node["id"]: node for node in graph.get("nodes", [])}
        seen = {start_id}
        queue = deque([(start_id, 0)])
        neighbors = []
        while queue:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for nxt in sorted(adjacency[current]):
                if nxt in seen:
                    continue
                seen.add(nxt)
                node = nodes_by_id.get(nxt, {"id": nxt, "title": nxt, "type": "page"})
                neighbors.append({**node, "distance": depth + 1})
                queue.append((nxt, depth + 1))

        return {
            "page": start_id,
            "hops": hops,
            "neighbors": neighbors,
            "count": len(neighbors),
        }

    # ── Initialization ───────────────────────────────────────────────────

    def _ensure_dirs(self):
        for path in [
            self.knowledge_base_dir,
            self.raw_sources_root,
            self.graph_root,
            self.wiki_root,
            self.wiki_root / "sources",
            self.wiki_root / "entities",
            self.wiki_root / "concepts",
            self.wiki_root / "syntheses",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _ensure_initialized(self):
        self._ensure_dirs()
        self._write_initial_files()

    def _write_initial_files(self) -> list[str]:
        created = []
        files = {
            self.wiki_root / "SCHEMA.md": self._initial_schema(),
            self.wiki_root / "log.md": "# Wiki Log\n\n| Time | Operation | Detail |\n| --- | --- | --- |\n",
            self.wiki_root / "overview.md": "# Wiki Overview\n\nNo pages ingested yet.\n",
            self.wiki_root / "index.md": "# Wiki Index\n\nNo pages ingested yet.\n",
        }
        for path, content in files.items():
            if not path.exists():
                path.write_text(content, encoding="utf-8")
                created.append(str(path))
        return created

    @staticmethod
    def _initial_schema() -> str:
        return """# LLM Wiki Schema

## Directory Model

- `raw/sources/`: immutable source snapshots.
- `wiki/sources/`: source pages that summarize one raw source.
- `wiki/entities/`: people, organizations, products, APIs, systems, files, or named things.
- `wiki/concepts/`: technical principles, patterns, decisions, and reusable explanations.
- `wiki/syntheses/`: durable cross-source analyses.

## Page Types

Allowed `type` frontmatter values: `source`, `entity`, `concept`, `synthesis`.

## Tags

Start with these tags: `source`, `entity`, `concept`, `synthesis`, `document`, `note`, `code`, `url`, `meeting`, `pasted`.

## Linking Rules

- Use `[[page-slug]]` wikilinks for durable references.
- Source snapshots are append-only. Update wiki pages, not raw snapshots.
- Prefer updating existing entity/concept pages over creating near-duplicates.
"""

    # ── Ingest internals ─────────────────────────────────────────────────

    def _analyze_source(self, title: str, text: str) -> dict[str, Any]:
        summary = self._summarize_text(text)
        concepts = []
        for name, excerpt_text in self._extract_concept_candidates(title, text):
            slug = _slugify(name)
            concepts.append({"name": name, "slug": slug, "excerpt": excerpt_text})
            if len(concepts) >= 10:
                break

        if not concepts:
            concepts = [{"name": title, "slug": _slugify(title), "excerpt": _excerpt(text)}]

        concept_slugs = {c["slug"] for c in concepts}
        entities = [
            entity for entity in self._extract_entities(text)
            if entity["slug"] not in concept_slugs
        ]

        return {"summary": summary, "concepts": concepts, "entities": entities}

    @staticmethod
    def _summarize_text(text: str) -> list[str]:
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("|") or set(line) <= {"-", "=", "#"}:
                continue
            line = re.sub(r"^#{1,6}\s+", "", line)
            lines.append(_excerpt(line, 180))
            if len(lines) >= 4:
                break
        return lines or [_excerpt(text, 220)]

    def _extract_concept_candidates(self, title: str, text: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen = set()

        def add(name: str, excerpt_text: str):
            clean = re.sub(r"\s+", " ", name.strip(" #`"))
            if len(clean) < 2:
                return
            slug = _slugify(clean)
            if slug in seen:
                return
            seen.add(slug)
            candidates.append((clean, excerpt_text))

        add(title, _excerpt(text))
        heading_matches = list(re.finditer(r"^\s{0,3}(#{1,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
        for i, match in enumerate(heading_matches):
            heading = match.group(2).strip()
            start = match.end()
            end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else min(len(text), start + 1600)
            add(heading, _excerpt(text[start:end]))

        return candidates

    @staticmethod
    def _extract_entities(text: str) -> list[dict[str, str]]:
        counts: Counter[str] = Counter()
        patterns = [
            r"\b[A-Z][A-Za-z0-9]+(?:[ -][A-Z][A-Za-z0-9]+){0,3}\b",
            r"\b[A-Z]{2,}(?:[-_][A-Z0-9]+)*\b",
            r"`([A-Za-z_][A-Za-z0-9_./:-]{1,80})`",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1) if match.lastindex else match.group(0)
                name = name.strip("`.,:;()[]{}")
                if not name or name in _COMMON_ENTITY_WORDS or len(name) < 2:
                    continue
                if re.fullmatch(r"\d+", name):
                    continue
                counts[name] += 1

        entities = []
        for name, _count in counts.most_common(16):
            excerpt_text = _find_context(text, name)
            entities.append({"name": name, "slug": _slugify(name), "excerpt": excerpt_text})
        return entities

    def _write_source_page(
        self,
        title: str,
        source_slug: str,
        source_type: str,
        source_hash: str,
        raw_path: Path,
        source_path: Path | None,
        analysis: dict[str, Any],
    ) -> str:
        rel_path = Path("sources") / f"{source_slug}.md"
        page_path = self.wiki_root / rel_path
        created = _now_iso()
        if page_path.exists():
            existing_meta, _body = _strip_frontmatter(page_path.read_text(encoding="utf-8", errors="replace"))
            created = existing_meta.get("created", created)

        concept_lines = "\n".join(f"- [[{c['slug']}]]" for c in analysis["concepts"]) or "- None"
        entity_lines = "\n".join(f"- [[{e['slug']}]]" for e in analysis["entities"]) or "- None"
        summary_lines = "\n".join(f"- {line}" for line in analysis["summary"])
        meta = {
            "title": f"Source: {title}",
            "created": created,
            "updated": _now_iso(),
            "type": "source",
            "tags": ["source", source_type],
            "source_sha256": source_hash,
            "raw_path": _display_path(raw_path, self.knowledge_base_dir),
        }
        if source_path:
            meta["original_path"] = str(source_path)

        body = f"""# Source: {title}

## Summary

{summary_lines}

## Key Concepts

{concept_lines}

## Key Entities

{entity_lines}

## Source Metadata

- Raw snapshot: `{_display_path(raw_path, self.knowledge_base_dir)}`
- SHA256: `{source_hash}`
- Source type: `{source_type}`
"""
        page_path.write_text(_format_frontmatter(meta) + body, encoding="utf-8")
        return rel_path.as_posix()

    def _upsert_topic_page(
        self,
        kind: str,
        name: str,
        source_slug: str,
        source_title: str,
        excerpt_text: str,
        related: list[str],
    ) -> str:
        slug = _slugify(name)
        folder = "concepts" if kind == "concept" else "entities"
        rel_path = Path(folder) / f"{slug}.md"
        page_path = self.wiki_root / rel_path
        source_link = f"[[{source_slug}]]"
        now = _now_iso()

        if page_path.exists():
            old_text = page_path.read_text(encoding="utf-8", errors="replace")
            meta, body = _strip_frontmatter(old_text)
            sources = list(meta.get("sources", [])) if isinstance(meta.get("sources"), list) else []
            if source_slug not in sources:
                sources.append(source_slug)
            meta.update({"updated": now, "sources": sources})
            if source_link not in body:
                body = body.rstrip() + (
                    f"\n\n### From {source_link} on {_today()}\n\n"
                    f"Related source title: {source_title}\n\n{_quote_block(excerpt_text)}\n"
                )
            page_path.write_text(_format_frontmatter(meta) + body.strip() + "\n", encoding="utf-8")
            return rel_path.as_posix()

        related_unique = []
        for item in related:
            if item and item not in related_unique and item != slug:
                related_unique.append(item)
        related_lines = "\n".join(f"- [[{item}]]" for item in related_unique[:8]) or f"- {source_link}"
        meta = {
            "title": name,
            "created": now,
            "updated": now,
            "type": kind,
            "tags": [kind],
            "sources": [source_slug],
        }
        body = f"""# {name}

## Summary

This {kind} page was compiled from {source_link}. Update it as more sources mention the same topic.

## Evidence

### From {source_link} on {_today()}

Related source title: {source_title}

{_quote_block(excerpt_text)}

## Related

{related_lines}
"""
        page_path.write_text(_format_frontmatter(meta) + body, encoding="utf-8")
        return rel_path.as_posix()

    # ── Page loading and search helpers ──────────────────────────────────

    def _load_pages(self, include_system: bool = False) -> list[dict[str, Any]]:
        pages = []
        if not self.wiki_root.exists():
            return pages
        for path in sorted(self.wiki_root.rglob("*.md")):
            rel = path.relative_to(self.wiki_root).as_posix()
            if not include_system and path.name in _SYSTEM_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            meta, body = _strip_frontmatter(text)
            title = meta.get("title") or _title_from_body(body) or path.stem
            pages.append({
                "path": path,
                "rel_path": rel,
                "slug": path.stem,
                "title": title,
                "type": meta.get("type", "page"),
                "meta": meta,
                "body": body,
                "content": text,
            })
        return pages

    def _resolve_page(self, page: str) -> Path | None:
        query = (page or "").strip()
        if not query:
            return None

        rel_candidate = (self.wiki_root / query).resolve()
        try:
            rel_candidate.relative_to(self.wiki_root.resolve())
        except ValueError:
            rel_candidate = self.wiki_root / "__invalid__"
        if rel_candidate.is_file():
            return rel_candidate
        if not query.endswith(".md"):
            md_candidate = (self.wiki_root / f"{query}.md").resolve()
            try:
                md_candidate.relative_to(self.wiki_root.resolve())
            except ValueError:
                md_candidate = self.wiki_root / "__invalid__"
            if md_candidate.is_file():
                return md_candidate

        query_l = query.lower().removesuffix(".md")
        query_slug = _slugify(query)
        for page_info in self._load_pages(include_system=True):
            rel_no_ext = page_info["rel_path"][:-3].lower() if page_info["rel_path"].endswith(".md") else page_info["rel_path"].lower()
            if query_l in {page_info["slug"].lower(), rel_no_ext}:
                return page_info["path"]
            if query_slug == page_info["slug"].lower():
                return page_info["path"]
            if query_l == str(page_info["title"]).lower():
                return page_info["path"]
        return None

    @staticmethod
    def _best_excerpt(content: str, query_tokens: list[str], exact: str) -> str:
        body = _strip_frontmatter(content)[1]
        body_l = body.lower()
        positions = []
        if exact:
            idx = body_l.find(exact)
            if idx >= 0:
                positions.append(idx)
        for token in query_tokens:
            idx = body_l.find(token)
            if idx >= 0:
                positions.append(idx)
        if not positions:
            return _excerpt(body, 420)
        pos = min(positions)
        start = max(0, pos - 160)
        end = min(len(body), pos + 360)
        return _excerpt(body[start:end], 520)

    # ── Index, health, and graph internals ───────────────────────────────

    def _rebuild_index_and_overview(self):
        pages = self._load_pages(include_system=False)
        groups: dict[str, list[dict[str, Any]]] = {kind: [] for kind in _PAGE_TYPES}
        groups["other"] = []
        for page in pages:
            groups.get(page["type"], groups["other"]).append(page)

        lines = [
            "# Wiki Index",
            "",
            f"Last updated: {_now_iso()}",
            "",
        ]
        section_names = {
            "source": "Sources",
            "entity": "Entities",
            "concept": "Concepts",
            "synthesis": "Syntheses",
            "other": "Other Pages",
        }
        for kind in [*_PAGE_TYPES, "other"]:
            items = sorted(groups.get(kind, []), key=lambda p: str(p["title"]).lower())
            lines.extend([f"## {section_names[kind]}", ""])
            if not items:
                lines.extend(["- None", ""])
                continue
            for page in items:
                lines.append(f"- [[{page['slug']}]] - {page['title']} (`{page['rel_path']}`)")
            lines.append("")

        (self.wiki_root / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        self._write_overview(pages)

    def _write_overview(self, pages: list[dict[str, Any]]):
        counts = Counter(page["type"] for page in pages)
        recent = sorted(
            pages,
            key=lambda p: str(p["meta"].get("updated", "")),
            reverse=True,
        )[:10]
        count_lines = "\n".join(f"- {kind}: {counts.get(kind, 0)}" for kind in [*_PAGE_TYPES, "page"])
        recent_lines = "\n".join(
            f"- [[{page['slug']}]] - {page['title']} (`{page['type']}`)"
            for page in recent
        ) or "- None"
        body = f"""# Wiki Overview

Last updated: {_now_iso()}

## Counts

{count_lines}

## Recently Updated

{recent_lines}
"""
        (self.wiki_root / "overview.md").write_text(body, encoding="utf-8")

    def _append_log(self, operation: str, detail: str):
        log_path = self.wiki_root / "log.md"
        if not log_path.exists():
            log_path.write_text("# Wiki Log\n\n| Time | Operation | Detail |\n| --- | --- | --- |\n", encoding="utf-8")
        safe_detail = detail.replace("|", "\\|").replace("\n", " ")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"| {_now_iso()} | {operation} | {safe_detail} |\n")

    def _page_lookup(self, pages: list[dict[str, Any]]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for page in pages:
            rel = page["rel_path"]
            rel_no_ext = rel[:-3] if rel.endswith(".md") else rel
            keys = {
                page["slug"].lower(),
                rel.lower(),
                rel_no_ext.lower(),
                str(page["title"]).lower(),
                _slugify(str(page["title"])).lower(),
            }
            for key in keys:
                lookup.setdefault(key, rel)
        return lookup

    def _collect_links(self, pages: list[dict[str, Any]]) -> dict[str, Any]:
        lookup = self._page_lookup(pages)
        edges = []
        broken = []
        seen_edges = set()
        for page in pages:
            for match in _WIKILINK_RE.finditer(page["content"]):
                target_raw = match.group(1).strip()
                target_key = target_raw.lower()
                target = lookup.get(target_key) or lookup.get(_slugify(target_raw).lower())
                if not target:
                    broken.append({
                        "source": page["rel_path"],
                        "target": target_raw,
                    })
                    continue
                if target == page["rel_path"]:
                    continue
                edge_key = (page["rel_path"], target)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append({
                    "source": page["rel_path"],
                    "target": target,
                    "type": "wikilink",
                })
        return {"edges": edges, "broken": broken}

    def _index_missing_pages(self, pages: list[dict[str, Any]]) -> list[str]:
        index_path = self.wiki_root / "index.md"
        if not index_path.exists():
            return [page["rel_path"] for page in pages]
        index_text = index_path.read_text(encoding="utf-8", errors="replace")
        missing = []
        for page in pages:
            if page["rel_path"] not in index_text and f"[[{page['slug']}]]" not in index_text:
                missing.append(page["rel_path"])
        return missing

    @staticmethod
    def _render_graph_report(graph: dict[str, Any]) -> str:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        broken = graph.get("broken_links", [])
        by_type = Counter(node.get("type", "page") for node in nodes)
        type_lines = "\n".join(f"- {kind}: {count}" for kind, count in sorted(by_type.items())) or "- None"
        edge_lines = "\n".join(
            f"- `{edge['source']}` -> `{edge['target']}`"
            for edge in edges[:200]
        ) or "- None"
        broken_lines = "\n".join(
            f"- `{item['source']}` -> `[[{item['target']}]]`"
            for item in broken[:100]
        ) or "- None"
        return f"""# Wiki Graph Report

Generated: {graph.get("generated_at", _now_iso())}

## Summary

- Nodes: {len(nodes)}
- Edges: {len(edges)}
- Broken links: {len(broken)}

## Node Types

{type_lines}

## Edges

{edge_lines}

## Broken Links

{broken_lines}
"""


def _title_from_body(body: str) -> str:
    match = re.search(r"^\s{0,3}#\s+(.+?)\s*$", body, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _find_context(text: str, needle: str, radius: int = 280) -> str:
    pos = text.find(needle)
    if pos < 0:
        return _excerpt(text, radius * 2)
    start = max(0, pos - radius)
    end = min(len(text), pos + len(needle) + radius)
    return _excerpt(text[start:end], radius * 2)


# ── Tool Dispatch ───────────────────────────────────────────────────────────

def handle_wiki_tool(
    name: str,
    tool_input: dict,
    wiki: WikiSystem,
    workspace_root: str | None = None,
) -> str:
    """Dispatch an LLM Wiki tool call and return a JSON result string."""
    try:
        if name == "wiki_init":
            result = wiki.init(force_rebuild=bool(tool_input.get("force_rebuild", False)))
        elif name == "wiki_ingest":
            result = wiki.ingest(
                path=tool_input.get("path"),
                content=tool_input.get("content"),
                title=tool_input.get("title"),
                source_type=tool_input.get("source_type", "document"),
                workspace_root=workspace_root,
            )
        elif name == "wiki_search":
            result = wiki.search(
                query=tool_input.get("query", ""),
                top_k=int(tool_input.get("top_k", 5) or 5),
                include_system_pages=bool(tool_input.get("include_system_pages", False)),
            )
        elif name == "wiki_read":
            result = wiki.read(
                page=tool_input["page"],
                max_chars=int(tool_input.get("max_chars", 20000) or 20000),
            )
        elif name == "wiki_health":
            result = wiki.health(
                init_if_missing=tool_input.get("init_if_missing", True) is not False
            )
        elif name == "wiki_lint":
            result = wiki.lint()
        elif name == "wiki_graph":
            action = tool_input.get("action", "build")
            if action == "neighbors":
                result = wiki.graph_neighbors(
                    page=tool_input.get("page", ""),
                    hops=int(tool_input.get("hops", 1) or 1),
                )
            elif action == "report":
                result = wiki.build_graph()
                report_path = wiki.graph_root / "graph.md"
                result["report_content"] = report_path.read_text(
                    encoding="utf-8", errors="replace"
                )[:20000]
            else:
                result = wiki.build_graph()
        else:
            result = {"error": f"Unknown wiki tool: {name}"}
    except Exception as exc:
        result = {"error": f"{name} failed: {exc}"}
    return json.dumps(result, ensure_ascii=False)
