#!/usr/bin/env python3
"""Command-line helper for the local LLM Wiki."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wiki_system import SUPPORTED_SOURCE_EXTS, WikiSystem


def _print_json(data: Any):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SOURCE_EXTS else []
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    ignored_parts = {".git", ".rag_cache", "__pycache__"}
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if any(part in ignored_parts for part in item.parts):
            continue
        if item.suffix.lower() in SUPPORTED_SOURCE_EXTS:
            files.append(item)
    return files


def cmd_init(args) -> int:
    wiki = WikiSystem()
    _print_json(wiki.init(force_rebuild=args.rebuild))
    return 0


def cmd_ingest(args) -> int:
    wiki = WikiSystem()
    source = Path(args.path).expanduser()
    files = _source_files(source)
    if not files:
        _print_json({
            "status": "no_supported_files",
            "path": str(source),
            "supported_extensions": sorted(SUPPORTED_SOURCE_EXTS),
        })
        return 1

    results = []
    errors = []
    for file_path in files:
        try:
            workspace_root = file_path.parent if file_path.is_file() else Path.cwd()
            result = wiki.ingest(
                path=str(file_path),
                title=args.title if len(files) == 1 else None,
                source_type=args.source_type,
                workspace_root=str(workspace_root),
            )
            results.append({
                "path": str(file_path),
                "wiki_page": result["source"]["wiki_page"],
                "pages_written": len(result["stage_2_writes"]["pages"]),
            })
        except Exception as exc:
            errors.append({"path": str(file_path), "error": str(exc)})

    _print_json({
        "status": "completed" if not errors else "completed_with_errors",
        "input": str(source),
        "ingested": len(results),
        "errors": errors,
        "results": results,
        "health": wiki.health(),
    })
    return 0 if not errors else 2


def cmd_search(args) -> int:
    wiki = WikiSystem()
    _print_json(wiki.search(args.query, top_k=args.top_k, include_system_pages=args.system))
    return 0


def cmd_read(args) -> int:
    wiki = WikiSystem()
    _print_json(wiki.read(args.page, max_chars=args.max_chars))
    return 0


def cmd_health(_args) -> int:
    wiki = WikiSystem()
    _print_json(wiki.health())
    return 0


def cmd_graph(args) -> int:
    wiki = WikiSystem()
    if args.action == "neighbors":
        _print_json(wiki.graph_neighbors(args.page, hops=args.hops))
    else:
        result = wiki.build_graph()
        if args.action == "report":
            report_path = Path(result["graph_report"])
            result["report_content"] = report_path.read_text(encoding="utf-8", errors="replace")
        _print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the local LLM Wiki")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize wiki directories")
    init.add_argument("--rebuild", action="store_true", help="rebuild index and graph")
    init.set_defaults(func=cmd_init)

    ingest = sub.add_parser("ingest", help="ingest a supported file or directory")
    ingest.add_argument("path", help="file or directory to ingest")
    ingest.add_argument("--title", help="title override for a single file")
    ingest.add_argument("--source-type", default="document", help="document, note, code, url, meeting, pasted")
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="search compiled wiki pages")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--system", action="store_true", help="include index/log/schema/overview")
    search.set_defaults(func=cmd_search)

    read = sub.add_parser("read", help="read a wiki page")
    read.add_argument("page", help="title, slug, or relative path")
    read.add_argument("--max-chars", type=int, default=20000)
    read.set_defaults(func=cmd_read)

    health = sub.add_parser("health", help="run deterministic health checks")
    health.set_defaults(func=cmd_health)

    graph = sub.add_parser("graph", help="build or query the wiki graph")
    graph.add_argument("--action", choices=["build", "report", "neighbors"], default="build")
    graph.add_argument("--page", default="", help="page for neighbors action")
    graph.add_argument("--hops", type=int, default=1)
    graph.set_defaults(func=cmd_graph)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
