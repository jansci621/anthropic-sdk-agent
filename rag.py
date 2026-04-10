"""RAG system: document loading, chunking, embedding, and retrieval.

Chunk strategy (V3):
  1. Parse document into typed blocks (header / table / list_item / text)
  2. Headers start a new chunk — cross-section mixing is forbidden
  3. Tables: each chunk carries the full header row(s)
  4. List items: lead sentence (前导句) is prepended to every item chunk
  5. Overlap aligned to sentence boundaries; CJK-aware char→token ratio
  6. Hybrid retrieval: dense (FAISS) + sparse (BM25) with RRF fusion
  7. Embedding cache keyed by content hash — fast restarts
"""

import hashlib
import json
import os
import re

import numpy as np

import config

# Apply HuggingFace mirror before any HF-related imports
if config.HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT


# ── Tool Definition ─────────────────────────────────────────────────────────

RAG_TOOL = {
    "name": "search_documents",
    "description": (
        "Search the knowledge base for relevant documents and information. "
        "Use this when the user asks questions that might be answered by "
        "reference materials, documentation, or other files in the knowledge base. "
        "Returns the most relevant text chunks with source file information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query — describe what information you need",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 3)",
            },
        },
        "required": ["query"],
    },
}


# ── File readers ─────────────────────────────────────────────────────────────

def _read_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf(filepath: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(filepath)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print("[RAG] Warning: pypdf not installed. Install with: pip install pypdf")
        return ""


def _read_json(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False, indent=2)


def _read_csv(filepath: str) -> str:
    import csv
    rows: list[str] = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            rows.append(" | ".join(row))
    return "\n".join(rows)


_FILE_READERS: dict[str, object] = {
    ".txt": _read_text,
    ".md":  _read_text,
    ".pdf": _read_pdf,
    ".json": _read_json,
    ".csv": _read_csv,
}


# ── Structure detection helpers ───────────────────────────────────────────────

# Markdown headers and Chinese section numbering systems
_HEADER_RE = re.compile(
    r'^(?:'
    r'#{1,6}\s'                                    # ## Markdown header
    r'|第[一二三四五六七八九十百\d]+[章条节篇]'      # 第X章/条/节
    r'|\d+\.\s+[\u4e00-\u9fa5A-Z]'                # 1. Title
    r'|\d+\.\d+\s'                                 # 1.1 subsection
    r'|（[一二三四五六七八九十]+）[\u4e00-\u9fa5]'  # （一）中文
    r')'
)

# List items: （1） ① 1. 1、 (a)
_LIST_ITEM_RE = re.compile(
    r'^(?:（\d+）|[①②③④⑤⑥⑦⑧⑨⑩]|\d+[\.、]\s|\([a-z]\)\s)'
)

# Lead sentence ending with colon or Chinese colon, or containing 以下/下列/包括 etc.
_LIST_INTRO_RE = re.compile(
    r'(?:以下|下列|如下|包括|包含|范围|情形|情况)[^：:]*[：:]?\s*$'
    r'|[：:]\s*$'
)

# Markdown table row
_TABLE_ROW_RE = re.compile(r'^\|.+\|')
# Markdown table separator row
_TABLE_SEP_RE = re.compile(r'^\|[\s\-\|:]+\|')


def _is_header(line: str) -> bool:
    return bool(_HEADER_RE.match(line.strip()))


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(line.strip()))


def _is_list_intro(line: str) -> bool:
    return bool(_LIST_INTRO_RE.search(line.strip()))


def _is_table_row(line: str) -> bool:
    return bool(_TABLE_ROW_RE.match(line.strip()))


# ── RAG System ──────────────────────────────────────────────────────────────

class RAGSystem:
    """Document retrieval using local sentence embeddings and FAISS + BM25."""

    _CACHE_SUBDIR = ".rag_cache"

    def __init__(
        self,
        knowledge_base_dir: str = config.KNOWLEDGE_BASE_DIR,
        embedding_model: str = config.EMBEDDING_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.TOP_K_RESULTS,
    ):
        self.knowledge_base_dir = knowledge_base_dir
        self.chunk_size = chunk_size        # in tokens
        self.chunk_overlap = chunk_overlap  # in tokens
        self.top_k = top_k
        self._cache_dir = os.path.join(knowledge_base_dir, self._CACHE_SUBDIR)

        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.index = None   # FAISS or None
        self._bm25 = None   # rank_bm25.BM25Okapi or None

        print(f"[RAG] Loading documents from {knowledge_base_dir} ...")
        self.model = self._load_embedding_model(embedding_model)
        self._load_and_index()
        print(f"[RAG] Indexed {len(self.chunks)} chunks from knowledge base.")

    # ── Public API ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None,
               extra_queries: list[str] | None = None) -> list[dict]:
        """Hybrid dense + BM25 retrieval with RRF fusion.

        Args:
            query:         primary query string
            top_k:         number of results to return
            extra_queries: additional query variants for multi-query retrieval
                           (e.g. generated by LLM query expansion or HyDE)
        """
        if not self.chunks or self.embeddings is None:
            return [{"text": "No documents indexed.", "source": "system"}]

        k = min(top_k or self.top_k, len(self.chunks))
        # Larger candidate pool improves recall before final ranking
        candidates_k = min(k * 5, len(self.chunks))

        all_queries = [query] + (extra_queries or [])

        # ── Dense retrieval (multi-query) ──────────────────────────────
        dense_scores: dict[int, float] = {}
        query_vecs = np.array(
            self.model.encode(all_queries, normalize_embeddings=True), dtype=np.float32
        )
        for q_vec in query_vecs:
            q_vec = q_vec.reshape(1, -1)
            if self.index is not None:
                scores, indices = self.index.search(q_vec, candidates_k)
                for s, i in zip(scores[0], indices[0]):
                    if i >= 0:
                        # Keep the best score across query variants
                        dense_scores[int(i)] = max(dense_scores.get(int(i), 0.0), float(s))
            else:
                raw = np.dot(self.embeddings, q_vec.T).flatten()
                for i in np.argsort(raw)[::-1][:candidates_k]:
                    dense_scores[int(i)] = max(dense_scores.get(int(i), 0.0), float(raw[i]))

        # ── Sparse BM25 retrieval (multi-query) ───────────────────────
        bm25_scores: dict[int, float] = {}
        if self._bm25 is not None:
            for q in all_queries:
                raw_bm25 = self._bm25.get_scores(self._tokenize(q))
                top_bm25 = np.argsort(raw_bm25)[::-1][:candidates_k]
                max_val = raw_bm25[top_bm25[0]] if raw_bm25[top_bm25[0]] > 0 else 1.0
                for i in top_bm25:
                    norm = float(raw_bm25[i]) / max_val
                    bm25_scores[int(i)] = max(bm25_scores.get(int(i), 0.0), norm)

        # ── RRF fusion ─────────────────────────────────────────────────
        if dense_scores:
            lo, hi = min(dense_scores.values()), max(dense_scores.values())
            span = hi - lo or 1.0
            dense_norm = {i: (s - lo) / span for i, s in dense_scores.items()}
        else:
            dense_norm = {}

        all_ids = set(dense_norm) | set(bm25_scores)
        fused = {
            i: 0.6 * dense_norm.get(i, 0.0) + 0.4 * bm25_scores.get(i, 0.0)
            for i in all_ids
        }

        top_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        results = []
        for i in top_ids:
            chunk = self.chunks[i].copy()
            chunk["score"] = round(fused[i], 4)
            results.append(chunk)
        return results

    # ── Internal ─────────────────────────────────────────────────────────

    def _load_embedding_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except ImportError:
            print("[RAG] Warning: sentence-transformers not installed.")
            return None

    def _load_and_index(self):
        if self.model is None:
            return
        if not os.path.isdir(self.knowledge_base_dir):
            os.makedirs(self.knowledge_base_dir, exist_ok=True)
            return

        # Read all supported files
        file_contents: dict[str, str] = {}
        for filename in sorted(os.listdir(self.knowledge_base_dir)):
            filepath = os.path.join(self.knowledge_base_dir, filename)
            if not os.path.isfile(filepath):
                continue
            ext = os.path.splitext(filename)[1].lower()
            reader = _FILE_READERS.get(ext)
            if reader is None:
                continue
            try:
                text = reader(filepath)  # type: ignore[operator]
            except Exception as e:
                print(f"[RAG] Warning: could not read {filename}: {e}")
                continue
            if text.strip():
                file_contents[filename] = text

        if not file_contents:
            return

        # Check cache
        content_hash = self._compute_hash(file_contents)
        cached = self._load_cache(content_hash)
        if cached:
            self.chunks, self.embeddings = cached
            self._build_index()
            self._build_bm25()
            print(f"[RAG] Loaded from cache ({content_hash[:8]}).")
            return

        # Build fresh
        all_chunks: list[dict] = []
        for filename, text in file_contents.items():
            all_chunks.extend(self._chunk_text(text, source_file=filename))

        self.chunks = all_chunks
        texts = [c["text"] for c in all_chunks]
        self.embeddings = np.array(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True),
            dtype=np.float32,
        )
        self._build_index()
        self._build_bm25()
        self._save_cache(content_hash)

    def _build_index(self):
        try:
            import faiss
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        except ImportError:
            print("[RAG] FAISS not available, using numpy fallback.")
            self.index = None

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(c["text"]) for c in self.chunks]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            print("[RAG] rank_bm25 not available, skipping sparse retrieval.")
            self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """CJK single-char + ASCII word tokenization."""
        return re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]|[a-zA-Z0-9]+', text.lower()) or [text]

    # ── Chunking (V3) ────────────────────────────────────────────────────

    def _chars_per_token(self, text: str) -> float:
        """CJK-aware char-to-token ratio: ~2 chars/token for Chinese, ~4 for English."""
        cjk = len(re.findall(r'[\u4e00-\u9fff]', text))
        ratio = cjk / max(len(text), 1)
        return 2.0 if ratio > 0.3 else 4.0

    def _parse_blocks(self, text: str) -> list[dict]:
        """Parse raw text into typed semantic blocks."""
        lines = text.split("\n")
        blocks: list[dict] = []
        pending_intro: str | None = None
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            # ── Markdown table ───────────────────────────────────────
            if _is_table_row(stripped):
                table_lines: list[str] = []
                while i < len(lines) and (
                    _is_table_row(lines[i].strip()) or _TABLE_SEP_RE.match(lines[i].strip())
                ):
                    table_lines.append(lines[i])
                    i += 1
                blocks.append({"type": "table", "lines": table_lines})
                pending_intro = None
                continue

            # ── Section header ───────────────────────────────────────
            if _is_header(stripped):
                blocks.append({"type": "header", "text": stripped})
                pending_intro = None
                i += 1
                continue

            # ── List item ────────────────────────────────────────────
            if _is_list_item(stripped):
                intro = (pending_intro + "\n") if pending_intro else ""
                blocks.append({"type": "list_item", "text": (intro + stripped).strip()})
                i += 1
                continue

            # ── Regular paragraph (collect until blank / structure change) ──
            para_lines = [stripped]
            i += 1
            while i < len(lines):
                ns = lines[i].strip()
                if not ns:
                    i += 1
                    break
                if _is_header(ns) or _is_table_row(ns) or _is_list_item(ns):
                    break
                para_lines.append(ns)
                i += 1

            para = " ".join(para_lines)
            if _is_list_intro(para):
                pending_intro = para
            else:
                pending_intro = None
            blocks.append({"type": "text", "text": para})

        return blocks

    def _align_overlap(self, text: str, overlap_chars: int) -> str:
        """Trim overlap to a sentence boundary so we don't split mid-sentence."""
        if not text or overlap_chars <= 0:
            return ""
        tail = text[-overlap_chars:]
        # Find first sentence start after the cut point
        m = re.search(r'(?<=[。！？.!?])\s*\S', tail)
        return tail[m.start():].strip() if m else tail.strip()

    @staticmethod
    def _header_level(text: str) -> int:
        """Return a numeric level for a header line (lower = higher level)."""
        stripped = text.strip()
        # Markdown #-level
        m = re.match(r'^(#{1,6})\s', stripped)
        if m:
            return len(m.group(1))
        # 第X章 → 1, 第X节 → 2, 第X条 → 3
        if re.match(r'^第[一二三四五六七八九十百\d]+章', stripped):
            return 1
        if re.match(r'^第[一二三四五六七八九十百\d]+[节篇]', stripped):
            return 2
        if re.match(r'^第[一二三四五六七八九十百\d]+条', stripped):
            return 3
        # 1. → 2, 1.1 → 3
        if re.match(r'^\d+\.\d+', stripped):
            return 3
        if re.match(r'^\d+\.', stripped):
            return 2
        # （一） → 3
        if re.match(r'^（[一二三四五六七八九十]+）', stripped):
            return 3
        return 4

    def _chunk_text(self, text: str, source_file: str) -> list[dict]:
        """V3 semantic chunking: structure-aware, table-header-preserving, intro-copying.

        Enhancement: each chunk carries the full ancestor header path
        (e.g. "第一章 总则 > 第一节 适用范围 > 1.1 定义") so retrieval
        can match queries that reference parent sections.
        """
        cpt = self._chars_per_token(text)
        max_chars = int(self.chunk_size * cpt)
        overlap_chars = int(self.chunk_overlap * cpt)

        blocks = self._parse_blocks(text)
        chunks: list[dict] = []
        idx = 0
        current = ""
        last_header = ""
        # Stack: list of (level, header_text)
        header_stack: list[tuple[int, str]] = []

        def _header_path() -> str:
            """Full ancestor path: 'Ch1 > Sec1 > 1.1'"""
            return " > ".join(h for _, h in header_stack) if header_stack else ""

        def flush(extra: str = "") -> str:
            nonlocal idx
            body = (current + ("\n" + extra if extra else "")).strip()
            if body:
                # Prepend full header path as context prefix
                path = _header_path()
                text_with_path = f"[{path}]\n{body}" if path else body
                chunks.append({
                    "text": text_with_path,
                    "source": source_file,
                    "chunk_index": idx,
                    "header_path": path,
                })
                idx += 1
            return body

        for block in blocks:
            btype = block["type"]

            # ── Header: always start a fresh chunk ───────────────────
            if btype == "header":
                flush()
                htext = block["text"]
                hlevel = self._header_level(htext)
                # Maintain a stack: pop same/deeper levels
                while header_stack and header_stack[-1][0] >= hlevel:
                    header_stack.pop()
                header_stack.append((hlevel, htext))
                last_header = htext
                current = htext + "\n"
                continue

            # ── Table: split rows while preserving header ─────────────
            if btype == "table":
                flush()
                current = last_header + "\n" if last_header else ""

                tlines = block["lines"]
                # Separate header rows from data rows
                header_rows: list[str] = []
                data_rows: list[str] = []
                sep_seen = False
                for tl in tlines:
                    if _TABLE_SEP_RE.match(tl.strip()):
                        header_rows.append(tl)
                        sep_seen = True
                    elif not sep_seen:
                        header_rows.append(tl)
                    else:
                        data_rows.append(tl)

                if not data_rows:
                    # No separator found — treat entire table as text block
                    current = "\n".join(tlines)
                    continue

                tbl_header = "\n".join(header_rows)
                tbl_header_len = len(tbl_header)
                bucket: list[str] = []
                bucket_len = tbl_header_len

                for row in data_rows:
                    row_len = len(row)
                    if bucket_len + row_len > max_chars and bucket:
                        tbl_chunk = tbl_header + "\n" + "\n".join(bucket)
                        chunks.append({"text": tbl_chunk, "source": source_file,
                                       "chunk_index": idx, "chunk_type": "table"})
                        idx += 1
                        bucket, bucket_len = [], tbl_header_len
                    bucket.append(row)
                    bucket_len += row_len

                if bucket:
                    tbl_chunk = tbl_header + "\n" + "\n".join(bucket)
                    chunks.append({"text": tbl_chunk, "source": source_file,
                                   "chunk_index": idx, "chunk_type": "table"})
                    idx += 1
                current = (last_header + "\n") if last_header else ""
                continue

            # ── Text / list_item ──────────────────────────────────────
            block_text = block["text"]

            if len(current) + len(block_text) + 1 <= max_chars:
                current = (current + "\n" + block_text).strip() + "\n"
            else:
                # Flush and start a new chunk with overlap
                flushed = flush()
                overlap = self._align_overlap(flushed, overlap_chars)
                header_ctx = (last_header + "\n").strip() if last_header else ""
                current = (header_ctx + "\n" + overlap + "\n" + block_text).strip() + "\n"

                # If a single block still exceeds max_chars, split by sentences
                if len(current) > max_chars:
                    sentences = re.split(r'(?<=[。！？.!?\n])', block_text)
                    current = header_ctx + "\n" if header_ctx else ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(current) + len(sent) <= max_chars:
                            current = (current + sent).strip() + " "
                        else:
                            flushed = flush()
                            overlap = self._align_overlap(flushed, overlap_chars)
                            current = (overlap + " " + sent).strip() + " "
                    current = current.strip() + "\n"

        flush()
        return chunks

    # ── Embedding cache ───────────────────────────────────────────────────

    def _compute_hash(self, file_contents: dict[str, str]) -> str:
        h = hashlib.md5()
        for name in sorted(file_contents):
            h.update(name.encode())
            h.update(file_contents[name].encode())
        h.update(str(self.chunk_size).encode())
        h.update(str(self.chunk_overlap).encode())
        return h.hexdigest()

    def _cache_paths(self, content_hash: str) -> tuple[str, str]:
        os.makedirs(self._cache_dir, exist_ok=True)
        return (
            os.path.join(self._cache_dir, f"{content_hash}.npz"),
            os.path.join(self._cache_dir, f"{content_hash}.json"),
        )

    def _load_cache(self, content_hash: str) -> tuple[list[dict], np.ndarray] | None:
        emb_path, chunks_path = self._cache_paths(content_hash)
        if not (os.path.exists(emb_path) and os.path.exists(chunks_path)):
            return None
        try:
            embeddings = np.load(emb_path)["embeddings"]
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            return chunks, embeddings
        except Exception as e:
            print(f"[RAG] Cache load failed ({e}), rebuilding.")
            return None

    def _save_cache(self, content_hash: str):
        emb_path, chunks_path = self._cache_paths(content_hash)
        try:
            np.savez_compressed(emb_path, embeddings=self.embeddings)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)
        except Exception as e:
            print(f"[RAG] Cache save failed: {e}")


# ── Tool Dispatch ───────────────────────────────────────────────────────────

def handle_rag_tool(tool_input: dict, rag: RAGSystem) -> str:
    """Dispatch a RAG tool call and return the JSON result string.

    Supports optional extra_queries for multi-query retrieval (query expansion).
    """
    query        = tool_input.get("query", "")
    top_k        = tool_input.get("top_k")
    extra        = tool_input.get("extra_queries")  # list[str] | None
    results      = rag.search(query, top_k, extra_queries=extra)
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)
