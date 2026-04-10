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


def _is_valid_table(table: list) -> bool:
    """Three-gate quality check for an extracted table."""
    if not table or len(table) < 2:
        return False
    col_counts = [len(row) for row in table]
    if len(set(col_counts)) > 1:          # inconsistent column count
        return False
    total = sum(col_counts)
    empty = sum(1 for row in table for cell in row if not (cell or "").strip())
    if total and empty / total > 0.3:     # >30% empty cells
        return False
    has_nums = any(
        any(c.isdigit() for c in (cell or ""))
        for row in table for cell in row
    )
    return has_nums


_OCR_NUM_MAP = str.maketrans({
    "O": "0", "o": "0",         # O/o → 0
    "l": "1", "I": "1",         # l/I → 1
    "S": "5",                   # S   → 5
    "B": "8",                   # B   → 8
    "Z": "2",                   # Z   → 2
    "q": "9",                   # q   → 9
})

_NUMERIC_CTX_RE = re.compile(
    r'^[\d\s,，.。¥￥$€£元万亿%±\-+/（）()\[\]OolISBZq]+$'
)


def _fix_ocr_numbers(cell: str) -> str:
    """Fix systematic OCR character confusions in numeric/amount table cells.

    Only applies when the cell is predominantly numeric context to avoid
    incorrectly replacing letters in text cells (e.g. "SaaS" → "5aa5" is wrong).
    """
    if not cell:
        return cell
    stripped = cell.strip()
    if not stripped:
        return cell
    if _NUMERIC_CTX_RE.match(stripped):
        return stripped.translate(_OCR_NUM_MAP)
    return cell


def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table to GitHub-flavored Markdown.

    Applies OCR number correction to each cell before formatting.
    """
    if not table:
        return ""

    def fmt(cell) -> str:
        return _fix_ocr_numbers(str(cell or "").strip())

    header = "| " + " | ".join(fmt(c) for c in table[0]) + " |"
    sep    = "|" + "|".join(["---"] * len(table[0])) + "|"
    rows   = [
        "| " + " | ".join(fmt(c) for c in row) + " |"
        for row in table[1:]
    ]
    return "\n".join([header, sep] + rows)


def _is_scanned_page(page_text: str, page_width: float = 0) -> bool:
    """Detect scanned pages by text density (very few characters → likely scanned)."""
    chars = len((page_text or "").strip())
    return chars < 20  # heuristic: real pages have at least a few dozen chars


def _extract_tables_camelot(filepath: str, page_num: int) -> list[str]:
    """P3: Camelot visual fallback for tables that pdfplumber fails on.

    Camelot uses two algorithms:
      - lattice: detects tables via visible grid lines (handles bordered tables)
      - stream:  detects tables via whitespace gaps (handles borderless tables)

    No GPU needed. Requires: pip install camelot-py[cv]

    Returns list of Markdown strings, one per detected table.
    Falls back gracefully if camelot is not installed.
    """
    try:
        import camelot
    except ImportError:
        return []

    results = []
    page_str = str(page_num + 1)  # camelot uses 1-based page numbers

    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(filepath, pages=page_str, flavor=flavor)
            for tbl in tables:
                df = tbl.df
                if df.empty or len(df) < 2:
                    continue
                # Convert DataFrame to list-of-lists for _is_valid_table check
                tbl_list = [df.columns.tolist()] + df.values.tolist()
                if not _is_valid_table(tbl_list):
                    continue
                # DataFrame to Markdown
                header = "| " + " | ".join(_fix_ocr_numbers(str(c)) for c in df.columns) + " |"
                sep    = "|" + "|".join(["---"] * len(df.columns)) + "|"
                rows   = [
                    "| " + " | ".join(_fix_ocr_numbers(str(v)) for v in row) + " |"
                    for _, row in df.iterrows()
                ]
                results.append("\n".join([header, sep] + rows))
            if results:
                break  # lattice succeeded, no need to try stream
        except Exception:
            continue

    return results


def _read_pdf(filepath: str) -> str:
    """Structured PDF reader: pdfplumber for tables → Markdown, plain text for prose.

    Strategy:
      1. Try pdfplumber (preserves table structure)
      2. For each page, extract tables → Markdown; remaining text as-is
      3. Detect scanned pages and mark them for future OCR routing
      4. Cross-page table merging: if current page ends with partial table
         and next page starts with matching column count, merge them
      5. Fall back to pypdf plain-text extraction if pdfplumber unavailable
    """
    try:
        import pdfplumber
    except ImportError:
        # Fallback to pypdf (structure-blind but better than nothing)
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                print(f"[RAG] Warning: {filepath} appears to be scanned (no text extracted)")
            return text
        except ImportError:
            print("[RAG] Warning: neither pdfplumber nor pypdf installed.")
            return ""

    pages_text: list[str] = []
    pending_table_header: list | None = None   # for cross-page table merging
    pending_table_rows:   list       = []

    def flush_pending_table() -> str:
        nonlocal pending_table_header, pending_table_rows
        if pending_table_header is None:
            return ""
        tbl = [pending_table_header] + pending_table_rows
        pending_table_header = None
        pending_table_rows   = []
        return _table_to_markdown(tbl) if _is_valid_table(tbl) else ""

    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages):
                raw_text   = page.extract_text() or ""
                is_scanned = _is_scanned_page(raw_text)

                if is_scanned:
                    # Mark scanned pages; full OCR pipeline is a future P2 item
                    pages_text.append(f"[Page {page_num + 1}: scanned image, text unavailable]")
                    pages_text.append(flush_pending_table())
                    continue

                page_parts: list[str] = []

                # ── Extract tables with structure ─────────────────────
                tables = page.extract_tables() or []
                for tbl in tables:
                    if not tbl:
                        continue

                    # Cross-page merge: if we have a pending table whose
                    # column count matches this table's first data row, merge
                    if (pending_table_header is not None
                            and len(tbl[0]) == len(pending_table_header)):
                        # First row of new page is likely data (not a new header)
                        pending_table_rows.extend(tbl)
                        continue

                    # Flush any pending cross-page table first
                    flushed = flush_pending_table()
                    if flushed:
                        page_parts.append(flushed)

                    if _is_valid_table(tbl):
                        md = _table_to_markdown(tbl)
                        page_parts.append(md)
                        # Track last table for potential cross-page continuation
                        pending_table_header = tbl[0]
                        pending_table_rows   = list(tbl[1:])
                    else:
                        # P3: pdfplumber quality check failed → try camelot fallback
                        camelot_tables = _extract_tables_camelot(filepath, page_num)
                        if camelot_tables:
                            page_parts.extend(camelot_tables)
                        else:
                            # Both failed — treat rows as plain text
                            page_parts.append(
                                "\n".join(
                                    " ".join(str(c or "") for c in row) for row in tbl
                                )
                            )

                # ── Remaining prose text (non-table regions) ──────────
                if raw_text.strip():
                    page_parts.append(raw_text)

                pages_text.append("\n\n".join(p for p in page_parts if p.strip()))

            # Flush any table that started on the last page
            flushed = flush_pending_table()
            if flushed:
                pages_text.append(flushed)

    except Exception as exc:
        print(f"[RAG] pdfplumber failed for {filepath}: {exc}, falling back to pypdf")
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    return "\n\n".join(p for p in pages_text if p.strip())


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
    """Document retrieval using local sentence embeddings and FAISS + BM25.

    Advanced retrieval features:
      - Multi-granularity indexing (child chunks for retrieval, parent for context)
      - Cross-encoder reranking (optional, activated when sentence-transformers cross-encoder is available)
      - HyDE support (caller passes hypothetical document via extra_queries)
    """

    _CACHE_SUBDIR = ".rag_cache"
    # Parent chunk is _PARENT_RATIO times larger than child chunk
    _PARENT_RATIO = 3

    def __init__(
        self,
        knowledge_base_dir: str = config.KNOWLEDGE_BASE_DIR,
        embedding_model: str = config.EMBEDDING_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.TOP_K_RESULTS,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.knowledge_base_dir = knowledge_base_dir
        self.chunk_size = chunk_size        # child chunk size in tokens
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._cache_dir = os.path.join(knowledge_base_dir, self._CACHE_SUBDIR)

        # child chunks: used for bi-encoder retrieval (fine-grained)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.index = None
        self._bm25 = None

        # parent chunks: returned to caller for full context
        # key = (source_file, parent_idx), value = text
        self._parent_chunks: dict[tuple, str] = {}

        # cross-encoder reranker (optional)
        self._cross_encoder = self._load_cross_encoder(cross_encoder_model)

        print(f"[RAG] Loading documents from {knowledge_base_dir} ...")
        self.model = self._load_embedding_model(embedding_model)
        self._load_and_index()
        rerank_status = "cross-encoder" if self._cross_encoder else "no reranker"
        print(f"[RAG] Indexed {len(self.chunks)} child chunks | {rerank_status}.")

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

        # ── Cross-encoder reranking ────────────────────────────────────
        # Expand candidate pool for reranking, then cut to k after
        rerank_k = min(k * 3, len(fused))
        candidate_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:rerank_k]

        if self._cross_encoder and len(candidate_ids) > 1:
            pairs = [(query, self.chunks[i]["text"]) for i in candidate_ids]
            try:
                ce_scores = self._cross_encoder.predict(pairs)
                # Re-sort by cross-encoder score
                ranked = sorted(zip(candidate_ids, ce_scores),
                                key=lambda x: x[1], reverse=True)
                top_ids = [i for i, _ in ranked[:k]]
            except Exception:
                top_ids = candidate_ids[:k]
        else:
            top_ids = candidate_ids[:k]

        # ── Expand child → parent chunk for richer context ────────────
        results = []
        for i in top_ids:
            child = self.chunks[i]
            parent_key = child.get("parent_key")
            if parent_key and parent_key in self._parent_chunks:
                text = self._parent_chunks[parent_key]
            else:
                text = child["text"]
            results.append({
                "text": text,
                "source": child["source"],
                "chunk_index": child["chunk_index"],
                "header_path": child.get("header_path", ""),
                "score": round(fused.get(i, 0.0), 4),
            })
        return results

    # ── Internal ─────────────────────────────────────────────────────────

    def _load_embedding_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except ImportError:
            print("[RAG] Warning: sentence-transformers not installed.")
            return None

    def _load_cross_encoder(self, model_name: str):
        """Load cross-encoder reranker. Returns None if not available (graceful degradation)."""
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(model_name)
            return ce
        except Exception:
            # Not installed or model unavailable — retrieval still works without reranking
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

        # Build fresh — multi-granularity indexing
        all_child_chunks: list[dict] = []
        for filename, text in file_contents.items():
            # Child chunks (small, for precise retrieval)
            child_chunks = self._chunk_text(text, source_file=filename)

            # Parent chunks (larger context, returned when child is retrieved)
            parent_size = self.chunk_size * self._PARENT_RATIO
            parent_chunks = self._chunk_text_with_size(
                text, source_file=filename, chunk_size=parent_size
            )
            # Build lookup: map each child to nearest parent
            for child in child_chunks:
                parent_key = self._find_parent_key(child, parent_chunks)
                child["parent_key"] = parent_key
                if parent_key and parent_key not in self._parent_chunks:
                    pid, pidx = parent_key
                    parent = next(
                        (p for p in parent_chunks
                         if p["source"] == pid and p["chunk_index"] == pidx),
                        None,
                    )
                    if parent:
                        self._parent_chunks[parent_key] = parent["text"]

            all_child_chunks.extend(child_chunks)

        self.chunks = all_child_chunks
        texts = [c["text"] for c in all_child_chunks]
        self.embeddings = np.array(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True),
            dtype=np.float32,
        )
        self._build_index()
        self._build_bm25()
        self._save_cache(content_hash)

    def _chunk_text_with_size(self, text: str, source_file: str,
                               chunk_size: int) -> list[dict]:
        """Chunk with a custom size (used for parent chunks)."""
        original = self.chunk_size
        self.chunk_size = chunk_size
        result = self._chunk_text(text, source_file)
        self.chunk_size = original
        return result

    @staticmethod
    def _find_parent_key(child: dict, parents: list[dict]) -> tuple | None:
        """Find the parent chunk whose text contains the child's text."""
        c_text = child.get("text", "")
        # Use a short fingerprint from the middle of the child text to locate parent
        fp = c_text[len(c_text) // 4: len(c_text) // 4 + 40].strip()
        if not fp:
            return None
        for p in parents:
            if fp in p.get("text", ""):
                return (p["source"], p["chunk_index"])
        return None

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

    # ── Chunking configuration ────────────────────────────────────────────────
    # Levels that ALWAYS force a flush regardless of current chunk size.
    # Level 1 = 第X章 / #
    # Level 2 = 第X节 / ##  / 1.
    # Level 3+ = 第X条 / ### / 1.1 / （一）  ← only flush when chunk is big enough
    _FORCE_FLUSH_LEVELS: frozenset = frozenset({1, 2})
    _MIN_CHUNK_RATIO = 0.25  # minor headers flush only when chunk >= 25% of max_chars

    def _chunk_text(self, text: str, source_file: str) -> list[dict]:
        """V3 semantic chunking with level-aware flushing and parent-section context.

        Key improvements over naive header splitting:

        1. Level-based flushing:
           H1/H2 (章/节) always start a new chunk — they are true semantic boundaries.
           H3+ (条/小节) only flush when the current chunk is ≥ 25% of max_chars.
           This prevents tiny isolated chunks for short subsections.

        2. Parent-section intro carrying:
           When a low-level header eventually does flush, the opening paragraph of
           the nearest H1/H2 ancestor ('section_intro') is prepended to the new chunk.
           This gives downstream LLMs the broader context ("保险责任" vs "责任免除").

        3. Full header path prefix:
           Every chunk carries [Ch1 > Sec1 > 1.1] so queries about parent sections
           still match child chunks.
        """
        cpt = self._chars_per_token(text)
        max_chars = int(self.chunk_size * cpt)
        min_chars = int(max_chars * self._MIN_CHUNK_RATIO)
        overlap_chars = int(self.chunk_overlap * cpt)

        blocks = self._parse_blocks(text)
        chunks: list[dict] = []
        idx = 0
        current = ""
        last_header = ""
        header_stack: list[tuple[int, str]] = []
        # First substantial paragraph after the nearest major (level≤2) header.
        # Carried into sub-chunks so they retain parent-section context.
        section_intro: str = ""
        intro_captured = False   # whether we've captured the intro for current major section

        def _header_path() -> str:
            return " > ".join(h for _, h in header_stack) if header_stack else ""

        def flush(extra: str = "") -> str:
            nonlocal idx
            body = (current + ("\n" + extra if extra else "")).strip()
            if body:
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

            # ── Header ───────────────────────────────────────────────
            if btype == "header":
                htext  = block["text"]
                hlevel = self._header_level(htext)

                # Level-based flushing decision
                is_major  = hlevel in self._FORCE_FLUSH_LEVELS
                big_enough = len(current) >= min_chars

                if is_major or big_enough:
                    flush()
                    if is_major:
                        # Entering a new major section: reset intro context
                        section_intro   = ""
                        intro_captured  = False
                    else:
                        # Minor header flush: next chunk starts with section_intro
                        # so the reader has parent context
                        ctx = section_intro + "\n" if section_intro else ""
                        current = ctx
                    # After flush, start fresh accumulation with this header
                    current = current + htext + "\n"
                else:
                    # Minor header, chunk too small: DON'T flush.
                    # Append header as an in-chunk separator — keeps subsections
                    # together when they're collectively short.
                    if current.strip():
                        current = current.rstrip() + "\n\n"
                    current += htext + "\n"

                # Always update the stack
                while header_stack and header_stack[-1][0] >= hlevel:
                    header_stack.pop()
                header_stack.append((hlevel, htext))
                last_header = htext
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

            # Capture the first substantial paragraph after a major header
            # as section_intro for sub-chunk context carrying.
            if (not intro_captured
                    and block_text.strip()
                    and header_stack
                    and header_stack[0][0] in self._FORCE_FLUSH_LEVELS):
                section_intro  = block_text[:300]  # first 300 chars is enough for context
                intro_captured = True

            if len(current) + len(block_text) + 1 <= max_chars:
                current = (current + "\n" + block_text).strip() + "\n"
            else:
                # Flush and start new chunk with section_intro + overlap for context
                flushed = flush()
                overlap    = self._align_overlap(flushed, overlap_chars)
                header_ctx = (last_header + "\n").strip() if last_header else ""
                # Prepend section_intro when available so sub-chunks retain
                # the broader "what this section is about" context
                intro_ctx  = (f"[Context: {section_intro}]\n"
                              if section_intro and section_intro not in flushed else "")
                current = (intro_ctx + header_ctx + "\n" + overlap + "\n" + block_text).strip() + "\n"

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


# ── P2-B: LLM-based retrieval quality evaluation ─────────────────────────────

def evaluate_rag_retrieval(
    rag: RAGSystem,
    test_cases: list[dict],
    llm_client=None,
    llm_model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Evaluate RAG retrieval quality using LLM as judge.

    Args:
        rag:        RAGSystem instance to evaluate
        test_cases: list of {"query": str, "expected_keywords": list[str]}
                    OR {"query": str, "ground_truth": str}
        llm_client: optional anthropic.Anthropic client for LLM scoring
                    (falls back to keyword matching if None)
        llm_model:  cheap model for scoring (Haiku recommended)

    Returns:
        {"total": N, "hits": M, "hit_rate": 0.xx, "badcases": [...]}

    Usage:
        from rag import RAGSystem, evaluate_rag_retrieval
        import anthropic

        rag = RAGSystem()
        cases = [
            {"query": "45岁重疾险保费", "expected_keywords": ["45", "保费", "重疾"]},
            {"query": "等待期规定",      "ground_truth": "本合同等待期为180天"},
        ]
        report = evaluate_rag_retrieval(rag, cases)
        print(f"Hit rate: {report['hit_rate']:.1%}")
    """
    hits = 0
    badcases = []

    for case in test_cases:
        query   = case.get("query", "")
        results = rag.search(query, top_k=3)
        retrieved_text = "\n---\n".join(r.get("text", "") for r in results)

        hit = False

        if "ground_truth" in case and llm_client:
            # LLM judge: does the retrieved text contain the answer?
            prompt = (
                f"Query: {query}\n\n"
                f"Expected answer hint: {case['ground_truth']}\n\n"
                f"Retrieved chunks:\n{retrieved_text[:2000]}\n\n"
                "Does the retrieved text contain enough information to answer the query? "
                "Reply with only YES or NO."
            )
            try:
                resp = llm_client.messages.create(
                    model=llm_model,
                    max_tokens=5,
                    messages=[{"role": "user", "content": prompt}],
                )
                hit = "yes" in (resp.content[0].text or "").lower()
            except Exception:
                hit = False

        elif "expected_keywords" in case:
            # Keyword matching fallback (no LLM needed)
            kws = [k.lower() for k in case["expected_keywords"]]
            text_lower = retrieved_text.lower()
            hit = all(kw in text_lower for kw in kws)

        if hit:
            hits += 1
        else:
            badcases.append({
                "query":     query,
                "retrieved": retrieved_text[:300],
                "expected":  case.get("ground_truth") or case.get("expected_keywords"),
            })

    total = len(test_cases)
    return {
        "total":    total,
        "hits":     hits,
        "hit_rate": hits / total if total else 0.0,
        "badcases": badcases,
    }
