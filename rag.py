"""RAG system: document loading, chunking, embedding, and retrieval."""

import json
import os

import numpy as np

import config


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


# ── RAG System ──────────────────────────────────────────────────────────────

class RAGSystem:
    """Document retrieval using local sentence embeddings and FAISS."""

    def __init__(
        self,
        knowledge_base_dir: str = config.KNOWLEDGE_BASE_DIR,
        embedding_model: str = config.EMBEDDING_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.TOP_K_RESULTS,
    ):
        self.knowledge_base_dir = knowledge_base_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Will be populated by _load_and_index
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.index = None  # FAISS index or numpy fallback

        print(f"[RAG] Loading documents from {knowledge_base_dir} ...")
        self.model = self._load_embedding_model(embedding_model)
        self._load_and_index()
        print(f"[RAG] Indexed {len(self.chunks)} chunks from knowledge base.")

    # ── Public API ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for relevant document chunks."""
        if not self.chunks or self.embeddings is None:
            return [{"text": "No documents have been indexed in the knowledge base.", "source": "system"}]

        k = top_k or self.top_k
        k = min(k, len(self.chunks))

        query_emb = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_emb, dtype=np.float32)

        if self.index is not None:
            # FAISS search
            scores, indices = self.index.search(query_vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
            return results
        else:
            # Numpy fallback (cosine similarity via dot product on normalized vectors)
            scores = np.dot(self.embeddings, query_vec.T).flatten()
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                results.append(chunk)
            return results

    # ── Internal ─────────────────────────────────────────────────────────

    def _load_embedding_model(self, model_name: str):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except ImportError:
            print("[RAG] Warning: sentence-transformers not installed. RAG will not work.")
            return None

    def _load_and_index(self):
        """Load all documents, chunk them, embed, and build the search index."""
        if self.model is None:
            return

        # Collect chunks from all supported files
        all_chunks: list[dict] = []
        if not os.path.isdir(self.knowledge_base_dir):
            os.makedirs(self.knowledge_base_dir, exist_ok=True)
            return

        for filename in sorted(os.listdir(self.knowledge_base_dir)):
            filepath = os.path.join(self.knowledge_base_dir, filename)
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in (".txt", ".md"):
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError as e:
                print(f"[RAG] Warning: could not read {filename}: {e}")
                continue

            chunks = self._chunk_text(text, source_file=filename)
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        self.chunks = all_chunks

        # Encode all chunks
        texts = [c["text"] for c in all_chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index (with numpy fallback)
        try:
            import faiss
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        except ImportError:
            print("[RAG] FAISS not available, using numpy fallback for similarity search.")
            self.index = None

    def _chunk_text(self, text: str, source_file: str) -> list[dict]:
        """Split text into overlapping chunks of approximately chunk_size tokens."""
        # Simple heuristic: ~4 characters per token
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + chars_per_chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": source_file,
                    "chunk_index": idx,
                })
                idx += 1
            start += chars_per_chunk - overlap_chars

        return chunks


# ── Tool Dispatch ───────────────────────────────────────────────────────────

def handle_rag_tool(tool_input: dict, rag: RAGSystem) -> str:
    """Dispatch a RAG tool call and return the JSON result string."""
    query = tool_input.get("query", "")
    top_k = tool_input.get("top_k")
    results = rag.search(query, top_k)
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)
