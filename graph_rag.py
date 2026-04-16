"""GraphRAG: Neo4j-backed entity extraction and graph-based retrieval.

Architecture
────────────
1. Index phase (``GraphRAGSystem.index_document``):
   - Call Claude API to extract (entity, relation, entity) triples from each chunk.
   - Write entities as Neo4j nodes, relations as edges.

2. Search phase (``GraphRAGSystem.search``):
   - Extract query entities with Claude.
   - Find matching nodes in Neo4j.
   - Traverse 1–2 hops to gather related context.
   - Return ranked text passages associated with the subgraph.

Usage
─────
    from graph_rag import GraphRAGSystem
    g = GraphRAGSystem(client)          # client = anthropic.Anthropic(...)
    g.index_document("file.md", text)   # call once per document
    results = g.search("query text")

Graceful degradation
────────────────────
If neo4j-driver is not installed, ``GraphRAGSystem`` initialises in *stub mode*:
all methods return empty results so the main agent continues working.

Setup
─────
    pip install neo4j
    docker run -p 7474:7474 -p 7687:7687 \\
        -e NEO4J_AUTH=neo4j/password neo4j:5

Set in .env:
    AI_NEO4J_URI=bolt://localhost:7687
    AI_NEO4J_USER=neo4j
    AI_NEO4J_PASSWORD=password
"""

import json
import os

import anthropic

import config


# ── Neo4j connection ─────────────────────────────────────────────────────────

def _neo4j_driver():
    """Return a Neo4j driver or None if neo4j package is not installed."""
    try:
        from neo4j import GraphDatabase
        uri  = os.environ.get("AI_NEO4J_URI",      "bolt://localhost:7687")
        user = os.environ.get("AI_NEO4J_USER",     "neo4j")
        pwd  = os.environ.get("AI_NEO4J_PASSWORD", "password")
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()
        return driver
    except ImportError:
        return None
    except Exception as exc:
        print(f"[GraphRAG] Neo4j connection failed: {exc}")
        return None


# ── LLM prompts ──────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are an information extraction assistant. Extract factual (subject, relation, object) triples from the text.

Rules:
- Subject and object must be specific named entities (product names, systems, error codes, people, etc.)
- Relation must be a short verb phrase in snake_case (e.g. causes, depends_on, resolves, belongs_to)
- Return ONLY a JSON array: [["subject", "relation", "object"], ...]
- Maximum 20 triples. Skip generic/vague entities.
- If no clear triples exist, return []
"""

_QUERY_ENTITY_SYSTEM = """\
You are an entity recogniser. List the key named entities in the user's query.
Return ONLY a JSON array of strings: ["entity1", "entity2", ...]
Maximum 5 entities. If none, return [].
"""


# ── GraphRAG System ──────────────────────────────────────────────────────────

class GraphRAGSystem:
    """Entity-relationship graph built from documents, queried for multi-hop retrieval."""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self._driver = _neo4j_driver()
        if self._driver is None:
            print("[GraphRAG] Running in stub mode (neo4j not available).")
        else:
            self._init_schema()

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._driver is not None

    def index_document(self, source: str, text: str, chunk_size: int = 1500) -> int:
        """Extract triples from *text* and write them to Neo4j.

        Args:
            source:     document filename / identifier (stored on nodes as metadata)
            text:       full document text
            chunk_size: characters per LLM call (keep within token budget)

        Returns:
            Number of triples written.
        """
        if not self.available:
            return 0

        total = 0
        for start in range(0, len(text), chunk_size):
            chunk = text[start:start + chunk_size]
            triples = self._extract_triples(chunk)
            if triples:
                self._upsert_triples(triples, source)
                total += len(triples)

        print(f"[GraphRAG] Indexed {total} triples from '{source}'.")
        return total

    def search(self, query: str, hops: int = 2, max_results: int = 5) -> list[dict]:
        """Find relevant passages by graph traversal from query entities.

        Args:
            query:       user query string
            hops:        how many relationship hops to traverse (1 or 2)
            max_results: max passages to return

        Returns:
            List of dicts with keys: entity, relation_path, context, source
        """
        if not self.available:
            return []

        entities = self._extract_query_entities(query)
        if not entities:
            return []

        results = []
        seen: set[str] = set()

        with self._driver.session() as session:
            for entity in entities[:3]:  # limit anchor entities
                rows = session.execute_read(
                    self._traverse, entity, hops, max_results
                )
                for row in rows:
                    key = (row["subject"], row["relation"], row["obj"])
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "entity":        row["subject"],
                        "relation_path": row["relation"],
                        "context":       f"{row['subject']} {row['relation'].replace('_', ' ')} {row['obj']}",
                        "source":        row.get("source", ""),
                    })
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break

        return results

    def clear(self, source: str | None = None):
        """Delete all graph data (or only nodes from a specific source)."""
        if not self.available:
            return
        with self._driver.session() as session:
            if source:
                session.run(
                    "MATCH (n {source: $src})-[r]-() DELETE r, n",
                    src=source,
                )
                session.run("MATCH (n {source: $src}) DELETE n", src=source)
            else:
                session.run("MATCH (n) DETACH DELETE n")

    def close(self):
        if self._driver:
            self._driver.close()

    # ── Schema ───────────────────────────────────────────────────────────

    def _init_schema(self):
        """Create indexes for fast entity lookup."""
        with self._driver.session() as session:
            session.run(
                "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)"
            )

    # ── Triple extraction (LLM) ──────────────────────────────────────────

    def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """Ask Claude to extract (subject, relation, object) triples."""
        try:
            resp = self.client.messages.create(
                model=config.MODEL,
                max_tokens=1024,
                system=_EXTRACT_SYSTEM,
                messages=[{"role": "user", "content": text[:3000]}],
            )
            raw = next((b.text for b in resp.content if b.type == "text"), "[]")
            data = json.loads(raw.strip())
            return [
                (str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip())
                for t in data
                if isinstance(t, list) and len(t) == 3 and all(t)
            ]
        except Exception as exc:
            print(f"[GraphRAG] Triple extraction error: {exc}")
            return []

    def _extract_query_entities(self, query: str) -> list[str]:
        """Ask Claude to identify named entities in the query."""
        try:
            resp = self.client.messages.create(
                model=config.MODEL,
                max_tokens=128,
                system=_QUERY_ENTITY_SYSTEM,
                messages=[{"role": "user", "content": query}],
            )
            raw = next((b.text for b in resp.content if b.type == "text"), "[]")
            return [str(e).strip() for e in json.loads(raw.strip()) if e]
        except Exception:
            return []

    # ── Neo4j writes ─────────────────────────────────────────────────────

    def _upsert_triples(
        self, triples: list[tuple[str, str, str]], source: str
    ):
        """Write triples as (Entity)-[RELATION]->(Entity) in Neo4j."""
        with self._driver.session() as session:
            for subj, rel, obj in triples:
                rel_type = rel.upper().replace(" ", "_").replace("-", "_")
                # MERGE prevents duplicates; SET updates source metadata
                session.run(
                    f"""
                    MERGE (s:Entity {{name: $subj}})
                    ON CREATE SET s.source = $src
                    MERGE (o:Entity {{name: $obj}})
                    ON CREATE SET o.source = $src
                    MERGE (s)-[r:{rel_type}]->(o)
                    ON CREATE SET r.source = $src, r.relation = $rel
                    """,
                    subj=subj, obj=obj, rel=rel, src=source,
                )

    # ── Neo4j reads ──────────────────────────────────────────────────────

    @staticmethod
    def _traverse(tx, entity: str, hops: int, limit: int):
        """Traverse up to *hops* hops from *entity* and return related triples."""
        if hops == 1:
            result = tx.run(
                """
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($name)
                   OR toLower(o.name) CONTAINS toLower($name)
                RETURN s.name AS subject, type(r) AS relation,
                       o.name AS obj, s.source AS source
                LIMIT $limit
                """,
                name=entity, limit=limit,
            )
        else:
            result = tx.run(
                """
                MATCH path = (s:Entity)-[*1..2]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($name)
                WITH s, relationships(path)[0] AS r, o, s.source AS source
                RETURN s.name AS subject, type(r) AS relation,
                       o.name AS obj, source
                LIMIT $limit
                """,
                name=entity, limit=limit,
            )
        return [dict(row) for row in result]
