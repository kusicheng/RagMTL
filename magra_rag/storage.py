                                                                                   

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Iterable

from .resources import BudgetExceeded, ResourceBudget

_MAX_ALLOWED_DOCUMENTS = 1000
_MAX_CANDIDATE_BYTES = 64_000_000
_MAX_RESULT_BYTES = 128_000_000
_SCOPE_METADATA_PREFIX = "bm25_scope:"


_DOC_METADATA = {"source", "domain", "url", "published_at", "fetched_at", "sha256",
                 "sentences", "context_id", "license", "snapshot_id", "synthetic",
                 "perturbation", "original_id", "title", "date", "as_of"}
_DOC_METADATA |= {"generator", "case", "document_date", "track", "source_url", "final_url",
                  "content_type", "last_modified"}
_FORBIDDEN = {"answer", "answers", "relevance", "qrels", "supporting_facts",
              "supporting_evidence", "gold", "labels", "question", "query"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


class DocumentStore:
    def __init__(self, path: str | Path, *, budget: ResourceBudget | None = None,
                 readonly: bool = False):
        self.path = Path(path).resolve()
        self.budget = budget or ResourceBudget(self.path.parent)
        self.budget.contains(self.path)
        self.readonly = readonly
        if readonly:
            self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("PRAGMA temp_store=FILE")
        if not readonly:
                                                                                          
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            page_size = self.connection.execute("PRAGMA page_size").fetchone()[0]
                                                                                         
            cap = max(1, (self.budget.disk_limit - self.budget.reserve) // (2 * page_size))
            self.connection.execute(f"PRAGMA max_page_count={cap}")
            self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL,
                content_hash TEXT NOT NULL, token_count INTEGER NOT NULL,
                source TEXT NOT NULL DEFAULT '', domain TEXT NOT NULL DEFAULT '',
                context_id TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS docs_source ON documents(source);
            CREATE INDEX IF NOT EXISTS docs_context ON documents(context_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, text, content='documents', content_rowid='rowid',
                tokenize='porter unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid,title,text) VALUES(new.rowid,new.title,new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts,rowid,title,text)
                    VALUES('delete',old.rowid,old.title,old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts,rowid,title,text)
                    VALUES('delete',old.rowid,old.title,old.text);
                INSERT INTO documents_fts(rowid,title,text) VALUES(new.rowid,new.title,new.text);
            END;
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS vectors (
                doc_id TEXT PRIMARY KEY REFERENCES documents(id), model TEXT NOT NULL,
                content_hash TEXT NOT NULL, dimensions INTEGER NOT NULL, embedding BLOB NOT NULL
            );
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _validate(document: dict) -> tuple:
        leaked = _FORBIDDEN.intersection(document)
        if leaked:
            raise ValueError(f"Refusing query/annotation fields in retrieval store: {sorted(leaked)}")
        for key in ("id", "title", "text"):
            if key not in document or not isinstance(document[key], str):
                raise ValueError(f"Document {key} must be a string")
        if not document["id"]:
            raise ValueError("Document ID cannot be empty")
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Document metadata must be a mapping")
        unknown = set(metadata) - _DOC_METADATA
        if unknown:
            raise ValueError(f"Document metadata not in public allowlist: {sorted(unknown)}")
                                                                                                
        for key, value in metadata.items():
            if isinstance(value, (dict, list, tuple)) and not (
                key == "sentences" and isinstance(value, list)
                and all(isinstance(s, str) for s in value)
            ):
                raise ValueError(f"Structured metadata is not allowed: {key}")
        title, text = document["title"], document["text"]
        serialized_metadata = json.dumps(metadata, ensure_ascii=False)
        if sum(len(value.encode()) for value in (document["id"], title, text, serialized_metadata)) > 16_000_000:
            raise ValueError("Document exceeds 16 MB bound; split owned source documents before ingestion")
        digest = hashlib.sha256((title + "\0" + text).encode()).hexdigest()
        return (document["id"], title, text, serialized_metadata, digest,
                len(tokenize(title + " " + text)), str(metadata.get("source", "")),
                str(metadata.get("domain", "")), str(metadata.get("context_id", "")))

    def ingest(self, documents: Iterable[dict], *, batch_size: int = 256) -> dict:
        if self.readonly:
            raise ValueError("Read-only store")
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        count, existing, batch = 0, 0, []
        for document in documents:
            batch.append(self._validate(document))
            if len(batch) >= batch_size:
                n, old = self._insert_batch(batch)
                count, existing, batch = count + n, existing + old, []
        if batch:
            n, old = self._insert_batch(batch)
            count, existing = count + n, existing + old
        return {"inserted": count, "already_present": existing, "total_documents": len(self),
                "resources": self.budget.check()}

    def _insert_batch(self, rows: list[tuple]) -> tuple[int, int]:
                                                                                     
        scope_count = self.connection.execute(
            "SELECT COUNT(*) FROM metadata WHERE substr(key,1,?)=?",
            (len(_SCOPE_METADATA_PREFIX), _SCOPE_METADATA_PREFIX)).fetchone()[0]
        estimated = (sum(len(str(v).encode()) for row in rows for v in row) * 12 + 65536) * (1 + scope_count)
                                                                                   
                                                                                   
        self.budget.check(estimated + self.path.stat().st_size)
        count, old = 0, 0
        with self.connection:
            for row in rows:
                found = self.connection.execute(
                    "SELECT content_hash,metadata FROM documents WHERE id=?", (row[0],)).fetchone()
                if found:
                    if found["content_hash"] != row[4] or found["metadata"] != row[3]:
                        raise ValueError(f"Conflicting immutable document ID: {row[0]}")
                    old += 1
                    continue
                self.connection.execute("""INSERT INTO documents
                    (id,title,text,metadata,content_hash,token_count,source,domain,context_id)
                    VALUES(?,?,?,?,?,?,?,?,?)""", row)
                count += 1
            self.budget.check()
        return count, old

    def __len__(self):
        return self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    @staticmethod
    def _document(row) -> dict:
        return {"id": row["id"], "title": row["title"], "text": row["text"],
                "metadata": json.loads(row["metadata"])}

    def get(self, doc_id: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return self._document(row) if row else None

    def iter_documents(self, *, source: str | None = None):
        cursor = self.connection.execute("SELECT * FROM documents" +
                                         (" WHERE source=?" if source else "") + " ORDER BY id",
                                         (source,) if source else ())
        for row in cursor:
            yield self._document(row)

    def corpus_hash(self) -> str:
        digest = hashlib.sha256()
        for row in self.connection.execute(
            "SELECT id,content_hash,metadata FROM documents ORDER BY id"
        ):
            digest.update(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode())
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _scope_conditions(source=None, id_prefix=None, *, alias="d"):
        conditions, parameters = [], []
        if source is not None:
            conditions.append(f"{alias}.source=?")
            parameters.append(source)
        if id_prefix is not None:
            conditions.append(f"substr({alias}.id,1,?)=?")
            parameters.extend([len(id_prefix), id_prefix])
        return conditions, parameters

    def ensure_bm25_scope(self, *, source=None, id_prefix=None) -> str:
        pass                                                                   

                                                                           
                                                                             
                                                                          
                                                                                   
           
        source, id_prefix = self._validate_scope(source, id_prefix)
        if source is None and id_prefix is None:
            return "documents_fts"
        spec = json.dumps({"version": 1, "source": source, "id_prefix": id_prefix,
                           "tokenizer": "porter unicode61"}, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(spec.encode()).hexdigest()
        table, key = "documents_scope_" + digest, _SCOPE_METADATA_PREFIX + digest
        with closing(self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,))) as cursor:
            registered = cursor.fetchone()
        with closing(self.connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))) as cursor:
            exists = cursor.fetchone() is not None
        if registered is not None:
            if registered[0] != spec or not exists:
                raise ValueError("Scoped BM25 index metadata is inconsistent; rebuild explicitly")
            return table
        if exists:
            raise ValueError("Unregistered scoped BM25 index; rebuild explicitly")
        if self.readonly:
            raise ValueError("Scoped BM25 index is absent; build ensure_bm25_scope in a writable store first")
        if self.connection.in_transaction:
            raise ValueError("Build a scoped BM25 index outside an active write transaction")
        conditions, parameters = self._scope_conditions(source, id_prefix)
        where = " AND ".join(conditions)
        with closing(self.connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(length(CAST(title AS BLOB))+length(CAST(text AS BLOB))),0) "
                "FROM documents d WHERE " + where, parameters)) as cursor:
            count, content_bytes = cursor.fetchone()
                                                                            
                                                                               
        self.budget.check(self.path.stat().st_size + content_bytes * 12 + count * 4096 + 1_048_576)
        predicate = {}
        for alias in ("new", "old"):
            checks = []
            if source is not None:
                literal = self.connection.execute("SELECT quote(?)", (source,)).fetchone()[0]
                checks.append(f"{alias}.source={literal}")
            if id_prefix is not None:
                literal = self.connection.execute("SELECT quote(?)", (id_prefix,)).fetchone()[0]
                checks.append(f"substr({alias}.id,1,{len(id_prefix)})={literal}")
            predicate[alias] = " AND ".join(checks) or "1"
        try:
            self.connection.execute("BEGIN IMMEDIATE")
                                                                           
                                                                              
            duplicate = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            if duplicate is not None:
                if duplicate[0] != spec:
                    raise ValueError("Scoped BM25 index metadata changed during creation")
                self.connection.commit()
                return table
            self.connection.execute(f"CREATE VIRTUAL TABLE {table} USING fts5("
                                    "title,text,content='',tokenize='porter unicode61')")
            statements = (
                f"CREATE TRIGGER {table}_ai AFTER INSERT ON documents WHEN {predicate['new']} BEGIN "
                f"INSERT INTO {table}(rowid,title,text) VALUES(new.rowid,new.title,new.text); END",
                f"CREATE TRIGGER {table}_ad AFTER DELETE ON documents WHEN {predicate['old']} BEGIN "
                f"INSERT INTO {table}({table},rowid,title,text) VALUES('delete',old.rowid,old.title,old.text); END",
                f"CREATE TRIGGER {table}_au AFTER UPDATE ON documents BEGIN "
                f"INSERT INTO {table}({table},rowid,title,text) SELECT 'delete',old.rowid,old.title,old.text "
                f"WHERE {predicate['old']}; "
                f"INSERT INTO {table}(rowid,title,text) SELECT new.rowid,new.title,new.text "
                f"WHERE {predicate['new']}; END",
            )
            for statement in statements:
                self.connection.execute(statement)
            self.connection.execute(f"INSERT INTO {table}(rowid,title,text) "
                                    "SELECT rowid,title,text FROM documents d WHERE " + where, parameters)
            self.connection.execute("INSERT INTO metadata(key,value) VALUES(?,?)", (key, spec))
            self.budget.check()
            self.connection.commit()
            self.budget.check()
        except BaseException:
            self.connection.rollback()
            raise
        return table

    @staticmethod
    def _validate_scope(source, id_prefix):
        for label, value in (("source", source), ("id_prefix", id_prefix)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{label} must be a string")
            if value is not None and "\x00" in value:
                raise ValueError(f"{label} cannot contain a NUL character")
        return source or None, id_prefix or None

    @staticmethod
    def _id_filter(values, label):
        if values is None:
            return None
        if isinstance(values, str):
            raise ValueError(f"{label} must be a collection of document IDs")
        result = set(values)
        if any(not isinstance(value, str) or not value for value in result):
            raise ValueError(f"{label} needs nonempty string IDs")
        return result

    def _collect_hits(self, rows) -> list[dict]:
        pass                                                                    
        result, used = [], 0
        for row in rows:
            doc = self._document(row)
            doc.update(score=-row["rank"], retriever="bm25")
            size = len(json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode())
            if used + size > _MAX_RESULT_BYTES:
                break
            used += size
            result.append(doc)
        self.budget.check()
        return result

    def _canonical_ranked_rows(self, hits):
        for hit in hits:
            with closing(self.connection.execute("SELECT * FROM documents WHERE rowid=?", (hit["rowid"],))) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise ValueError("Candidate corpus changed during BM25 search")
            yield {**dict(row), "rank": hit["rank"]}

    def _search_allowed(self, query, *, k, title_weight, allowed_ids, exclude_ids, source, id_prefix):
        if len(allowed_ids) > _MAX_ALLOWED_DOCUMENTS:
            raise ValueError("Scoped candidate BM25 is limited to 1000 allowed documents")
        conditions, parameters = self._scope_conditions(source, id_prefix)
        conditions.append("d.id IN (SELECT value FROM json_each(?))")
        parameters.append(json.dumps(sorted(allowed_ids)))
        if exclude_ids:
            conditions.append("d.id NOT IN (SELECT value FROM json_each(?))")
            parameters.append(json.dumps(sorted(exclude_ids)))
        where = " AND ".join(conditions)
        with closing(self.connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(length(CAST(id AS BLOB))+length(CAST(title AS BLOB))"
                "+length(CAST(text AS BLOB))+length(CAST(metadata AS BLOB))),0) "
                "FROM documents d WHERE " + where, parameters)) as cursor:
            count, public_bytes = cursor.fetchone()
        if not count:
            return []
        if public_bytes > _MAX_CANDIDATE_BYTES:
            raise ValueError("Scoped candidate BM25 exceeds the 64 MB public-document bound")
        telemetry = self.budget.check()
        if telemetry["process_peak_rss_bytes"] + public_bytes * 12 + 32_000_000 >= self.budget.ram_limit:
            raise BudgetExceeded("Scoped candidate BM25 would exceed the process RAM budget")
                                                                             
                                                                             
                                                                               
        with closing(sqlite3.connect(":memory:")) as scope:
            scope.row_factory = sqlite3.Row
            scope.execute("PRAGMA cache_size=-32768")
            scope.execute("PRAGMA temp_store=MEMORY")
            scope.execute("CREATE VIRTUAL TABLE candidate_fts USING fts5(title,text,content='',tokenize='porter unicode61')")
            scope.execute("CREATE TABLE candidate_ids(rowid INTEGER PRIMARY KEY,id TEXT NOT NULL)")
            with closing(self.connection.execute("SELECT rowid,id,title,text FROM documents d WHERE " + where,
                                                  parameters)) as cursor:
                for row in cursor:
                    scope.execute("INSERT INTO candidate_fts(rowid,title,text) VALUES(?,?,?)",
                                  (row["rowid"], row["title"], row["text"]))
                    scope.execute("INSERT INTO candidate_ids VALUES(?,?)", (row["rowid"], row["id"]))
            scope.commit()
            self.budget.check()
            hits = scope.execute("SELECT candidate_fts.rowid,bm25(candidate_fts,?,1.0) AS rank "
                                 "FROM candidate_fts JOIN candidate_ids c ON c.rowid=candidate_fts.rowid "
                                 "WHERE candidate_fts MATCH ? ORDER BY rank,c.id LIMIT ?", (title_weight, query, k))
            with closing(hits):
                return self._collect_hits(self._canonical_ranked_rows(hits))

    def search(self, question: str, *, k: int = 20, title_weight: float = 2.0,
               allowed_ids: set[str] | None = None, exclude_ids: set[str] | None = None,
               source: str | None = None, id_prefix: str | None = None) -> list[dict]:
        if k < 1:
            return []
        if isinstance(k, bool) or not isinstance(k, int) or k > 1000:
            raise ValueError("k must be an integer no larger than 1000")
        if isinstance(title_weight, bool) or not math.isfinite(title_weight) or title_weight <= 0:
            raise ValueError("title_weight must be finite and positive")
        source, id_prefix = self._validate_scope(source, id_prefix)
        allowed_ids = self._id_filter(allowed_ids, "allowed_ids")
        exclude_ids = self._id_filter(exclude_ids, "exclude_ids")
        if allowed_ids is not None and not allowed_ids:
            return []
        terms = list(dict.fromkeys(tokenize(question)))[:64]
        if not terms:
            return []
        query = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        if allowed_ids is not None:
            return self._search_allowed(query, k=k, title_weight=float(title_weight), allowed_ids=allowed_ids,
                                        exclude_ids=exclude_ids, source=source, id_prefix=id_prefix)
        table = self.ensure_bm25_scope(source=source, id_prefix=id_prefix)
        scope_conditions, scope_parameters = self._scope_conditions(source, id_prefix)
        conditions, parameters = [f"{table} MATCH ?", *scope_conditions], [query, *scope_parameters]
        if exclude_ids:
            conditions.append("d.id NOT IN (SELECT value FROM json_each(?))")
            parameters.append(json.dumps(sorted(exclude_ids)))
                                                                               
                                                                              
                                                                             
        parameters.append(k)
        sql = f"""SELECT d.rowid,d.id,bm25({table},{float(title_weight)},1.0) AS rank
                  FROM {table} JOIN documents d ON d.rowid={table}.rowid
                  WHERE {' AND '.join(conditions)} ORDER BY rank,d.id LIMIT ?"""
        with closing(self.connection.execute(sql, parameters)) as cursor:
            return self._collect_hits(self._canonical_ranked_rows(cursor))
