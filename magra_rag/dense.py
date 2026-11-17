                                                                                

from __future__ import annotations

from contextlib import closing
import heapq
import hashlib
import importlib.metadata
import json
import os

from .storage import DocumentStore


DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
QWEN_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


def retrieval_model_identity(model: str, revision: str | None = None) -> str:
    pass                                                                         
    if model in {QWEN_DENSE_MODEL, QWEN_RERANKER_MODEL}:
        from .qwen import qwen_model_identity
        return qwen_model_identity(model, revision)
    if model.startswith("Qwen/"):
        raise ValueError("Only the native Qwen3 0.6B retrieval adapters are supported")
    return model + "@" + (revision or "UNPINNED")


def reranker_runtime_metadata(model: str, revision: str | None = None, *, device="cpu") -> dict:
    pass                                                                       
    if model == QWEN_DENSE_MODEL:
        raise ValueError("An embedding model cannot be used as a reranker")
    if device == "cpu":
        return {"reranker_model": retrieval_model_identity(model, revision)}
    if device != "cuda:0" or model != QWEN_RERANKER_MODEL:
        raise ValueError("CUDA reranking requires explicit cuda:0 and Qwen/Qwen3-Reranker-0.6B")
    from .qwen import qwen_reranker_execution, qwen_reranker_identity
    identity = qwen_reranker_identity(model, revision, device=device)
    packages = ("torch", "transformers", "tokenizers", "safetensors", "psutil", "nvidia-ml-py")
    try:
        versions = {name: importlib.metadata.version(name) for name in packages}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("CUDA reranking requires installed torch/transformers/tokenizers/safetensors/psutil/nvidia-ml-py dependencies") from error
    execution = {"format": "magra-reranker-execution-v1", "model_identity": identity,
                 "policy": qwen_reranker_execution(model, revision, device=device), "packages": versions}
    fingerprint = hashlib.sha256(json.dumps(execution, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"reranker_model": identity, "reranker_execution": execution,
            "reranker_execution_hash": fingerprint}


def _cpu_threads(torch) -> int:
    pass                                                                 
    count = min(4, os.cpu_count() or 1)
    torch.set_num_threads(count)
    return count


def _source_names(sources):
    if sources is None:
        return None
    names = [sources] if isinstance(sources, str) else list(sources)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("sources must contain nonempty source names")
    return sorted(set(names))


def _candidate_filters(*, allowed_ids=None, exclude_ids=None, source=None, id_prefix=None):
    pass                                                                        
    clauses, parameters = [], []
    if source is not None:
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a nonempty string")
        clauses.append("d.source=?")
        parameters.append(source)
    if id_prefix is not None:
        if not isinstance(id_prefix, str):
            raise ValueError("id_prefix must be a string")
        clauses.append("substr(d.id,1,?)=?")
        parameters.extend([len(id_prefix), id_prefix])
    for ids, exclude in ((allowed_ids, False), (exclude_ids, True)):
        if ids is None:
            continue
        if isinstance(ids, str):
            raise ValueError("Document ID filters must be collections, not strings")
        ids = list(ids)
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
            raise ValueError("Document ID filters require nonempty string IDs")
        clauses.append("d.id " + ("NOT IN" if exclude else "IN") + " (SELECT value FROM json_each(?))")
        parameters.append(json.dumps(sorted(set(ids))))
    return clauses, parameters


class DenseRetriever:
    def __init__(self, store: DocumentStore, *, model: str = DEFAULT_DENSE_MODEL,
                 revision: str | None = None, device: str = "cpu", batch_size: int | None = None):
        self.is_qwen = model == QWEN_DENSE_MODEL
        if model == QWEN_RERANKER_MODEL:
            raise ValueError("A reranker cannot be used as an embedding model")
        model_id = retrieval_model_identity(model, revision)
        if batch_size is None:
            batch_size = 1 if self.is_qwen else 16
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
            raise ValueError("Embedding batch size must be 1..128")
        if device != "cpu":
            raise ValueError("This reference implementation uses CPU for bounded VRAM")
        cache = store.budget.root / "models"
        os.environ["HF_HOME"] = str(cache)
        store.budget.check(1_000_000_000)
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install optional dense dependencies: pip install -e .[dense]") from exc
        self.np, self.store, self.batch_size = np, store, batch_size
        if self.is_qwen:
            from .qwen import QwenEncoder
            self.encoder = QwenEncoder(store, model, revision, batch_size=batch_size)
            self.cpu_threads = self.encoder.cpu_threads
            if self.encoder.model_id != model_id:
                raise ValueError("Qwen encoder protocol identity mismatch")
        else:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install optional dense dependencies: pip install -e .[dense]") from exc
            self.cpu_threads = _cpu_threads(torch)
            self.encoder = SentenceTransformer(model, revision=revision, device="cpu",
                                               cache_folder=str(cache), trust_remote_code=False)
        self.model_id = model_id
        self.revision = revision
        store.budget.check()

    def build(self, sources=None) -> dict:
        pass                                                             

                                                                             
                                                                               
                                                                               
                                                                          
           
        selected_sources = _source_names(sources)
        filters, parameters = [], []
        if selected_sources is not None:
            filters.append("d.source IN (SELECT value FROM json_each(?))")
            parameters.append(json.dumps(selected_sources))
            with closing(self.store.connection.execute(
                    "SELECT DISTINCT d.source FROM documents d WHERE " + filters[0], parameters)) as cursor:
                present_sources = {row[0] for row in cursor.fetchall()}
            unknown = set(selected_sources) - present_sources
            if unknown:
                raise ValueError("Requested dense build sources are absent from SQL: " + ", ".join(sorted(unknown)))
                                                                              
                                                                             
        conflict_filters = [*filters, "v.model!=?"]
        with closing(self.store.connection.execute(
                "SELECT v.model FROM documents d JOIN vectors v ON v.doc_id=d.id WHERE "
                + " AND ".join(conflict_filters), [*parameters, self.model_id])) as cursor:
            for (other_model,) in cursor:
                if self.is_qwen or other_model.startswith("Qwen/"):
                    raise ValueError("Embedding model conflicts with existing vectors; use a separate SQL store")
        stale = "(v.doc_id IS NULL OR v.model!=? OR v.content_hash!=d.content_hash)"
        filters.append(stale)
        parameters.append(self.model_id)
        count = 0
        last_id = None
        while True:
            page_filters, page_parameters = list(filters), list(parameters)
            if last_id is not None:
                page_filters.append("d.id>?")
                page_parameters.append(last_id)
            page_parameters.append(self.batch_size)
            sql = ("SELECT d.id,d.title,d.text,d.content_hash FROM documents d "
                   "INDEXED BY sqlite_autoindex_documents_1 "
                   "LEFT JOIN vectors v ON v.doc_id=d.id WHERE " + " AND ".join(page_filters)
                   + " ORDER BY d.id LIMIT ?")
            with closing(self.store.connection.execute(sql, page_parameters)) as cursor:
                rows = [tuple(row) for row in cursor.fetchall()]
            if not rows:
                break
            last_id = rows[-1][0]
            count += self._write(rows)
        with closing(self.store.connection.execute(
                "SELECT COUNT(*) FROM documents d LEFT JOIN vectors v ON v.doc_id=d.id WHERE "
                + " AND ".join(filters), parameters)) as cursor:
            remaining = cursor.fetchone()[0]
        if remaining:
            raise ValueError(f"Dense source universe changed during build: {remaining} documents still need embeddings")
        return {"embedded_documents": count, "model": self.model_id,
                "sources": selected_sources, "status": "complete", "cpu_threads": self.cpu_threads,
                "revision_pinned": self.revision is not None,
                "truncation": "encoder maximum sequence length; full text retained in SQL",
                "resources": self.store.budget.check()}

    def _write(self, rows):
        self.store.budget.check(len(rows) * 10000)
        embeddings = self.encoder.encode([title + "\n" + text for _, title, text, _ in rows],
                                         normalize_embeddings=True, batch_size=self.batch_size,
                                         show_progress_bar=False)
        if len(embeddings) != len(rows):
            raise ValueError("Encoder did not return one embedding per document")
        prepared = []
        for row, embedding in zip(rows, embeddings):
            v = self.np.asarray(embedding, dtype="<f4")
            if v.ndim != 1 or not len(v) or not bool(self.np.isfinite(v).all()):
                raise ValueError("Encoder returned a malformed or nonfinite document vector")
            prepared.append((row, len(v), v.tobytes()))
                                                                              
                                                           
        self.store.budget.check(2 * sum(len(blob) for _, _, blob in prepared) + len(rows) * 4096 + 65536)
        with self.store.connection:
            for row, dimensions, blob in prepared:
                self.store.connection.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?)",
                                              (row[0], self.model_id, row[3], dimensions, blob))
        self.store.budget.check()
        return len(rows)

    def index_fingerprint(self, *, scopes):
        pass                                                                  
        from .index_integrity import dense_index_fingerprint
        dimensions = 1024 if self.is_qwen else 384 if self.model_id.startswith(DEFAULT_DENSE_MODEL + "@") else None
        return dense_index_fingerprint(self.store, self.model_id, scopes=scopes,
                                       expected_dimensions=dimensions)

    def search(self, question: str, *, k: int = 20, allowed_ids=None,
               exclude_ids=None, source=None, id_prefix=None) -> list[dict]:
        if k < 1:
            return []
        where, params = _candidate_filters(allowed_ids=allowed_ids, exclude_ids=exclude_ids,
                                          source=source, id_prefix=id_prefix)
        stale = "(v.doc_id IS NULL OR v.model!=? OR v.content_hash!=d.content_hash)"
        with closing(self.store.connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(CASE WHEN " + stale + " THEN 1 ELSE 0 END),0) "
                "FROM documents d LEFT JOIN vectors v ON v.doc_id=d.id"
                + (" WHERE " + " AND ".join(where) if where else ""), [self.model_id, *params])) as cursor:
            candidates, missing = cursor.fetchone()
        if missing:
            raise ValueError(f"Dense index incomplete or stale: {missing} matching documents need embeddings")
        if not candidates:
            return []
        if self.is_qwen:
            q = self.encoder.encode([question], normalize_embeddings=True,
                                    show_progress_bar=False, is_query=True)[0]
        else:
            prefix = "Represent this sentence for searching relevant passages: "
            q = self.encoder.encode([prefix + question], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
        where += ["v.model=?", "v.content_hash=d.content_hash"]
        params.append(self.model_id)
        cursor = self.store.connection.execute(
            "SELECT v.doc_id,v.embedding FROM vectors v JOIN documents d ON d.id=v.doc_id WHERE "
            + " AND ".join(where), params)
        heap, scanned = [], 0
        with closing(cursor):
            while rows := cursor.fetchmany(1024):
                matrix = self.np.stack([self.np.frombuffer(row["embedding"], dtype="<f4") for row in rows])
                scores = matrix @ q
                for row, score in zip(rows, scores):
                    item = (float(score), row["doc_id"])
                    if len(heap) < k:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                scanned += len(rows)
                self.store.budget.check()
        if scanned != candidates:
            raise ValueError("Dense candidate universe changed while searching; retry explicitly")
        result = []
        for score, doc_id in sorted(heap, key=lambda t: (-t[0], t[1])):
            doc = self.store.get(doc_id)
            doc.update(score=score, retriever="dense")
            result.append(doc)
        return result


class CrossEncoderReranker:
    def __init__(self, store: DocumentStore, *,
                 model=DEFAULT_RERANKER_MODEL, revision=None, device="cpu"):
        if model == QWEN_DENSE_MODEL:
            raise ValueError("An embedding model cannot be used as a reranker")
        self.runtime_metadata = reranker_runtime_metadata(model, revision, device=device)
        self.model_id = self.runtime_metadata["reranker_model"]
        self.device = device
        self.store = store
        self.native = None
        if model == QWEN_RERANKER_MODEL:
            from .qwen import QwenReranker
            options = {"device": device} if device != "cpu" else {}
            self.native = QwenReranker(store, model, revision, **options)
            self.cpu_threads = self.native.cpu_threads
            if self.native.model_id != self.model_id:
                if device != "cpu":self.native.close()
                raise ValueError("Qwen reranker protocol identity mismatch")
            return
        cache = store.budget.root / "models"
        store.budget.check(1_000_000_000)
        os.environ["HF_HOME"] = str(cache)
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install optional dense dependencies: pip install -e .[dense]") from exc
        self.cpu_threads = _cpu_threads(torch)
        self.model = CrossEncoder(model, revision=revision, device="cpu",
                                  cache_folder=str(cache), trust_remote_code=False)
        store.budget.check()

    def close(self):
        if self.device != "cpu" and self.native is not None:
            self.native.close()

    def resource_observations(self):
        return self.native.resource_observations() if self.device != "cpu" and self.native is not None else None

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        if self.native is not None:
            return self.native.rerank(question, candidates)
        if not candidates:
            return []
        scores = self.model.predict([(question, d["title"] + "\n" + d["text"])
                                     for d in candidates], batch_size=8, show_progress_bar=False)
        results = [dict(d, rerank_score=float(score)) for d, score in zip(candidates, scores)]
        self.store.budget.check()
        return sorted(results, key=lambda d: (-d["rerank_score"], d["id"]))
