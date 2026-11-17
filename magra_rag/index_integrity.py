                                                                             

                                                                              
                                                                                
                                                                     
   

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
from collections.abc import Mapping


_SCOPE_FIELDS = {"query_id", "source", "id_prefix", "allowed_ids", "exclude_ids"}
_ROW_PAGE = 256
_BLOB_PAGE_BYTES = 4_194_304
_MAX_VECTOR_BYTES = 1_048_576
_MAX_OR_GROUPS = 128                                                              


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _json_hash(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _ids(values, name: str, *, allow_none: bool):
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be a collection of document IDs")
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a collection of document IDs")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a collection of document IDs") from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} requires nonempty string document IDs")
    return sorted(set(values))


def _normalize_scope(scope: Mapping) -> dict:
    if not isinstance(scope, Mapping) or set(scope) - _SCOPE_FIELDS:
        raise ValueError("Dense scopes must contain only query_id and search filters")
    query_id = scope.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("Dense scopes require nonempty string query_id values")
    source, prefix = scope.get("source"), scope.get("id_prefix")
    if source is not None and (not isinstance(source, str) or not source):
        raise ValueError("source must be a nonempty string or None")
    if prefix is not None and not isinstance(prefix, str):
        raise ValueError("id_prefix must be a string or None")
    return {"query_id": query_id, "source": source, "id_prefix": prefix,
            "allowed_ids": _ids(scope.get("allowed_ids"), "allowed_ids", allow_none=True),
            "exclude_ids": _ids(scope.get("exclude_ids", []), "exclude_ids", allow_none=False)}


def effective_query_scope(query: Mapping) -> dict:
    pass                                                                         

                                                                           
                                                                          
                                                                         
       
    if not isinstance(query, Mapping):
        raise ValueError("A public query mapping is required")
    query_id = query.get("id", query.get("query_id"))
    allowed = query.get("allowed_doc_ids", query.get("context_doc_ids"))
    if query.get("setting") == "provided_context" and allowed is None:
        raise ValueError("Provided-context track requires explicit candidate document IDs")
    excluded = _ids(query.get("exclude_doc_ids", []), "exclude_doc_ids", allow_none=False)
    if query.get("source") == "arguana":
        excluded.append(query_id)
    return _normalize_scope({
        "query_id": query_id, "source": query.get("corpus_source"),
        "id_prefix": "hotpotqa:wiki:" if query.get("source") == "hotpotqa"
        and query.get("setting") == "full_corpus" else None,
        "allowed_ids": allowed, "exclude_ids": excluded,
    })


def _union_groups(scopes: list[dict]) -> list[dict]:
    pass                                                                            
    groups = {}
    for scope in scopes:
        key = scope["source"], scope["id_prefix"]
        group = groups.setdefault(key, {"limited": set(), "unrestricted_exclusions": None})
        excluded = set(scope["exclude_ids"])
        if scope["allowed_ids"] is None:
            if group["unrestricted_exclusions"] is None:
                group["unrestricted_exclusions"] = excluded
            else:
                group["unrestricted_exclusions"].intersection_update(excluded)
        else:
            group["limited"].update(set(scope["allowed_ids"]) - excluded)
    merged = []
    for (source, prefix), group in sorted(groups.items(), key=lambda item: _json(item[0])):
        excluded = group["unrestricted_exclusions"]
        if excluded is None:
            allowed, excluded = sorted(group["limited"]), []
        else:
            allowed, excluded = None, sorted(excluded - group["limited"])
        if allowed == []:
            continue
        merged.append({"source": source, "id_prefix": prefix,
                       "allowed_ids": allowed, "exclude_ids": excluded})
    return merged


def _where(groups: list[dict]) -> tuple[str, list]:
    if not groups:
        return "0", []
    if len(groups) > _MAX_OR_GROUPS:
                                                                             
                                                                               
        return """EXISTS (
            SELECT 1 FROM json_each(?) AS scope
            WHERE (json_extract(scope.value,'$.source') IS NULL
                   OR d.source=json_extract(scope.value,'$.source'))
              AND (json_extract(scope.value,'$.id_prefix') IS NULL
                   OR substr(d.id,1,length(json_extract(scope.value,'$.id_prefix')))
                      =json_extract(scope.value,'$.id_prefix'))
              AND (json_type(scope.value,'$.allowed_ids')='null'
                   OR d.id IN (SELECT value FROM json_each(scope.value,'$.allowed_ids')))
              AND d.id NOT IN (SELECT value FROM json_each(scope.value,'$.exclude_ids'))
        )""", [_json(groups)]
    predicates, parameters = [], []
    for group in groups:
        clauses = []
        if group["source"] is not None:
            clauses.append("d.source=?")
            parameters.append(group["source"])
        if group["id_prefix"] is not None:
            clauses.append("substr(d.id,1,?)=?")
            parameters.extend([len(group["id_prefix"]), group["id_prefix"]])
        if group["allowed_ids"] is not None:
            clauses.append("d.id IN (SELECT value FROM json_each(?))")
            parameters.append(_json(group["allowed_ids"]))
        if group["exclude_ids"]:
            clauses.append("d.id NOT IN (SELECT value FROM json_each(?))")
            parameters.append(_json(group["exclude_ids"]))
        predicates.append("(" + " AND ".join(clauses or ["1"]) + ")")
    return "(" + " OR ".join(predicates) + ")", parameters


def _frame(digest, value):
    if isinstance(value, str):
        value = value.encode()
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _validate_row(row, model_id, dimensions):
    doc_id, document_hash, vector_id, model, vector_hash, size, blob_type, blob_bytes = row
    if not isinstance(doc_id, str) or not doc_id or not isinstance(document_hash, str) or not document_hash:
        raise ValueError("Dense index has malformed canonical document identities")
    if vector_id is None:
        raise ValueError(f"Dense index incomplete: missing vector for {doc_id}")
    if model != model_id:
        raise ValueError(f"Dense index has a wrong-model vector for {doc_id}")
    if vector_hash != document_hash:
        raise ValueError(f"Dense index has a stale content hash for {doc_id}")
    if type(size) is not int or not 0 < size <= _MAX_VECTOR_BYTES // 4:
        raise ValueError(f"Dense index has invalid dimensions for {doc_id}")
    if dimensions is not None and size != dimensions:
        raise ValueError(f"Dense index has inconsistent or unexpected dimensions for {doc_id}")
    if blob_type != "blob" or blob_bytes != size * 4:
        raise ValueError(f"Dense index has invalid float32 blob length/type for {doc_id}")
    return size


def _blob_batches(rows):
    batch, size = [], 0
    for row in rows:
        if batch and size + row[7] > _BLOB_PAGE_BYTES:
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += row[7]
    if batch:
        yield batch


def dense_index_fingerprint(store, model_id: str, *, scopes, expected_dimensions=None) -> dict:
    pass                                                                        

                                                                                 
                                                                               
                                                                             
                                                                          

                                                                             
                                                                                
                                                                                
       
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("A nonempty dense model identity is required")
    if expected_dimensions is not None and (
            type(expected_dimensions) is not int or not 0 < expected_dimensions <= _MAX_VECTOR_BYTES // 4):
        raise ValueError("expected_dimensions must be a positive bounded integer or None")
    if isinstance(scopes, (str, bytes, Mapping)):
        raise ValueError("scopes must be a sequence of query scopes")
    try:
        normalized = [_normalize_scope(scope) for scope in scopes]
    except TypeError as exc:
        raise ValueError("scopes must be a sequence of query scopes") from exc
    if len({scope["query_id"] for scope in normalized}) != len(normalized):
        raise ValueError("Dense scopes require unique query IDs")
    normalized.sort(key=lambda scope: scope["query_id"])
    groups = _union_groups(normalized)
    where, parameters = _where(groups)
    scopes_hash = _json_hash({"format": "magra-dense-scopes-v1", "scopes": normalized})
    if store.connection.in_transaction:
        raise ValueError("Commit or roll back SQL changes before fingerprinting a dense index")
    path = Path(store.path).resolve()
    universe = hashlib.sha256(b"magra-dense-universe-v1\0")
    vectors = hashlib.sha256(b"magra-dense-vectors-v1\0")
    _frame(vectors, model_id)
    _frame(vectors, "<f4")
    dimensions, count, vector_bytes, last_id = expected_dimensions, 0, 0, None
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute("PRAGMA mmap_size=0")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("BEGIN")
        try:
            while True:
                page_where, page_parameters = where, list(parameters)
                if last_id is not None:
                    page_where += " AND d.id>?"
                    page_parameters.append(last_id)
                page_parameters.append(_ROW_PAGE)
                                                                             
                                                                      
                sql = ("SELECT d.id,d.content_hash,v.doc_id,v.model,v.content_hash,v.dimensions,"
                       "typeof(v.embedding),length(v.embedding) FROM documents d "
                       "INDEXED BY sqlite_autoindex_documents_1 "
                       "LEFT JOIN vectors v ON v.doc_id=d.id WHERE " + page_where + " ORDER BY d.id LIMIT ?")
                with closing(connection.execute(sql, page_parameters)) as cursor:
                    rows = cursor.fetchall()
                if not rows:
                    break
                last_id = rows[-1][0]
                for row in rows:
                    dimensions = _validate_row(row, model_id, dimensions)
                for batch in _blob_batches(rows):
                    ids = [row[0] for row in batch]
                    with closing(connection.execute(
                            "SELECT doc_id,embedding FROM vectors WHERE doc_id IN (SELECT value FROM json_each(?))",
                            [_json(ids)])) as cursor:
                        blobs = dict(cursor.fetchall())
                    for row in batch:
                        doc_id, document_hash, _, model, vector_hash, size, _, blob_bytes = row
                        blob = blobs.pop(doc_id, None)
                        if not isinstance(blob, bytes) or len(blob) != blob_bytes:
                            raise ValueError("Dense index vector changed within its read snapshot")
                        if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", blob)):
                            raise ValueError(f"Dense index has nonfinite float32 values for {doc_id}")
                        _frame(universe, doc_id)
                        _frame(universe, document_hash)
                        for field in (doc_id, model, vector_hash):
                            _frame(vectors, field)
                        _frame(vectors, struct.pack(">Q", size))
                        _frame(vectors, blob)
                        count += 1
                        vector_bytes += blob_bytes
        finally:
            connection.rollback()
    metadata = {
        "format": "magra-dense-index-v1", "model_id": model_id, "dtype": "<f4",
        "scopes_hash": scopes_hash, "universe_hash": universe.hexdigest(),
        "vectors_hash": vectors.hexdigest(), "document_count": count,
        "vector_bytes": vector_bytes, "dimensions": dimensions if count else None,
        "expected_dimensions": expected_dimensions, "query_count": len(normalized),
        "scope_group_count": len(groups), "status": "complete",
    }
    metadata["fingerprint_hash"] = _json_hash(metadata)
    return metadata
