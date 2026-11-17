                                                                       

                                                                              
                                                                             
                                                                               
   

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path

from .datasets import (BEIR_DATASETS, HOTPOT_DATASETS, _beir_identity,
                       iter_beir_documents, iter_hotpot_documents,
                       iter_hotpot_wikipedia, sha256_file)


RECEIPT_PREFIX = "import_receipt:"


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _acquisition(path: Path, supplied) -> tuple[dict | None, Path]:
    if isinstance(supplied, Mapping):
        return dict(supplied), path if path.is_dir() else path.parent
    if supplied is not None:
        manifest_path = Path(supplied)
    else:
        candidates = ([path / "acquisition.json"] if path.is_dir() else
                      [path.with_suffix(path.suffix + ".acquisition.json"), path.parent / "acquisition.json"])
        manifest_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if manifest_path is None:
            return None, path if path.is_dir() else path.parent
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Acquisition manifest must be an object")
    return value, manifest_path.parent


def _verify_acquisition(manifest: dict | None, base: Path, source: Path) -> bool:
    pass                                                                             
    if manifest is None:
        return False
    pinned = False
    base = base.resolve()
    files = manifest.get("files_sha256", {})
    if not isinstance(files, dict):
        raise ValueError("Invalid acquisition file-hash mapping")
    for relative, expected in files.items():
        if not isinstance(relative, str) or "\\" in relative or ":" in relative:
            raise ValueError("Unsafe acquisition manifest path")
        target = (base / relative).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise ValueError("Acquisition manifest references a missing or unsafe source file")
        if sha256_file(target) != expected:
            raise ValueError("Acquisition source hash mismatch")
        if source.is_dir() or target == source.resolve():
            pinned = True
    if source.is_file():
        expected = manifest.get("output_sha256", manifest.get("download_sha256", manifest.get("source_sha256")))
        if expected is not None:
            if manifest.get("output_filename", source.name) != source.name or sha256_file(source) != expected:
                raise ValueError("Acquisition source hash mismatch")
            pinned = True
    return pinned


def _source_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    files = {p.relative_to(path).as_posix(): sha256_file(p)
             for p in sorted(path.rglob("*")) if p.is_file() and p.name != "acquisition.json"}
    return _hash(files)


def _documents(format: str, path: Path):
    if format == "beir":
        yield from iter_beir_documents(path)
    elif format == "hotpot_context":
        yield from iter_hotpot_documents(path)
    elif format == "hotpot_wiki":
        yield from iter_hotpot_wikipedia(path)
    elif format == "jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError("Unknown corpus import format")


def _document_hash(document: Mapping) -> str:
    content = hashlib.sha256((document.get("title", "") + "\0" + document["text"]).encode()).hexdigest()
    return _hash([document["id"], content, document.get("metadata", {})])


def _namespace(store, *, source: str | None = None, prefix: str | None = None) -> tuple[int, str]:
    clauses, parameters = [], []
    if source is not None:
        clauses.append("source=?")
        parameters.append(source)
    if prefix is not None:
        clauses.append("substr(id,1,?)=?")
        parameters.extend([len(prefix), prefix])
    sql = "SELECT id,title,text,metadata,content_hash FROM documents"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    count, accumulator = 0, 0
    for row in store.connection.execute(sql, parameters):
        actual_content = hashlib.sha256((row["title"] + "\0" + row["text"]).encode()).hexdigest()
        if actual_content != row["content_hash"]:
            raise ValueError("SQL document content hash is stale or corrupted")
        digest = _hash([row["id"], actual_content, json.loads(row["metadata"])])
        accumulator = (accumulator + int(digest, 16)) % (1 << 256)
        count += 1
    return count, f"{accumulator:064x}"


def record_import(store, *, format: str, source_path: str | Path,
                  imported_count: int | None = None, acquisition=None) -> dict:
    pass                                                                        

                                                                                
                                                                                 
                                                                                 
                                                                      
       
    if getattr(store, "readonly", False):
        raise ValueError("Cannot record an import in a read-only store")
    path = Path(source_path).resolve()
    if not path.exists():
        raise ValueError("Import source is missing")
    store.budget.contains(path)
    if imported_count is not None and (type(imported_count) is not int or imported_count < 0):
        raise ValueError("Imported count must be a nonnegative integer")
    manifest, manifest_base = _acquisition(path, acquisition)
    pinned = _verify_acquisition(manifest, manifest_base, path)
    source = _beir_identity(path)[0] if format == "beir" else "hotpotqa" if format.startswith("hotpot_") else None
    expected_count = BEIR_DATASETS[source]["documents"] if source in BEIR_DATASETS else None
    if source in BEIR_DATASETS:
        if (not pinned or not manifest or manifest.get("dataset") != source
                or manifest.get("complete_corpus") is not True
                or manifest.get("archive_md5") != BEIR_DATASETS[source]["md5"]):
            raise ValueError("Known BEIR corpus requires its verified complete acquisition manifest")
        if manifest.get("corpus_documents") != expected_count:
            raise ValueError("BEIR acquisition corpus count differs from catalog")
    if format == "hotpot_wiki":
        if (not pinned or not manifest or manifest.get("complete_corpus") is not True
                or manifest.get("archive_md5") != HOTPOT_DATASETS["wiki"]["md5"]):
            raise ValueError("Full Wikipedia requires its verified complete acquisition manifest")
        expected_count = manifest.get("corpus_documents")
        if type(expected_count) is not int or expected_count < 1:
            raise ValueError("Full Wikipedia acquisition requires an audited document count")
    count = unique_count = accumulator = 0
    sources: set[str] = set()
                                                                              
                                                                               
    context_seen = {} if format in {"hotpot_context", "jsonl"} else None
    for document in _documents(format, path):
        count += 1
        expected_hash = _document_hash(document)
        if context_seen is not None and document["id"] in context_seen:
            if context_seen[document["id"]] != expected_hash:
                raise ValueError("Conflicting context document IDs in import source")
            continue
        if context_seen is not None:
            context_seen[document["id"]] = expected_hash
        actual = store.get(document["id"])
        if actual is None or _document_hash(actual) != expected_hash:
            raise ValueError("Import is incomplete or SQL content differs from source")
        sources.add(str(document.get("metadata", {}).get("source", "")))
        unique_count += 1
        accumulator = (accumulator + int(expected_hash, 16)) % (1 << 256)
    if imported_count is not None and count != imported_count:
        raise ValueError("Accepted import count differs from complete source")
    if count < 1 or (expected_count is not None and unique_count != expected_count):
        raise ValueError("Import source is incomplete or has an unexpected document count")
    fingerprint = f"{accumulator:064x}"
    if format == "beir":
        sql_count, sql_fingerprint = _namespace(store, source=source)
        if (sql_count, sql_fingerprint) != (unique_count, fingerprint):
            raise ValueError("BEIR SQL corpus differs from the complete imported source")
    elif format == "hotpot_wiki":
        sql_count, sql_fingerprint = _namespace(store, source="hotpotqa", prefix="hotpotqa:wiki:")
        if (sql_count, sql_fingerprint) != (unique_count, fingerprint):
            raise ValueError("Full Wikipedia SQL corpus differs from the complete imported source")
    receipt = {"format": "magra-import-receipt-v1", "import_format": format,
               "source": source, "document_sources": sorted(sources), "source_path": str(path),
               "source_sha256": _source_hash(path), "acquisition_sha256": _hash(manifest) if manifest else None,
               "acquisition_source_pinned": pinned, "source_records": count,
               "documents": unique_count, "namespace_fingerprint": fingerprint,
               "catalog_documents": expected_count, "complete": True, "sql_corpus_hash": store.corpus_hash()}
    receipt_hash = _hash(receipt)
    key = RECEIPT_PREFIX + receipt_hash
    store.budget.check(len(json.dumps(receipt).encode()) * 4 + 65536 + store.path.stat().st_size)
    with store.connection:
        store.connection.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES(?,?)",
                                 (key, json.dumps(receipt, sort_keys=True, ensure_ascii=False)))
    return {**receipt, "receipt_hash": receipt_hash}


def _receipts(store) -> list[dict]:
    result = []
    for row in store.connection.execute("SELECT key,value FROM metadata WHERE substr(key,1,?)=?",
                                         (len(RECEIPT_PREFIX), RECEIPT_PREFIX)):
        receipt = json.loads(row["value"])
        if row["key"] != RECEIPT_PREFIX + _hash(receipt) or receipt.get("format") != "magra-import-receipt-v1":
            raise ValueError("Import receipt integrity check failed")
        if receipt.get("complete") is True:
            result.append(receipt)
    return result


def audit_corpus(store, queries: Iterable[Mapping], *, strict: bool = True) -> dict:
    pass                                                                         

                                                                                  
                                                                               
                                                                        
       
    failures = []
    receipts = _receipts(store)
    required_beir = set()
    full_wiki = False
    context_ids: set[str] = set()
    query_count = 0
    for query in queries:
        query_count += 1
        source = query.get("corpus_source", query.get("source"))
        if source in BEIR_DATASETS:
            required_beir.add(source)
        if query.get("source") == "hotpotqa":
            if query.get("setting") == "full_corpus":
                full_wiki = True
            elif query.get("setting") == "provided_context":
                candidates = query.get("allowed_doc_ids", query.get("context_doc_ids"))
                if not isinstance(candidates, list) or not candidates:
                    failures.append("Hotpot provided-context query lacks a complete candidate list")
                else:
                    if any(not isinstance(value, str) or not value.startswith("hotpotqa:context:") for value in candidates):
                        failures.append("Hotpot provided-context candidates use the wrong corpus namespace")
                    else:
                        context_ids.update(candidates)
    scopes = {}
    for source in sorted(required_beir):
        count, fingerprint = _namespace(store, source=source)
        expected = BEIR_DATASETS[source]["documents"]
        matching = [r for r in receipts if r["import_format"] == "beir" and r["source"] == source
                    and r["documents"] == count and r["namespace_fingerprint"] == fingerprint
                    and r["acquisition_source_pinned"]]
        if count != expected:
            failures.append(f"{source}: SQL corpus count differs from the complete catalog")
        if not matching:
            failures.append(f"{source}: no matching completed import receipt")
        scopes[source] = {"documents": count, "expected_documents": expected, "completed_import": bool(matching)}
    if full_wiki:
        count, fingerprint = _namespace(store, source="hotpotqa", prefix="hotpotqa:wiki:")
        matching = [r for r in receipts if r["import_format"] == "hotpot_wiki"
                    and r["documents"] == count and r["namespace_fingerprint"] == fingerprint
                    and r["acquisition_source_pinned"]]
        if not count or not matching:
            failures.append("Hotpot full-corpus requires the complete independent Wikipedia import receipt")
        scopes["hotpotqa:wiki"] = {"documents": count, "completed_import": bool(matching)}
    missing = 0
    for doc_id in context_ids:
        if store.get(doc_id) is None:
            missing += 1
    if missing:
        failures.append("Hotpot provided-context corpus is missing candidate documents")
    if context_ids:
        scopes["hotpotqa:context"] = {"required_documents": len(context_ids), "missing_documents": missing}
    result = {"format": "magra-corpus-audit-v1", "status": "failed" if failures else "verified",
              "query_count": query_count, "sql_documents": len(store), "sql_corpus_hash": store.corpus_hash(),
              "completed_import_receipts": len(receipts), "scopes": scopes, "failures": failures,
              "scope": "Known benchmark completeness and selected provided contexts; owned data has no external catalog count."}
    if strict and failures:
        raise ValueError("Corpus completeness audit failed: " + "; ".join(failures))
    return result
