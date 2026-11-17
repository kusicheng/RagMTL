import hashlib
import json
import re

from .config import ContinuityError
from .store import bounded_name, hashed, limit_value


MAX_SOURCE_FILES = 1000
MAX_SOURCE_BYTES = 64_000_000
MAX_SOURCE_FILE_BYTES = 16_000_000
MAX_SOURCE_BLOCKS = 100_000
MAX_BLOCK_BYTES = 4_000_000
SOURCE_KINDS = frozenset({"module_docstring", "class_docstring", "function_docstring", "standalone_string", "comment"})
SOURCE_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS source_document_files (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        migration_id VARCHAR(191) NOT NULL,
        path_hash CHAR(64) CHARACTER SET ascii NOT NULL, path TEXT NOT NULL,
        original_sha256 CHAR(64) CHARACTER SET ascii NOT NULL,
        stripped_sha256 CHAR(64) CHARACTER SET ascii NOT NULL,
        semantic_sha256 CHAR(64) CHARACTER SET ascii NOT NULL,
        source_encoding VARCHAR(64) NOT NULL,
        original_bytes LONGBLOB NOT NULL, stripped_bytes LONGBLOB NOT NULL,
        original_byte_count BIGINT UNSIGNED NOT NULL, stripped_byte_count BIGINT UNSIGNED NOT NULL,
        block_count INT UNSIGNED NOT NULL,
        imported_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        UNIQUE KEY source_document_version (path_hash, original_sha256),
        INDEX source_document_migration (migration_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS source_documentation_blocks (
        source_file_id BIGINT UNSIGNED NOT NULL, ordinal INT UNSIGNED NOT NULL,
        kind VARCHAR(32) NOT NULL, symbol TEXT NOT NULL,
        symbol_hash CHAR(64) CHARACTER SET ascii NOT NULL,
        start_line INT UNSIGNED NOT NULL, start_col INT UNSIGNED NOT NULL,
        end_line INT UNSIGNED NOT NULL, end_col INT UNSIGNED NOT NULL,
        start_byte BIGINT UNSIGNED NOT NULL, end_byte BIGINT UNSIGNED NOT NULL,
        raw_bytes LONGBLOB NOT NULL, raw_sha256 CHAR(64) CHARACTER SET ascii NOT NULL,
        body_text LONGTEXT NOT NULL, preserve_reason VARCHAR(191) NULL,
        metadata_sha256 CHAR(64) CHARACTER SET ascii NOT NULL,
        PRIMARY KEY (source_file_id, ordinal),
        INDEX source_document_symbol (symbol_hash),
        FULLTEXT KEY source_document_text (body_text),
        CONSTRAINT continuity_source_blocks_file FOREIGN KEY (source_file_id) REFERENCES source_document_files(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin""",
)
FILE_FIELDS = ("id", "migration_id", "path", "original_sha256", "stripped_sha256", "semantic_sha256",
               "source_encoding", "original_bytes", "stripped_bytes", "original_byte_count", "stripped_byte_count", "block_count")
BLOCK_FIELDS = ("ordinal", "kind", "symbol", "start_line", "start_col", "end_line", "end_col",
                "start_byte", "end_byte", "raw_bytes", "raw_sha256", "body_text", "preserve_reason", "metadata_sha256")


def _need(value, code):
    if not value:
        raise ContinuityError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _checksum(value):
    _need(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None, "invalid_source_sha256")
    return value


def _text(value, maximum, code):
    _need(type(value) is str, code)
    try:
        _need(len(value.encode("utf-8")) <= maximum, code)
    except UnicodeError:
        raise ContinuityError(code) from None
    return value


def _block_checksum(row):
    values = {key: row[key] for key in BLOCK_FIELDS if key not in {"raw_bytes", "metadata_sha256"}}
    return _sha(json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _prepare(store, analyses):
    prepared, total, block_total, paths = [], 0, 0, set()
    for analysis in analyses:
        _need(type(analysis) is dict and len(prepared) < MAX_SOURCE_FILES, "source_import_batch_too_large_or_invalid")
        path = store.content_path(analysis["path"])
        _need(path not in paths, "duplicate_source_path")
        paths.add(path)
        original, stripped = analysis["original_bytes"], analysis["stripped_bytes"]
        _need(type(original) is bytes and type(stripped) is bytes
              and len(original) <= MAX_SOURCE_FILE_BYTES and len(stripped) <= MAX_SOURCE_FILE_BYTES,
              "bounded_original_and_stripped_source_bytes_required")
        total += len(original) + len(stripped)
        _need(total <= MAX_SOURCE_BYTES, "source_import_batch_too_large_or_invalid")
        encoding = bounded_name(analysis["encoding"], 64)
        try:
            original.decode(encoding)
            stripped.decode(encoding)
        except (LookupError, UnicodeError):
            raise ContinuityError("source_encoding_invalid") from None
        original_sha, stripped_sha = _checksum(analysis["original_sha256"]), _checksum(analysis["stripped_sha256"])
        semantic_sha = _checksum(analysis["semantic_sha256"])
        _need(_sha(original) == original_sha and _sha(stripped) == stripped_sha, "source_input_hash_mismatch")
        if "stripped_semantic_sha256" in analysis:
            _need(_checksum(analysis["stripped_semantic_sha256"]) == semantic_sha, "source_executable_semantics_changed")
        blocks = analysis["blocks"]
        _need(type(blocks) is list, "source_blocks_array_required")
        block_total += len(blocks)
        _need(block_total <= MAX_SOURCE_BLOCKS, "source_import_block_limit")
        rows = []
        for ordinal, block in enumerate(blocks):
            _need(type(block) is dict and type(block["ordinal"]) is int and block["ordinal"] == ordinal,
                  "source_block_order_required")
            kind = block["kind"]
            _need(type(kind) is str and kind in SOURCE_KINDS, "invalid_source_block_kind")
            symbol = _text(block["symbol"], 16_384, "invalid_source_symbol")
            _need(bool(symbol), "invalid_source_symbol")
            for key in ("start_line", "start_col", "end_line", "end_col", "start_byte", "end_byte"):
                _need(type(block[key]) is int and 0 <= block[key] < 2**32, "invalid_source_block_offset")
            _need(1 <= block["start_line"] <= block["end_line"]
                  and (block["start_line"] < block["end_line"] or block["start_col"] <= block["end_col"])
                  and 0 <= block["start_byte"] < block["end_byte"] <= len(original), "invalid_source_block_range")
            raw = block["raw_bytes"]
            _need(type(raw) is bytes and len(raw) <= MAX_BLOCK_BYTES
                  and raw == original[block["start_byte"]:block["end_byte"]]
                  and _sha(raw) == _checksum(block["sha256"]), "source_block_bytes_or_hash_mismatch")
            text = _text(block["text"], MAX_BLOCK_BYTES, "invalid_source_block_text")
            total += len(raw) + len(text.encode("utf-8"))
            _need(total <= MAX_SOURCE_BYTES, "source_import_batch_too_large_or_invalid")
            reason = block["preserve_reason"]
            if reason is not None:
                reason = bounded_name(reason)
            row = {key: block[key] for key in ("ordinal", "kind", "symbol", "start_line", "start_col", "end_line", "end_col", "start_byte", "end_byte")}
            row.update(raw_bytes=raw, raw_sha256=block["sha256"], body_text=text, preserve_reason=reason)
            row["metadata_sha256"] = _block_checksum(row)
            rows.append(row)
        prepared.append({"path": path, "path_hash": hashed(path), "original_sha256": original_sha,
                         "stripped_sha256": stripped_sha, "semantic_sha256": semantic_sha,
                         "source_encoding": encoding, "original_bytes": original, "stripped_bytes": stripped,
                         "original_byte_count": len(original), "stripped_byte_count": len(stripped),
                         "block_count": len(rows), "blocks": rows})
    return prepared


def initialize_source_schema(store):
    with store.transaction() as cursor:
        for statement in SOURCE_SCHEMA:
            cursor.execute(statement)
    return {"database": store.config.database, "source_schema_tables": 2}


def _read_file(cursor, *, path=None, original_sha256=None, source_id=None):
    statement = "SELECT " + ",".join(FILE_FIELDS) + " FROM source_document_files"
    if source_id is not None:
        _need(type(source_id) is int and source_id > 0, "invalid_source_file_id")
        cursor.execute(statement + " WHERE id=%s", (source_id,))
    elif original_sha256 is not None:
        cursor.execute(statement + " WHERE path_hash=%s AND original_sha256=%s", (hashed(path), _checksum(original_sha256)))
    else:
        cursor.execute(statement + " WHERE path_hash=%s AND path=%s ORDER BY id DESC LIMIT 1", (hashed(path), path))
    row = cursor.fetchone()
    if row is not None and path is not None:
        _need(row["path"] == path, "source_path_readback_mismatch")
    return row


def _read_blocks(cursor, source_id):
    cursor.execute("SELECT " + ",".join(BLOCK_FIELDS) + " FROM source_documentation_blocks WHERE source_file_id=%s ORDER BY ordinal", (source_id,))
    rows = cursor.fetchall()
    for row in rows:
        row["raw_bytes"] = bytes(row["raw_bytes"])
    return rows


def _verify_saved(row, blocks, expected=None):
    _need(row is not None, "source_document_missing")
    row = dict(row)
    _need(type(row["id"]) is int and row["id"] > 0
          and all(type(row[key]) is int and row[key] >= 0 for key in ("original_byte_count", "stripped_byte_count", "block_count"))
          and all(type(row[key]) is str for key in ("migration_id", "path", "original_sha256", "stripped_sha256", "semantic_sha256", "source_encoding")),
          "source_document_readback_mismatch")
    row["original_bytes"], row["stripped_bytes"] = bytes(row["original_bytes"]), bytes(row["stripped_bytes"])
    _need(len(row["original_bytes"]) == row["original_byte_count"] and len(row["stripped_bytes"]) == row["stripped_byte_count"]
          and _sha(row["original_bytes"]) == row["original_sha256"]
          and _sha(row["stripped_bytes"]) == row["stripped_sha256"]
          and len(blocks) == row["block_count"], "source_document_readback_mismatch")
    for ordinal, block in enumerate(blocks):
        _need(block["ordinal"] == ordinal and _sha(block["raw_bytes"]) == block["raw_sha256"]
              and block["raw_bytes"] == row["original_bytes"][block["start_byte"]:block["end_byte"]]
              and _block_checksum(block) == block["metadata_sha256"], "source_block_readback_mismatch")
    if expected is not None:
        _need(all(row[key] == value for key, value in expected.items() if key not in {"path_hash", "blocks"})
              and blocks == expected["blocks"], "source_document_readback_mismatch")
    return row


def import_source_documents(store, analyses, migration_id):
    migration_id = bounded_name(migration_id)
    prepared = _prepare(store, analyses)
    result = []
    with store.transaction() as cursor:
        for expected in prepared:
            saved = _read_file(cursor, path=expected["path"], original_sha256=expected["original_sha256"])
            inserted = saved is None
            if inserted:
                keys = ("path_hash", "path", "original_sha256", "stripped_sha256", "semantic_sha256", "source_encoding",
                        "original_bytes", "stripped_bytes", "original_byte_count", "stripped_byte_count", "block_count")
                cursor.execute("INSERT INTO source_document_files (migration_id," + ",".join(keys) + ") VALUES (" + ",".join(["%s"] * (len(keys) + 1)) + ")",
                               (migration_id, *(expected[key] for key in keys)))
                source_id = cursor.lastrowid
                for block in expected["blocks"]:
                    keys = ("source_file_id", *BLOCK_FIELDS)
                    values = (source_id, *(block[key] for key in BLOCK_FIELDS))
                    cursor.execute("INSERT INTO source_documentation_blocks (" + ",".join(keys) + ",symbol_hash) VALUES (" + ",".join(["%s"] * (len(keys) + 1)) + ")",
                                   (*values, hashed(block["symbol"])))
                saved = _read_file(cursor, source_id=source_id)
            source_id = saved["id"]
            _verify_saved(saved, _read_blocks(cursor, source_id), expected)
            if inserted:
                store._event(cursor, "source_documentation_imported", None,
                             {"migration_id": migration_id, "source_file_id": source_id, "path": expected["path"],
                              "original_sha256": expected["original_sha256"], "stripped_sha256": expected["stripped_sha256"],
                              "blocks": expected["block_count"]})
            result.append({"source_file_id": source_id, "path": expected["path"], "original_sha256": expected["original_sha256"],
                           "stripped_sha256": expected["stripped_sha256"], "blocks": expected["block_count"], "inserted": inserted})
    return result


def verify_source_documents(store, analyses):
    prepared = _prepare(store, analyses)
    result = []
    with store.transaction() as cursor:
        for expected in prepared:
            saved = _read_file(cursor, path=expected["path"], original_sha256=expected["original_sha256"])
            _need(saved is not None, "source_document_missing")
            saved = _verify_saved(saved, _read_blocks(cursor, saved["id"]), expected)
            result.append({"source_file_id": saved["id"], "path": saved["path"], "original_sha256": saved["original_sha256"],
                           "stripped_sha256": saved["stripped_sha256"], "blocks": saved["block_count"],
                           "original_bytes_verified": saved["original_byte_count"], "stripped_bytes_verified": saved["stripped_byte_count"]})
    return result


def source_file_versions(store, path=None, *, limit=100, before_id=None):
    limit_value(limit)
    conditions, values = [], []
    if path is not None:
        path = store.relative_path(path)
        conditions.extend(("path_hash=%s", "path=%s")); values.extend((hashed(path), path))
    if before_id is not None:
        _need(type(before_id) is int and before_id > 0, "invalid_source_file_id")
        conditions.append("id<%s"); values.append(before_id)
    statement = "SELECT " + ",".join(key for key in FILE_FIELDS if key not in {"original_bytes", "stripped_bytes"}) + " FROM source_document_files"
    if conditions:
        statement += " WHERE " + " AND ".join(conditions)
    with store.transaction() as cursor:
        cursor.execute(statement + " ORDER BY id DESC LIMIT %s", (*values, limit))
        return cursor.fetchall()


def source_documentation(store, path, *, source_id=None, kind=None, symbol=None, limit=200, after_ordinal=None, include_bytes=False):
    limit_value(limit)
    path = store.relative_path(path)
    if kind is not None:
        _need(type(kind) is str and kind in SOURCE_KINDS, "invalid_source_block_kind")
    if symbol is not None:
        _text(symbol, 16_384, "invalid_source_symbol")
    if after_ordinal is not None:
        _need(type(after_ordinal) is int and after_ordinal >= 0, "invalid_source_block_ordinal")
    with store.transaction() as cursor:
        saved = _read_file(cursor, path=path, source_id=source_id)
        if saved is None:
            return None
        blocks = _read_blocks(cursor, saved["id"])
        _verify_saved(saved, blocks)
        selected = [row for row in blocks if (kind is None or row["kind"] == kind)
                    and (symbol is None or row["symbol"] == symbol)
                    and (after_ordinal is None or row["ordinal"] > after_ordinal)][:limit]
        for row in selected:
            if not include_bytes:
                row.pop("raw_bytes")
            row["path"] = path
            row["source_file_id"] = saved["id"]
        return {"path": path, "source_file_id": saved["id"], "original_sha256": saved["original_sha256"],
                "stripped_sha256": saved["stripped_sha256"], "total_blocks": saved["block_count"], "blocks": selected}


def source_doc(store, path, *, symbol="__module__", kind="module_docstring"):
    result = source_documentation(store, path, kind=kind, symbol=symbol, limit=1)
    return None if result is None or not result["blocks"] else result["blocks"][0]["body_text"]


def source_file(store, path, *, original_sha256=None, include_bytes=False):
    path = store.relative_path(path)
    with store.transaction() as cursor:
        saved = _read_file(cursor, path=path, original_sha256=original_sha256)
        if saved is None:
            return None
        row = _verify_saved(saved, _read_blocks(cursor, saved["id"]))
        if not include_bytes:
            row.pop("original_bytes"); row.pop("stripped_bytes")
        return row


def source_counts(store):
    with store.transaction() as cursor:
        result = {}
        for name in ("source_document_files", "source_documentation_blocks"):
            cursor.execute("SELECT COUNT(*) AS count FROM " + name)
            result[name] = cursor.fetchone()["count"]
    return result


def search_source_documentation(store, query, *, limit=20):
    limit_value(limit)
    query = bounded_name(query, 1000)
    statement = ("SELECT f.path,f.id AS source_file_id,b.ordinal,b.kind,b.symbol,b.start_line,b.end_line,b.body_text,b.raw_sha256 "
                 "FROM source_documentation_blocks b JOIN source_document_files f ON f.id=b.source_file_id "
                 "WHERE MATCH(b.body_text) AGAINST (%s IN NATURAL LANGUAGE MODE) "
                 "AND NOT EXISTS (SELECT 1 FROM source_document_files n WHERE n.path_hash=f.path_hash AND n.path=f.path AND n.id>f.id) "
                 "ORDER BY f.id DESC,b.ordinal LIMIT %s")
    with store.transaction() as cursor:
        cursor.execute(statement, (query, limit))
        return [{**row, "kind_of_result": "source_documentation"} for row in cursor.fetchall()]
