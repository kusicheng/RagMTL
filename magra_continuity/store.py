                                                                                    
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import sys

from .config import ContinuityError, database_name
from .documents import document_parts


ROLES = frozenset({"active", "supporting", "excluded", "cleanup", "temporary"})
MAX_DOCUMENTS, MAX_DOCUMENT_BYTES, MAX_FILES = 1000, 64_000_000, 100_000
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS documents (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        path_hash CHAR(64) CHARACTER SET ascii NOT NULL, path TEXT NOT NULL,
        content_sha256 CHAR(64) CHARACTER SET ascii NOT NULL, original_bytes LONGBLOB NOT NULL,
        utf8_text LONGTEXT NOT NULL, byte_count BIGINT UNSIGNED NOT NULL,
        imported_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        UNIQUE KEY document_version (path_hash, content_sha256), FULLTEXT KEY document_text (utf8_text)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS sections (
        document_id BIGINT UNSIGNED NOT NULL, ordinal INT UNSIGNED NOT NULL,
        heading TEXT NOT NULL, level TINYINT UNSIGNED NOT NULL,
        start_byte BIGINT UNSIGNED NOT NULL, end_byte BIGINT UNSIGNED NOT NULL,
        start_character BIGINT UNSIGNED NOT NULL, end_character BIGINT UNSIGNED NOT NULL,
        PRIMARY KEY (document_id, ordinal),
        CONSTRAINT continuity_sections_document FOREIGN KEY (document_id) REFERENCES documents(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS files (
        path_hash CHAR(64) CHARACTER SET ascii PRIMARY KEY, path TEXT NOT NULL, parent TEXT NOT NULL,
        parent_hash CHAR(64) CHARACTER SET ascii NOT NULL, basename VARCHAR(255) NOT NULL,
        role VARCHAR(32) NOT NULL, size_bytes BIGINT UNSIGNED NOT NULL, mtime_ns BIGINT UNSIGNED NOT NULL,
        active_content MEDIUMBLOB NULL, content_sha256 CHAR(64) CHARACTER SET ascii NULL,
        INDEX file_parent (parent_hash), INDEX file_name (basename(191)), INDEX file_role (role)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS current_state (
        state_key VARCHAR(191) PRIMARY KEY, value_json JSON NOT NULL,
        value_sha256 CHAR(64) CHARACTER SET ascii NOT NULL, revision BIGINT UNSIGNED NOT NULL,
        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin""",
    """CREATE TABLE IF NOT EXISTS events (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, kind VARCHAR(64) NOT NULL,
        event_key VARCHAR(191) NULL, payload_json JSON NOT NULL,
        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        INDEX event_kind (kind), INDEX event_key_lookup (event_key)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin""",
)


def canonical(value):
    try:
        result = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ContinuityError("value_must_be_finite_JSON") from None
    if len(result.encode("utf-8")) > 1_000_000:
        raise ContinuityError("continuity_JSON_too_large")
    return result


def hashed(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bounded_name(value, maximum=191):
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ContinuityError("invalid_continuity_key")
    return value


def limit_value(value):
    if type(value) is not int or not 1 <= value <= 200:
        raise ContinuityError("limit_must_be_1_to_200")
    return value


def decoded(value):
    return json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value


class ContinuityStore:
    def __init__(self, config, workspace, *, connection=None):
        self.config, self.workspace = config, Path(workspace).resolve()
        database_name(config.database)
        self._db = connection

    def relative_path(self, value, *, parent=False):
        pass                                                                    
        if not isinstance(value, str) or not value or "\0" in value or "\n" in value or "\r" in value:
            raise ContinuityError("invalid_project_relative_path")
        value = value.replace("\\", "/")
        path = PurePosixPath(value)
        if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or (str(path) == "." and not parent):
            raise ContinuityError("path_outside_workspace")
        if str(path) == "." and parent:
            return "."
        for component in value.split("/"):
                                                                         
                                                                       
            if (not component or component.endswith((".", " "))
                    or any(ord(c) < 32 or c in ':<>"|?*' for c in component)):
                raise ContinuityError("nonportable_project_path")
            stem = component.split(".", 1)[0].rstrip(" ").upper()
            if stem in {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"} or (
                    stem.startswith(("COM", "LPT")) and stem[3:] in "123456789\u00b9\u00b2\u00b3" and len(stem[3:]) == 1):
                raise ContinuityError("reserved_device_path")
        return str(path)

    def content_path(self, value):
        pass                                                                         
        value = self.relative_path(value)
        name = PurePosixPath(value).name.lower()
        if name == ".env" or name.startswith(".env."):
            raise ContinuityError("credential_file_content_forbidden")
        current = self.workspace
        for component in PurePosixPath(value).parts:
            current = current/component
            try:
                info = current.lstat()
            except FileNotFoundError:
                break                                                          
            except OSError:
                raise ContinuityError("content_path_metadata_unavailable") from None
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ContinuityError("linked_content_path_forbidden")
        resolved = (self.workspace/value).resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ContinuityError("path_outside_workspace")
        resolved_name = resolved.name.lower()
        if resolved_name == ".env" or resolved_name.startswith(".env."):
            raise ContinuityError("credential_file_content_forbidden")
        return value

    def _connect(self, *, server=False):
        if self._db is not None:
            return self._db
        try:
            try:
                driver = importlib.import_module("mysql.connector")
            except ModuleNotFoundError as error:
                if error.name not in {"mysql", "mysql.connector"}:
                    raise
                overlay = self.workspace/".runtime/mysql"
                if not overlay.is_dir():
                    raise ContinuityError("mysql_connector_not_installed") from None
                sys.path.insert(0, str(overlay)); importlib.invalidate_caches()
                driver = importlib.import_module("mysql.connector")
            options = dict(host=self.config.host, port=self.config.port, user=self.config.user,
                           password=self.config.password, autocommit=False, charset="utf8mb4",
                           collation="utf8mb4_bin", connection_timeout=10)
            if not server:
                options["database"] = self.config.database
            self._db = driver.connect(**options)
            return self._db
        except ContinuityError:
            raise
        except Exception:
            raise ContinuityError("mysql_connection_failed") from None

    @contextmanager
    def transaction(self):
        connection = self._connect(); cursor = None
        try:
            connection.start_transaction()
            cursor = connection.cursor(dictionary=True)
            yield cursor
                                                                              
            cursor.close()
            cursor = None
            connection.commit()
        except BaseException as error:
            try: connection.rollback()
            except Exception: pass
            if isinstance(error, ContinuityError): raise
            if not isinstance(error, Exception): raise
            raise ContinuityError("continuity_transaction_failed") from None
        finally:
            if cursor is not None:
                try: cursor.close()
                except Exception: pass                                                 

    def bootstrap(self):
        pass                                                                   
        connection = self._connect(server=True); cursor = None
        try:
            cursor = connection.cursor()
            name = database_name(self.config.database)
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin")
            cursor.execute(f"USE `{name}`")
            for statement in SCHEMA: cursor.execute(statement)
            connection.commit()
        except Exception:
            raise ContinuityError("continuity_bootstrap_failed") from None
        finally:
            if cursor is not None:
                try: cursor.close()
                except Exception: raise ContinuityError("continuity_cursor_close_failed") from None
        return {"database": self.config.database, "schema_tables": 5}

    def close(self):
        if self._db is not None:
            try: self._db.close()
            except Exception: raise ContinuityError("continuity_connection_close_failed") from None
            finally: self._db = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def _event(self, cursor, kind, key, payload):
        kind = bounded_name(kind, 64); key = None if key is None else bounded_name(key)
        serialized = canonical(payload)
        cursor.execute("INSERT INTO events (kind, event_key, payload_json) VALUES (%s,%s,%s)", (kind, key, serialized))
        identity = cursor.lastrowid
        cursor.execute("SELECT id,kind,event_key,payload_json FROM events WHERE id=%s", (identity,))
        row = cursor.fetchone()
        if not row or row["kind"] != kind or row["event_key"] != key or canonical(decoded(row["payload_json"])) != serialized:
            raise ContinuityError("event_readback_mismatch")
        return identity

    def import_documents(self, documents):
        prepared = []; total = 0
        for path, raw in documents:
            if not isinstance(raw, bytes): raise ContinuityError("bounded_original_document_bytes_required")
            total += len(raw)
            if len(prepared) >= MAX_DOCUMENTS or total > MAX_DOCUMENT_BYTES:
                raise ContinuityError("document_import_batch_too_large")
            path = self.content_path(path)
            text, checksum, sections = document_parts(raw)
            prepared.append((path, raw, text, checksum, sections))
        results = []
        with self.transaction() as cursor:
            for path, raw, text, checksum, sections in prepared:
                cursor.execute("SELECT id, path, original_bytes, utf8_text FROM documents WHERE path_hash=%s AND content_sha256=%s", (hashed(path), checksum))
                existing = cursor.fetchone()
                if existing:
                    if existing["path"] != path or bytes(existing["original_bytes"]) != raw or existing["utf8_text"] != text:
                        raise ContinuityError("historical_document_readback_mismatch")
                    cursor.execute("SELECT ordinal,heading,level,start_byte,end_byte,start_character,end_character FROM sections WHERE document_id=%s ORDER BY ordinal", (existing["id"],))
                    if cursor.fetchall() != sections:
                        raise ContinuityError("historical_section_readback_mismatch")
                    results.append({"path": path, "sha256": checksum, "id": existing["id"], "inserted": False})
                    continue
                cursor.execute("INSERT INTO documents (path_hash,path,content_sha256,original_bytes,utf8_text,byte_count) VALUES (%s,%s,%s,%s,%s,%s)",
                               (hashed(path), path, checksum, raw, text, len(raw)))
                identity = cursor.lastrowid
                for section in sections:
                    cursor.execute("INSERT INTO sections (document_id,ordinal,heading,level,start_byte,end_byte,start_character,end_character) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                                   (identity, *(section[k] for k in ("ordinal", "heading", "level", "start_byte", "end_byte", "start_character", "end_character"))))
                cursor.execute("SELECT original_bytes, utf8_text, content_sha256 FROM documents WHERE id=%s", (identity,))
                saved = cursor.fetchone()
                cursor.execute("SELECT ordinal,heading,level,start_byte,end_byte,start_character,end_character FROM sections WHERE document_id=%s ORDER BY ordinal", (identity,))
                if not saved or bytes(saved["original_bytes"]) != raw or saved["utf8_text"] != text or saved["content_sha256"] != checksum or cursor.fetchall() != sections:
                    raise ContinuityError("historical_document_readback_mismatch")
                self._event(cursor, "historical_document_imported", None, {"path": path, "sha256": checksum, "document_id": identity})
                results.append({"path": path, "sha256": checksum, "id": identity, "inserted": True})
        return results

    def import_paths(self, paths):
        rows = []; total = 0
        for name in paths:
            if len(rows) >= MAX_DOCUMENTS: raise ContinuityError("document_import_batch_too_large")
            relative = self.content_path(name)
            try:
                path = self.workspace/relative
                size = path.stat().st_size; total += size
                if size > 16_000_000:
                    raise ContinuityError("document_too_large")
                if total > MAX_DOCUMENT_BYTES: raise ContinuityError("document_import_batch_too_large")
                rows.append((relative, path.read_bytes()))
            except OSError:
                raise ContinuityError("explicit_document_unavailable") from None
        return self.import_documents(rows)

    def latest(self, path, *, include_bytes=False):
        path = self.relative_path(path)
        with self.transaction() as cursor:
            cursor.execute("SELECT id,path,content_sha256,original_bytes,utf8_text,byte_count FROM documents WHERE path_hash=%s AND path=%s ORDER BY id DESC LIMIT 1", (hashed(path), path))
            row = cursor.fetchone()
            if row is None: return None
            raw = bytes(row.pop("original_bytes"))
            if hashlib.sha256(raw).hexdigest() != row["content_sha256"] or raw.decode("utf-8") != row["utf8_text"]:
                raise ContinuityError("historical_document_readback_mismatch")
            cursor.execute("SELECT ordinal,heading,level,start_byte,end_byte,start_character,end_character FROM sections WHERE document_id=%s ORDER BY ordinal", (row["id"],))
            row["sections"] = cursor.fetchall(); row["kind"] = "historical_document"
            if row["sections"] != document_parts(raw)[2]:
                raise ContinuityError("historical_section_readback_mismatch")
            if include_bytes: row["original_bytes"] = raw
            return row

    def index_files(self, records, *, active_content=None):
        active_content = {} if active_content is None else {self.relative_path(k): v for k, v in active_content.items()}
        prepared = []
        for record in records:
            if len(prepared) >= MAX_FILES: raise ContinuityError("file_metadata_batch_too_large")
            path = self.relative_path(record["path"]); parent, basename = str(PurePosixPath(path).parent), PurePosixPath(path).name
            role, size, mtime = record["role"], record["size_bytes"], record["mtime_ns"]
            if role not in ROLES or type(size) is not int or not 0 <= size < 2**64 or type(mtime) is not int or not 0 <= mtime < 2**64 or len(basename) > 255:
                raise ContinuityError("invalid_file_metadata")
            content = active_content.get(path)
            if content is not None:
                self.content_path(path)
                if role != "active" or not isinstance(content, bytes) or len(content) != size or len(content) > 1_000_000:
                    raise ContinuityError("explicit_active_content_only")
            prepared.append((hashed(path), path, parent, hashed(parent), basename, role, size, mtime, content,
                             None if content is None else hashlib.sha256(content).hexdigest()))
        if set(active_content) - {row[1] for row in prepared}:
            raise ContinuityError("active_content_requires_file_record")
        with self.transaction() as cursor:
            for row in prepared:
                cursor.execute("INSERT INTO files (path_hash,path,parent,parent_hash,basename,role,size_bytes,mtime_ns,active_content,content_sha256) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE path=VALUES(path),parent=VALUES(parent),parent_hash=VALUES(parent_hash),basename=VALUES(basename),role=VALUES(role),size_bytes=VALUES(size_bytes),mtime_ns=VALUES(mtime_ns),active_content=VALUES(active_content),content_sha256=VALUES(content_sha256)", row)
                cursor.execute("SELECT path,parent,basename,role,size_bytes,mtime_ns,active_content,content_sha256 FROM files WHERE path_hash=%s AND path=%s", (row[0], row[1]))
                saved = cursor.fetchone()
                expected = dict(zip(("path", "parent", "basename", "role", "size_bytes", "mtime_ns", "active_content", "content_sha256"), (row[1], row[2], *row[4:])))
                if saved != expected:
                    raise ContinuityError("file_metadata_readback_mismatch")
        return {"indexed": len(prepared), "filesystem_scan_performed": False}

    def path(self, path):
        path = self.relative_path(path)
        with self.transaction() as cursor:
            cursor.execute("SELECT path,parent,basename,role,size_bytes,mtime_ns,content_sha256 FROM files WHERE path_hash=%s AND path=%s", (hashed(path), path))
            return cursor.fetchone()

    def list_paths(self, parent=".", *, limit=100, after=None):
        parent = self.relative_path(parent, parent=True); limit_value(limit)
        with self.transaction() as cursor:
            statement = "SELECT path,parent,basename,role,size_bytes,mtime_ns,content_sha256 FROM files WHERE parent_hash=%s AND parent=%s"
            values = [hashed(parent), parent]
            if after is not None:
                after = self.relative_path(after); name = PurePosixPath(after).name
                if str(PurePosixPath(after).parent) != parent:
                    raise ContinuityError("pagination_parent_mismatch")
                statement += " AND (basename>%s OR (basename=%s AND path>%s))"; values += [name, name, after]
            cursor.execute(statement+" ORDER BY basename,path LIMIT %s", (*values, limit))
            return cursor.fetchall()

    def search_paths(self, prefix, *, limit=20):
        prefix = bounded_name(prefix, 255); limit_value(limit)
        escaped = prefix.replace("=", "==").replace("%", "=%").replace("_", "=_") + "%"
        with self.transaction() as cursor:
            cursor.execute("SELECT path,parent,basename,role,size_bytes,mtime_ns,content_sha256 FROM files WHERE basename LIKE %s ESCAPE '=' ORDER BY basename,path LIMIT %s", (escaped, limit))
            return cursor.fetchall()

    def search(self, query, *, limit=20):
        if not isinstance(query, str) or not query.strip() or len(query.encode("utf-8")) > 5000:
            raise ContinuityError("bounded_search_text_required")
        limit_value(limit)
        with self.transaction() as cursor:
            cursor.execute("SELECT d.id,d.path,d.content_sha256,SUBSTRING(d.utf8_text,1,600) AS excerpt,MATCH(d.utf8_text) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score FROM documents d WHERE MATCH(d.utf8_text) AGAINST(%s IN NATURAL LANGUAGE MODE) AND NOT EXISTS (SELECT 1 FROM documents newer WHERE newer.path_hash=d.path_hash AND newer.id>d.id) ORDER BY score DESC,d.id DESC LIMIT %s", (query, query, limit))
            return [{**row, "kind": "historical_document"} for row in cursor.fetchall()]

    def set_state(self, key, value, *, reason="explicit_state_update"):
        key = bounded_name(key); payload = canonical(value); checksum = hashed(payload)
        with self.transaction() as cursor:
            cursor.execute("INSERT INTO current_state (state_key,value_json,value_sha256,revision) VALUES (%s,%s,%s,1) ON DUPLICATE KEY UPDATE value_json=VALUES(value_json),value_sha256=VALUES(value_sha256),revision=revision+1", (key, payload, checksum))
            cursor.execute("SELECT state_key,value_json,value_sha256,revision FROM current_state WHERE state_key=%s", (key,))
            row = cursor.fetchone()
            if not row or canonical(decoded(row["value_json"])) != payload or row["value_sha256"] != checksum:
                raise ContinuityError("state_readback_mismatch")
            self._event(cursor, "state_set", key, {"value_sha256": checksum, "revision": row["revision"], "reason": bounded_name(reason)})
            return {"key": key, "value": value, "revision": row["revision"]}

    def get_state(self, key):
        key = bounded_name(key)
        with self.transaction() as cursor:
            cursor.execute("SELECT state_key,value_json,value_sha256,revision FROM current_state WHERE state_key=%s", (key,))
            row = cursor.fetchone()
            if row is None: return None
            value = decoded(row["value_json"])
            if hashed(canonical(value)) != row["value_sha256"]:
                raise ContinuityError("state_readback_mismatch")
            return {"key": row["state_key"], "value": value, "revision": row["revision"]}

    def states(self, *, limit=100):
        limit_value(limit)
        with self.transaction() as cursor:
            cursor.execute("SELECT state_key,value_json,value_sha256,revision FROM current_state ORDER BY state_key LIMIT %s", (limit,))
            result = []
            for row in cursor.fetchall():
                value = decoded(row["value_json"])
                if hashed(canonical(value)) != row["value_sha256"]:
                    raise ContinuityError("state_readback_mismatch")
                result.append({"key": row["state_key"], "value": value, "revision": row["revision"]})
            return result

    def event(self, kind, payload, *, key=None):
        with self.transaction() as cursor:
            return {"id": self._event(cursor, kind, key, payload)}

    def events(self, *, kind=None, limit=20):
        limit_value(limit)
        with self.transaction() as cursor:
            statement = "SELECT id,kind,event_key,payload_json FROM events"
            values = (limit,)
            if kind is not None:
                statement += " WHERE kind=%s"; values = (bounded_name(kind, 64), limit)
            cursor.execute(statement+" ORDER BY id DESC LIMIT %s", values)
            return [{"id": row["id"], "kind": row["kind"], "key": row["event_key"], "payload": decoded(row["payload_json"])} for row in cursor.fetchall()]

    def status(self):
        with self.transaction() as cursor:
            counts = {}
            for table in ("documents", "sections", "files", "current_state", "events"):
                cursor.execute(f"SELECT COUNT(*) AS count FROM `{table}`")
                counts[table] = cursor.fetchone()["count"]
        return {"database": self.config.database, "counts": counts, "historical_docs_are_not_current_state": True}
