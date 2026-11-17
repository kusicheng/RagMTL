import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from magra_continuity import ContinuityError, ContinuityStore, MySQLConfig
from magra_continuity import source_store as source


class FakeConnection:
    def __init__(self):
        self.data = {"source_document_files": [], "source_documentation_blocks": [], "events": [],
                     "documents": [{"original_bytes": b"unchanged historical continuity"}],
                     "sections": [{"heading": "unchanged"}]}
        self.calls = []
        self.commits = self.rollbacks = self.transactions = 0
        self.fail_block = self.corrupt_file = self.corrupt_block = self.fail_close = False

    def start_transaction(self):
        self.transactions += 1
        self.before = copy.deepcopy(self.data)

    def cursor(self, **options):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.data = self.before
        self.rollbacks += 1

    def close(self):
        pass


class FakeCursor:
    def __init__(self, connection):
        self.c = connection
        self.rows = []
        self.lastrowid = None

    def close(self):
        if self.c.fail_close:
            raise RuntimeError("SECRET_DRIVER_CLOSE")

    def fetchone(self):
        return copy.deepcopy(self.rows[0]) if self.rows else None

    def fetchall(self):
        return copy.deepcopy(self.rows)

    def execute(self, statement, params=None):
        self.c.calls.append((statement, params))
        self.rows = []
        if statement.startswith("CREATE TABLE"):
            return
        if statement.startswith("INSERT INTO"):
            table = statement.split()[2]
            columns = [column.strip() for column in statement.split("(", 1)[1].split(")", 1)[0].split(",")]
            row = dict(zip(columns, params))
            if table == "source_documentation_blocks" and self.c.fail_block:
                raise RuntimeError("SECRET_DRIVER_BLOCK")
            if table in {"source_document_files", "events"}:
                row["id"] = self.lastrowid = len(self.c.data[table]) + 1
            self.c.data[table].append(row)
            return
        if statement.startswith("SELECT COUNT(*)"):
            table = statement.split(" FROM ", 1)[1]
            self.rows = [{"count": len(self.c.data[table])}]
            return
        if " JOIN source_document_files " in statement:
            latest = {row["path"]: row for row in self.c.data["source_document_files"]}
            rows = []
            for block in self.c.data["source_documentation_blocks"]:
                file = next(r for r in self.c.data["source_document_files"] if r["id"] == block["source_file_id"])
                if file["id"] != latest[file["path"]]["id"] or params[0].casefold() not in block["body_text"].casefold():
                    continue
                rows.append({"path": file["path"], "source_file_id": file["id"], **{k: block[k] for k in
                             ("ordinal", "kind", "symbol", "start_line", "end_line", "body_text", "raw_sha256")}})
            self.rows = sorted(rows, key=lambda r: (-r["source_file_id"], r["ordinal"]))[:params[-1]]
            return
        for table in ("source_document_files", "source_documentation_blocks", "events"):
            if " FROM " + table not in statement:
                continue
            rows = list(self.c.data[table])
            if " WHERE " in statement:
                conditions = statement.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0].split(" AND ")
                for condition, value in zip(conditions, params):
                    if "<%s" in condition:
                        key = condition.split("<", 1)[0]
                        rows = [r for r in rows if r[key] < value]
                    else:
                        key = condition.split("=", 1)[0]
                        rows = [r for r in rows if r[key] == value]
            if "ORDER BY id DESC" in statement:
                rows.sort(key=lambda r: -r["id"])
            elif "ORDER BY ordinal" in statement:
                rows.sort(key=lambda r: r["ordinal"])
            if " LIMIT " in statement:
                limit = params[-1] if statement.endswith("%s") else 1
                rows = rows[:limit]
            columns = [column.strip() for column in statement.split("SELECT ", 1)[1].split(" FROM ", 1)[0].split(",")]
            self.rows = [{key: row[key] for key in columns} for row in rows]
            if self.rows and table == "source_document_files" and self.c.corrupt_file and "original_bytes" in self.rows[0]:
                self.rows[0]["original_bytes"] = b"corrupt"
            if self.rows and table == "source_documentation_blocks" and self.c.corrupt_block:
                self.rows[0]["body_text"] += " corrupt"
            return
        raise AssertionError("Unsupported source documentation SQL fixture")


def analysis(path="code/example.py", suffix=b""):
    original = b'\xef\xbb\xbf"""Caf\xc3\xa9 documentation.\r\nSecond line."""\r\nx = "# functional string"\r\n# explanation\r\n' + suffix
    first_start = original.index(b'"""')
    first_end = original.index(b'"""', first_start + 3) + 3
    comment_start = original.index(b"# explanation")
    comment_end = comment_start + len(b"# explanation")
    stripped = bytearray(original)
    for start, end in ((first_start, first_end), (comment_start, comment_end)):
        for index in range(start, end):
            if stripped[index] not in (10, 13):
                stripped[index] = 32
    stripped = bytes(stripped)
    blocks = []
    for ordinal, (kind, start, end, text, line, column) in enumerate([
            ("module_docstring", first_start, first_end, "Café documentation.\nSecond line.", 1, 0),
            ("comment", comment_start, comment_end, "# explanation", 4, 0)]):
        raw = original[start:end]
        blocks.append({"ordinal": ordinal, "kind": kind, "symbol": "__module__", "start_line": line,
                       "start_col": column, "end_line": line + (1 if ordinal == 0 else 0), "end_col": len(raw),
                       "start_byte": start, "end_byte": end, "raw_bytes": raw, "sha256": hashlib.sha256(raw).hexdigest(),
                       "text": text, "preserve_reason": None})
    return {"path": path, "encoding": "utf-8-sig", "original_bytes": original, "stripped_bytes": stripped,
            "original_sha256": hashlib.sha256(original).hexdigest(), "stripped_sha256": hashlib.sha256(stripped).hexdigest(),
            "semantic_sha256": hashlib.sha256(b"fictional executable semantics").hexdigest(), "blocks": blocks}


class SourceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.connection = FakeConnection()
        self.store = ContinuityStore(MySQLConfig("PRIVATE_PASSWORD"), self.temporary.name, connection=self.connection)

    def test_two_additive_schema_tables_leave_historical_documents_unchanged(self):
        before = copy.deepcopy(self.connection.data)
        result = source.initialize_source_schema(self.store)
        self.assertEqual(result["source_schema_tables"], 2)
        self.assertEqual(self.connection.data, before)
        statements = [sql for sql, params in self.connection.calls]
        self.assertEqual(len(statements), 2)
        self.assertTrue(all(sql.startswith("CREATE TABLE IF NOT EXISTS source_document") for sql in statements))

    def test_exact_source_and_block_bytes_unicode_BOM_CRLF_hashes_and_recovery(self):
        item = analysis()
        result = source.import_source_documents(self.store, [item], "migration-1")
        self.assertTrue(result[0]["inserted"])
        raw = source.source_file(self.store, item["path"], include_bytes=True)
        self.assertEqual(raw["original_bytes"], item["original_bytes"])
        self.assertEqual(raw["stripped_bytes"], item["stripped_bytes"])
        self.assertEqual(raw["source_encoding"], "utf-8-sig")
        blocks = source.source_documentation(self.store, item["path"], include_bytes=True)["blocks"]
        self.assertEqual([row["raw_bytes"] for row in blocks], [row["raw_bytes"] for row in item["blocks"]])
        self.assertEqual(source.source_doc(self.store, item["path"]), item["blocks"][0]["text"])
        verified = source.verify_source_documents(self.store, [item])
        self.assertEqual(verified[0]["original_bytes_verified"], len(item["original_bytes"]))
        self.assertNotIn("original_bytes", source.source_file(self.store, item["path"]))
        self.assertNotIn("raw_bytes", source.source_documentation(self.store, item["path"])["blocks"][0])

    def test_idempotence_versions_latest_and_explicit_old_source_recovery(self):
        old, new = analysis(), analysis(suffix=b"x += 1\r\n")
        first = source.import_source_documents(self.store, [old], "migration-1")
        repeated = source.import_source_documents(self.store, [old], "migration-2")
        self.assertEqual(first[0]["source_file_id"], repeated[0]["source_file_id"])
        self.assertFalse(repeated[0]["inserted"])
        source.import_source_documents(self.store, [new], "migration-3")
        self.assertEqual(source.source_file(self.store, old["path"])["original_sha256"], new["original_sha256"])
        self.assertEqual(source.source_file(self.store, old["path"], original_sha256=old["original_sha256"], include_bytes=True)["original_bytes"], old["original_bytes"])
        self.assertEqual(source.source_file_versions(self.store, old["path"], limit=1)[0]["id"], 2)
        self.assertEqual(source.source_file_versions(self.store, old["path"], before_id=2)[0]["id"], 1)
        self.assertEqual(source.source_counts(self.store), {"source_document_files": 2, "source_documentation_blocks": 4})

    def test_whole_batch_rolls_back_for_failed_block_or_corrupt_readback(self):
        for flag, expected in (("fail_block", "transaction_failed"), ("corrupt_file", "readback_mismatch"),
                               ("corrupt_block", "readback_mismatch"), ("fail_close", "transaction_failed")):
            with self.subTest(flag=flag):
                self.connection = FakeConnection()
                self.store = ContinuityStore(MySQLConfig("PRIVATE_PASSWORD"), self.temporary.name, connection=self.connection)
                setattr(self.connection, flag, True)
                with self.assertRaisesRegex(ContinuityError, expected):
                    source.import_source_documents(self.store, [analysis(), analysis("code/second.py")], "migration-1")
                self.assertEqual(self.connection.data["source_document_files"], [])
                self.assertEqual(self.connection.data["source_documentation_blocks"], [])
                self.assertEqual(self.connection.data["events"], [])
                self.assertEqual(self.connection.commits, 0)
                self.assertEqual(self.connection.rollbacks, 1)

    def test_changed_existing_snapshot_or_block_metadata_cannot_silently_replace(self):
        item = analysis()
        source.import_source_documents(self.store, [item], "migration-1")
        changed = copy.deepcopy(item)
        changed["blocks"][0]["text"] = "Changed documentation"
        with self.assertRaisesRegex(ContinuityError, "readback_mismatch"):
            source.import_source_documents(self.store, [changed], "migration-2")
        self.assertEqual(source.source_doc(self.store, item["path"]), item["blocks"][0]["text"])
        self.connection.data["source_documentation_blocks"][0]["start_line"] += 1
        with self.assertRaisesRegex(ContinuityError, "readback_mismatch"):
            source.source_doc(self.store, item["path"])

    def test_preserved_directives_and_licenses_keep_their_reasons_and_filters(self):
        item = analysis()
        item["blocks"][1]["preserve_reason"] = "functional_directive"
        source.import_source_documents(self.store, [item], "migration-1")
        result = source.source_documentation(self.store, item["path"], kind="comment", symbol="__module__")
        self.assertEqual(result["blocks"][0]["preserve_reason"], "functional_directive")
        self.assertEqual(source.source_documentation(self.store, item["path"], after_ordinal=0)["blocks"][0]["ordinal"], 1)
        self.assertIsNone(source.source_doc(self.store, "code/missing.py"))

    def test_bad_paths_types_hashes_offsets_semantics_and_caps_fail_before_transaction(self):
        changes = [{"path": "../escape.py"}, {"path": ".env"}, {"encoding": "unknown-encoding"},
                   {"original_sha256": "0" * 64}, {"stripped_sha256": "0" * 64},
                   {"stripped_semantic_sha256": "0" * 64}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ContinuityError):
                source.import_source_documents(self.store, [{**analysis(), **change}], "migration-1")
        for key, value in (("ordinal", False), ("start_byte", -1), ("end_byte", 99999),
                           ("raw_bytes", b"different"), ("sha256", "0" * 64), ("kind", "unknown")):
            item = analysis()
            item["blocks"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ContinuityError):
                source.import_source_documents(self.store, [item], "migration-1")
        with patch.object(source, "MAX_SOURCE_FILES", 1), self.assertRaises(ContinuityError):
            source.import_source_documents(self.store, [analysis(), analysis("code/two.py")], "migration-1")
        with patch.object(source, "MAX_SOURCE_BYTES", 1), self.assertRaises(ContinuityError):
            source.import_source_documents(self.store, [analysis()], "migration-1")
        with self.assertRaises(ContinuityError):
            source.import_source_documents(self.store, [analysis(), analysis()], "migration-1")
        self.assertEqual(self.connection.transactions, 0)

    def test_zero_documentation_source_is_stored_verified_and_search_excludes_old_versions(self):
        item = analysis()
        item["blocks"] = []
        source.import_source_documents(self.store, [item], "migration-1")
        self.assertEqual(source.source_documentation(self.store, item["path"])["blocks"], [])
        self.assertEqual(source.verify_source_documents(self.store, [item])[0]["blocks"], 0)
        source.import_source_documents(self.store, [analysis(suffix=b"x += 1\r\n")], "migration-2")
        hits = source.search_source_documentation(self.store, "documentation")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["source_file_id"], 2)

    def test_values_are_SQL_parameters_and_cross_path_source_ids_are_rejected(self):
        hostile = "code/a'); DROP TABLE fake;--.py"
        item = analysis(hostile)
        item["blocks"][0]["symbol"] = "name'); DROP TABLE fake; --"
        source.import_source_documents(self.store, [item], "migration-1")
        source.search_source_documentation(self.store, "query'); DROP TABLE fake; --")
        self.assertTrue(all(hostile not in sql and item["blocks"][0]["symbol"] not in sql for sql, params in self.connection.calls))
        with self.assertRaisesRegex(ContinuityError, "path_readback"):
            source.source_documentation(self.store, "code/other.py", source_id=1)
        self.assertEqual(self.connection.data["documents"], [{"original_bytes": b"unchanged historical continuity"}])
        self.assertEqual(self.connection.data["sections"], [{"heading": "unchanged"}])


if __name__ == "__main__":
    unittest.main()
