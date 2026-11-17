                                                                                   
import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from magra_continuity import ContinuityError, ContinuityStore, MySQLConfig, parse_mysql_config
from magra_continuity.documents import document_parts
from magra_continuity.__main__ import main, parser


class FakeConnection:
    def __init__(self):
        self.data = {name: [] for name in ("documents", "sections", "events")}
        self.data.update(files={}, current_state={})
        self.calls = []; self.commits = self.rollbacks = 0
        self.section_writes = 0; self.fail_section = None; self.corrupt_readback = False
        self.fail_cursor_close = False

    def cursor(self, **_): return FakeCursor(self)
    def start_transaction(self): self.before = copy.deepcopy(self.data)
    def commit(self): self.commits += 1
    def rollback(self): self.data = self.before; self.rollbacks += 1
    def close(self): pass


class FakeCursor:
    def __init__(self, connection): self.c = connection; self.rows = []; self.lastrowid = None
    def close(self):
        if self.c.fail_cursor_close: raise RuntimeError("DO_NOT_EXPOSE_CLOSE_PASSWORD")
    def fetchone(self): return copy.deepcopy(self.rows[0]) if self.rows else None
    def fetchall(self): return copy.deepcopy(self.rows)

    def execute(self, sql, params=None):
        self.c.calls.append((sql, params)); self.rows = []; d = self.c.data
        if sql.startswith(("CREATE ", "USE ")): return
        if sql.startswith("INSERT INTO documents"):
            keys = ("path_hash", "path", "content_sha256", "original_bytes", "utf8_text", "byte_count")
            row = dict(zip(keys, params)); row["id"] = self.lastrowid = len(d["documents"])+1
            d["documents"].append(row); return
        if sql.startswith("INSERT INTO sections"):
            self.c.section_writes += 1
            if self.c.fail_section == self.c.section_writes: raise RuntimeError("DO_NOT_EXPOSE_PASSWORD")
            keys = ("document_id", "ordinal", "heading", "level", "start_byte", "end_byte", "start_character", "end_character")
            d["sections"].append(dict(zip(keys, params))); return
        if sql.startswith("INSERT INTO events"):
            self.lastrowid = len(d["events"])+1
            d["events"].append(dict(id=self.lastrowid, kind=params[0], event_key=params[1], payload_json=params[2])); return
        if sql.startswith("INSERT INTO files"):
            keys = ("path_hash", "path", "parent", "parent_hash", "basename", "role", "size_bytes", "mtime_ns", "active_content", "content_sha256")
            d["files"][params[0]] = dict(zip(keys, params)); return
        if sql.startswith("INSERT INTO current_state"):
            previous = d["current_state"].get(params[0], {}).get("revision", 0)
            d["current_state"][params[0]] = dict(state_key=params[0], value_json=params[1], value_sha256=params[2], revision=previous+1); return
        if sql.startswith("SELECT COUNT(*)"):
            table = sql.split("`")[1]; self.rows = [{"count": len(d[table])}]; return
        if " FROM documents " in sql:
            rows = d["documents"]
            if "WHERE id=%s" in sql: rows = [r for r in rows if r["id"] == params[0]]
            elif "AND content_sha256=%s" in sql: rows = [r for r in rows if (r["path_hash"], r["content_sha256"]) == params]
            elif "WHERE path_hash=%s AND path=%s" in sql: rows = sorted([r for r in rows if (r["path_hash"], r["path"]) == params], key=lambda r: -r["id"])[:1]
            fields = sql.split("SELECT ", 1)[1].split(" FROM ", 1)[0].replace(" ", "").split(",")
            self.rows = [{key: row[key] for key in fields} for row in rows]
            if self.c.corrupt_readback and self.rows and "original_bytes" in self.rows[0]: self.rows[0]["original_bytes"] = b"corrupt"
            return
        if " FROM sections " in sql:
            self.rows = [{k: v for k, v in row.items() if k != "document_id"} for row in d["sections"] if row["document_id"] == params[0]]; return
        if " FROM files " in sql:
            rows = list(d["files"].values())
            if "WHERE path_hash=%s" in sql: rows = [r for r in rows if (r["path_hash"], r["path"]) == params]
            elif "WHERE parent_hash=%s" in sql:
                rows = [r for r in rows if (r["parent_hash"], r["parent"]) == params[:2]]
                if len(params) > 3: rows = [r for r in rows if (r["basename"], r["path"]) > (params[2], params[4])]
                rows = sorted(rows, key=lambda r: (r["basename"], r["path"]))[:params[-1]]
            fields = sql.split("SELECT ", 1)[1].split(" FROM ", 1)[0].split(",")
            self.rows = [{key: row[key] for key in fields} for row in rows]; return
        if " FROM current_state" in sql:
            rows = [d["current_state"][params[0]]] if "WHERE state_key" in sql and params[0] in d["current_state"] else []
            if "WHERE state_key" not in sql: rows = sorted(d["current_state"].values(), key=lambda r: r["state_key"])[:params[0]]
            self.rows = rows; return
        if " FROM events" in sql:
            rows = d["events"]
            if "WHERE id=%s" in sql:
                self.rows = [r for r in rows if r["id"] == params[0]]; return
            if "WHERE kind=%s" in sql: rows = [r for r in rows if r["kind"] == params[0]]
            self.rows = list(reversed(rows))[:params[-1]]; return
        raise AssertionError("Unimplemented fixture SQL contract")


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.connection = FakeConnection()
        self.store = ContinuityStore(MySQLConfig("PRIVATE_PASSWORD"), self.tmp.name, connection=self.connection)

    def test_literal_password_optional_fields_and_repr_redaction(self):
        config = parse_mysql_config('OPENROUTER_API_KEY=ignored\nmysql="mysql://literal:p=a#b"\nMYSQL_PORT=3307\nMYSQL_DATABASE=magra_continuity_test\n')
        self.assertEqual(config.password, "mysql://literal:p=a#b"); self.assertEqual(config.port, 3307)
        self.assertNotIn(config.password, repr(config)); self.assertNotIn("password=", repr(config))
        for text in ("mysql=a\nmysql=b", "mysql=x\nMYSQL_PORT=not-number", "mysql=x\nMYSQL_DATABASE=mysql", "mysql=x\nMYSQL_DATABASE=magra_continuity`; DROP DATABASE mysql;--"):
            with self.assertRaises(ContinuityError): parse_mysql_config(text)

    def test_bootstrap_only_allowlisted_schema_and_five_create_tables(self):
        self.assertEqual(self.store.bootstrap()["schema_tables"], 5)
        statements = [sql for sql, _ in self.connection.calls]
        self.assertEqual(sum(sql.startswith("CREATE TABLE") for sql in statements), 5)
        self.assertTrue(statements[0].startswith("CREATE DATABASE IF NOT EXISTS `magra_continuity`"))
        self.assertFalse(any(sql.startswith(("DROP ", "DELETE ", "TRUNCATE ")) for sql in statements))

    def test_exact_unicode_CRLF_bytes_and_section_offsets(self):
        raw = "# Start\r\n第一条 😀\r\n```text\r\n# Not a heading\r\n```\r\n## Next\r\nLast\r\n".encode()
        result = self.store.import_documents([("cc_mnt/INDEX.md", raw)])
        saved = self.store.latest("cc_mnt/INDEX.md", include_bytes=True)
        self.assertEqual(saved["original_bytes"], raw)
        self.assertEqual(saved["utf8_text"].encode(), raw)
        self.assertEqual(saved["content_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual([s["heading"] for s in saved["sections"]], ["Start", "Next"])
        self.assertEqual(saved["sections"][1]["start_byte"], raw.index(b"## Next"))
        self.assertEqual(b"".join(raw[s["start_byte"]:s["end_byte"]] for s in saved["sections"]), raw)
        self.assertTrue(result[0]["inserted"]); self.assertEqual(self.connection.data["current_state"], {})

    def test_idempotence_and_new_versions_do_not_rewrite_original(self):
        original = b"# Old\r\nold status\r\n"; new = b"# Updated\nnew status\n"
        first = self.store.import_documents([("cc_mnt/INDEX.md", original)])[0]
        repeated = self.store.import_documents([("cc_mnt/INDEX.md", original)])[0]
        self.assertFalse(repeated["inserted"]); self.assertEqual(first["id"], repeated["id"])
        self.store.import_documents([("cc_mnt/INDEX.md", new)])
        self.assertEqual(self.connection.data["documents"][0]["original_bytes"], original)
        self.assertEqual(self.store.latest("cc_mnt/INDEX.md")["utf8_text"], new.decode())
        self.assertEqual(len(self.connection.data["events"]), 2)

    def test_atomic_multi_document_rollback_and_safe_driver_error(self):
        self.connection.fail_section = 2
        with self.assertRaises(ContinuityError) as failure:
            self.store.import_documents([("a.md", b"# First\n"), ("b.md", b"# Second\n")])
        self.assertNotIn("PASSWORD", str(failure.exception))
        self.assertEqual(self.connection.data["documents"], []); self.assertEqual(self.connection.data["events"], [])
        self.assertEqual(self.connection.rollbacks, 1)

    def test_cursor_close_failure_rolls_back_before_commit_and_keeps_primary_error(self):
        self.connection.fail_cursor_close = True
        with self.assertRaisesRegex(ContinuityError, "transaction_failed") as failure:
            self.store.import_documents([("a.md", b"# Original\r\n")])
        self.assertNotIn("PASSWORD", str(failure.exception))
        self.assertEqual(self.connection.commits, 0); self.assertEqual(self.connection.rollbacks, 1)
        self.assertEqual(self.connection.data["documents"], []); self.assertEqual(self.connection.data["events"], [])

    def test_corrupt_readback_rolls_back_and_corrupt_existing_sections_reject(self):
        self.connection.corrupt_readback = True
        with self.assertRaises(ContinuityError): self.store.import_documents([("a.md", b"# A\n")])
        self.assertFalse(self.connection.data["documents"])
        self.connection.corrupt_readback = False; self.store.import_documents([("a.md", b"# A\n")])
        self.connection.data["sections"][0]["end_byte"] = 999
        with self.assertRaisesRegex(ContinuityError, "section_readback"): self.store.import_documents([("a.md", b"# A\n")])

    def test_all_document_state_and_event_values_are_SQL_parameters(self):
        hostile = "x'); DROP TABLE events;--"
        self.store.import_documents([(hostile+".md", ("# "+hostile).encode())])
        self.store.set_state(hostile, {"instruction": hostile}, reason="user_preference")
        self.store.event("preference", {"text": hostile}, key=hostile)
        for sql, params in self.connection.calls:
            self.assertNotIn(hostile, sql)
            if not sql.startswith(("CREATE ", "USE ")): self.assertIsInstance(params, tuple)
        self.assertEqual(self.store.get_state(hostile)["value"], {"instruction": hostile})
        self.assertEqual(len(self.store.events(kind="preference")), 1)

    def test_current_state_revisions_and_append_only_event_API(self):
        self.store.set_state("budget", {"monthly": 100})
        self.assertEqual(self.store.set_state("budget", {"monthly": 100, "stop_on_exhaustion": True})["revision"], 2)
        self.assertEqual(self.store.states()[0]["revision"], 2)
        self.assertEqual([e["kind"] for e in self.store.events()], ["state_set", "state_set"])
        self.assertFalse(any(sql.startswith(("DELETE ", "UPDATE events")) for sql, _ in self.connection.calls))

    def test_path_escape_and_credential_content_reject_before_DB(self):
        for path in ("../secret", "C:/secret", "/secret", "safe/../../secret", "\\\\server\\share\\secret"):
            with self.subTest(path=path), self.assertRaises(ContinuityError): self.store.relative_path(path)
        for path in (".env", "backup/.env.old"):
            with self.assertRaisesRegex(ContinuityError, "credential"): self.store.import_documents([(path, b"mysql=private")])
        self.assertFalse(self.connection.calls)

    def test_portable_stream_trailing_alias_and_device_names_reject_before_reads(self):
        canaries = (".env::$DATA", ".env:stream", ".env. ", ".env.", "safe. /notes.md", "NUL.txt", "CON", "aux.md", "COM1.log", "LPT\u00b9.txt", "CONIN$")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("credential content read")):
            for name in canaries:
                with self.subTest(name=name), self.assertRaises(ContinuityError): self.store.import_paths([name])
        self.assertFalse(self.connection.calls)

    def test_symlink_canary_content_rejects_while_excluded_metadata_stays_indexable(self):
        root = Path(self.tmp.name)
        (root/".env").write_bytes(b"CANARY_NEVER_READ")
        try:
            (root/"notes.md").symlink_to(root/".env")
        except (OSError, NotImplementedError):
            self.skipTest("Host does not permit temporary symlinks; reparse double covers policy")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("credential content read")):
            with self.assertRaisesRegex(ContinuityError, "linked_content"):
                self.store.import_paths(["notes.md"])
        self.store.index_files([dict(path="notes.md", role="excluded", size_bytes=17, mtime_ns=1)])
        self.assertEqual(self.store.path("notes.md")["role"], "excluded")

    def test_reparse_parent_and_final_link_doubles_never_reach_file_read(self):
        ordinary = SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0)
        reparse = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        for linked in ("notes.md", "linked"):
            def lstat(path):
                return reparse if path.name == linked else ordinary
            with self.subTest(linked=linked), patch.object(Path, "lstat", lstat), patch.object(Path, "read_bytes", side_effect=AssertionError("credential content read")):
                with self.assertRaisesRegex(ContinuityError, "linked_content"):
                    self.store.import_paths(["notes.md" if linked == "notes.md" else "linked/notes.md"])
                path = "notes.md" if linked == "notes.md" else "linked/notes.md"
                self.store.index_files([dict(path=path, role="excluded", size_bytes=1, mtime_ns=1)])
                self.assertEqual(self.store.path(path)["role"], "excluded")

    def test_explicit_file_metadata_no_scan_content_opt_in_and_pagination(self):
        rows = [dict(path="code/"+name, role="active", size_bytes=3, mtime_ns=4) for name in ("a.py", "b.py")]
        self.store.index_files(rows)
        self.assertIsNone(next(iter(self.connection.data["files"].values()))["active_content"])
        self.store.index_files(rows, active_content={"code/a.py": b"abc"})
        self.assertEqual(self.store.path("code/a.py")["content_sha256"], hashlib.sha256(b"abc").hexdigest())
        self.assertEqual([r["path"] for r in self.store.list_paths("code", after="code/a.py")], ["code/b.py"])
        bad = [{**rows[0], "role": "excluded"}]
        with self.assertRaisesRegex(ContinuityError, "active_content"): self.store.index_files(bad, active_content={"code/a.py": b"abc"})

    def test_import_reads_only_explicit_paths_and_preserves_BOM(self):
        root = Path(self.tmp.name); original = b"\xef\xbb\xbf# Head\r\nText\r\n"
        (root/"one.md").write_bytes(original); (root/"two.md").write_bytes(b"not selected")
        self.store.import_paths(["one.md"])
        self.assertEqual(len(self.connection.data["documents"]), 1)
        self.assertEqual(self.store.latest("one.md", include_bytes=True)["original_bytes"], original)
        self.assertEqual(document_parts(original)[2][0]["heading"], "Head")

    def test_bounded_batches_fail_before_any_transaction(self):
        with patch("magra_continuity.store.MAX_DOCUMENTS", 1), self.assertRaisesRegex(ContinuityError, "batch"):
            self.store.import_documents([("a.md", b"a"), ("b.md", b"b")])
        with patch("magra_continuity.store.MAX_DOCUMENT_BYTES", 1), self.assertRaisesRegex(ContinuityError, "batch"):
            self.store.import_documents([("a.md", b"ab")])
        self.assertEqual(self.connection.calls, [])

    def test_CLI_has_no_reset_delete_and_does_not_print_driver_or_password(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit): parser().parse_args(["delete"])
        with patch("magra_continuity.__main__.load_mysql_config", return_value=MySQLConfig("SECRET")), patch("magra_continuity.__main__.ContinuityStore", side_effect=RuntimeError("SECRET raw driver failure")), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["status"]), 1)
            self.assertNotIn("SECRET", output.getvalue()); self.assertIn("continuity_command_failed", output.getvalue())


if __name__ == "__main__": unittest.main()
