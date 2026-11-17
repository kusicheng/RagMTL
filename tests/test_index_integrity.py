from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch

from magra_rag import index_integrity
from magra_rag.index_integrity import dense_index_fingerprint, effective_query_scope
from magra_rag.storage import DocumentStore


MODEL = "fixture/encoder@immutable-revision"


class IndexIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = DocumentStore(self.root / "corpus.sqlite")
        self.addCleanup(self.store.close)
        self.docs = [
            {"id": doc_id, "title": doc_id, "text": "Canonical owned text " + doc_id,
             "metadata": {"source": source}}
            for doc_id, source in [
                ("a", "scifact"), ("b", "scifact"), ("c", "nfcorpus"),
                ("fiqa:a", "fiqa"), ("fiqa:b", "fiqa"),
                ("hotpotqa:wiki:Article", "hotpotqa"), ("hotpotqa:context:Article", "hotpotqa"),
                ("arguana:q", "arguana"), ("owned:a%:x", "owned"), ("owned:abc:x", "owned"),
            ]
        ]
        self.store.ingest(self.docs)
        for doc in self.docs:
            if doc["metadata"]["source"] != "fiqa":
                self.write_vector(doc["id"])
        self.scopes = [effective_query_scope({"id": "science:q", "corpus_source": "scifact"}),
                       effective_query_scope({"id": "nutrition:q", "corpus_source": "nfcorpus"})]

    def write_vector(self, doc_id, values=(0.25, 0.75), *, store=None, model=MODEL):
        store = store or self.store
        content_hash = store.connection.execute(
            "SELECT content_hash FROM documents WHERE id=?", (doc_id,)).fetchone()[0]
        with store.connection:
            store.connection.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?)",
                                     (doc_id, model, content_hash, len(values), struct.pack("<" + "f" * len(values), *values)))

    def fingerprint(self, *, scopes=None, **kwargs):
        return dense_index_fingerprint(self.store, MODEL, scopes=self.scopes if scopes is None else scopes,
                                       **kwargs)

    def scope(self, qid="q", **filters):
        return {"query_id": qid, **filters}

    def assert_same_rows(self, actual_scopes, expected_ids):
        actual = self.fingerprint(scopes=actual_scopes)
        expected = self.fingerprint(scopes=[self.scope("reference", allowed_ids=expected_ids)])
        self.assertEqual(actual["document_count"], len(expected_ids))
        self.assertEqual(actual["universe_hash"], expected["universe_hash"])
        self.assertEqual(actual["vectors_hash"], expected["vectors_hash"])
        return actual

    def test_effective_query_scope_matches_engine_precedence_and_special_cases(self):
        scope = effective_query_scope({"id": "arguana:q", "query_id": "ignored", "source": "arguana",
                                       "corpus_source": "arguana", "exclude_doc_ids": ["b", "b"],
                                       "allowed_doc_ids": {"b", "a"}, "context_doc_ids": ["c"]})
        self.assertEqual(scope, {"query_id": "arguana:q", "source": "arguana", "id_prefix": None,
                                 "allowed_ids": ["a", "b"], "exclude_ids": ["arguana:q", "b"]})
        hotpot = effective_query_scope({"query_id": "hotpot:q", "source": "hotpotqa", "setting": "full_corpus"})
        self.assertEqual(hotpot["id_prefix"], "hotpotqa:wiki:")
        self.assertIsNone(hotpot["source"])
        self.assertIsNone(effective_query_scope({"id": "q", "source": "scifact"})["source"])
        self.assertIsNone(effective_query_scope({"id": "q", "allowed_doc_ids": None,
                                                "context_doc_ids": ["a"]})["allowed_ids"])
        empty = effective_query_scope({"id": "q", "setting": "provided_context", "context_doc_ids": []})
        self.assertEqual(empty["allowed_ids"], [])
        self.assertEqual(effective_query_scope({"id": "q", "context_doc_ids": ["b", "a", "a"]})["allowed_ids"], ["a", "b"])
        for query in ({"id": "q", "setting": "provided_context"},
                      {"id": "q", "setting": "provided_context", "allowed_doc_ids": None, "context_doc_ids": ["a"]}):
            with self.assertRaisesRegex(ValueError, "explicit candidate"):
                effective_query_scope(query)

    def test_public_filter_helper_ignores_answers_and_annotations(self):
        query = {"id": "q", "corpus_source": "scifact", "question": "Public question"}
        self.assertEqual(effective_query_scope(query), effective_query_scope({
            **query, "answer": "PRIVATE", "supporting_facts": [["a", 0]], "labels": [1],
        }))
        self.assertNotIn("PRIVATE", json.dumps(effective_query_scope(query)))

    def test_scoped_fingerprint_is_reusable_json_and_pins_the_bytes(self):
        result = self.fingerprint(expected_dimensions=2)
        self.assertEqual(result["format"], "magra-dense-index-v1")
        self.assertEqual(result["document_count"], 3)
        self.assertEqual(result["vector_bytes"], 24)
        self.assertEqual(result["dimensions"], 2)
        self.assertEqual(result["dtype"], "<f4")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(json.loads(json.dumps(result)), result)
        unsigned = {key: value for key, value in result.items() if key != "fingerprint_hash"}
        expected = hashlib.sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(result["fingerprint_hash"], expected)
        for key in ("scopes_hash", "universe_hash", "vectors_hash", "fingerprint_hash"):
            self.assertRegex(result[key], "^[0-9a-f]{64}$")

    def test_relevant_finite_vector_tamper_changes_pin_but_not_document_universe(self):
        original = self.fingerprint()
        original_corpus_hash = self.store.corpus_hash()
        self.write_vector("b", (0.25, 0.5))
        altered = self.fingerprint()
        self.assertEqual(original_corpus_hash, self.store.corpus_hash())
        self.assertEqual(original["universe_hash"], altered["universe_hash"])
        self.assertNotEqual(original["vectors_hash"], altered["vectors_hash"])
        self.assertNotEqual(original["fingerprint_hash"], altered["fingerprint_hash"])
                                                                                 
        self.write_vector("b", (0.0, 0.5))
        positive_zero = self.fingerprint()
        self.write_vector("b", (-0.0, 0.5))
        self.assertNotEqual(positive_zero["vectors_hash"], self.fingerprint()["vectors_hash"])

    def test_unrelated_fiqa_growth_wrong_model_and_logs_do_not_change_scoped_pin(self):
        original = self.fingerprint()
        self.write_vector("fiqa:a")
        self.write_vector("fiqa:b", (99.0, -37.0), model="different/model@commit")
        with self.store.connection:
            self.store.connection.execute("INSERT INTO metadata VALUES(?,?)", ("worker_progress", "unrelated"))
        self.assertEqual(original, self.fingerprint())
        with self.assertRaisesRegex(ValueError, "wrong-model"):
            self.fingerprint(scopes=[self.scope(source="fiqa")])

    def test_missing_stale_wrong_model_dimensions_and_nonfinite_rows_refuse_fingerprint(self):
        mutations = [
            ("DELETE FROM vectors WHERE doc_id='b'", (), "missing vector"),
            ("UPDATE vectors SET model=? WHERE doc_id='b'", ("other/model@commit",), "wrong-model"),
            ("UPDATE vectors SET content_hash=? WHERE doc_id='b'", ("stale",), "stale content"),
            ("UPDATE vectors SET dimensions=0 WHERE doc_id='b'", (), "invalid dimensions"),
            ("UPDATE vectors SET dimensions=3,embedding=? WHERE doc_id='b'", (struct.pack("<3f", 1, 2, 3),), "dimensions"),
            ("UPDATE vectors SET embedding=? WHERE doc_id='b'", (b"bad",), "blob length/type"),
            ("UPDATE vectors SET embedding=? WHERE doc_id='b'", ("12345678",), "blob length/type"),
            ("UPDATE vectors SET embedding=? WHERE doc_id='b'", (struct.pack("<2f", float("nan"), 1),), "nonfinite"),
            ("UPDATE vectors SET embedding=? WHERE doc_id='b'", (struct.pack("<2f", float("inf"), 1),), "nonfinite"),
        ]
        for sql, parameters, message in mutations:
            with self.subTest(sql=sql, message=message):
                self.write_vector("b")
                with self.store.connection:
                    self.store.connection.execute(sql, parameters)
                with self.assertRaisesRegex(ValueError, message):
                    self.fingerprint()
        self.write_vector("b")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.fingerprint(expected_dimensions=3)

    def test_per_query_exclusions_are_unioned_before_global_filtering(self):
        self.assert_same_rows([
            self.scope("one", allowed_ids=["a", "b"], exclude_ids=["a"]),
            self.scope("two", allowed_ids=["a", "c"]),
        ], ["a", "b", "c"])
        self.assert_same_rows([
            self.scope("one", source="scifact", exclude_ids=["a"]),
            self.scope("two", source="scifact", exclude_ids=["b"]),
        ], ["a", "b"])
        self.assert_same_rows([
            self.scope("one", source="scifact", exclude_ids=["a", "b"]),
            self.scope("two", source="scifact", allowed_ids=["a"]),
        ], ["a"])

    def test_all_filter_axes_and_literal_prefixes_define_exact_candidate_union(self):
        self.assert_same_rows([
            self.scope("one", source="scifact", id_prefix="a", allowed_ids=["a", "b", "c"]),
            self.scope("two", source="nfcorpus", exclude_ids=["b"]),
        ], ["a", "c"])
        self.assert_same_rows([self.scope(id_prefix="owned:a%:")], ["owned:a%:x"])
        self.assert_same_rows([effective_query_scope({"id": "hotpot:q", "source": "hotpotqa", "setting": "full_corpus"})],
                              ["hotpotqa:wiki:Article"])
        self.assert_same_rows([effective_query_scope({"id": "arguana:q", "source": "arguana", "corpus_source": "arguana"})], [])

    def test_empty_scopes_and_allowlists_do_not_become_unrestricted(self):
        for scopes in ([], [self.scope(allowed_ids=[])],
                       [self.scope(allowed_ids=["missing-document"])],
                       [self.scope(allowed_ids=["a"], exclude_ids=["a"])],
                       [self.scope(source="missing-source")]):
            with self.subTest(scopes=scopes):
                result = self.fingerprint(scopes=scopes, expected_dimensions=2)
                self.assertEqual(result["document_count"], 0)
                self.assertEqual(result["vector_bytes"], 0)
                self.assertIsNone(result["dimensions"])
        with self.assertRaisesRegex(ValueError, "missing vector"):
            self.fingerprint(scopes=[self.scope(allowed_ids=None)])

    def test_scope_ids_and_filter_changes_bind_dev_and_eval_even_for_same_rows(self):
        dev = self.fingerprint(scopes=[self.scope("dev:q", allowed_ids=["a", "b"])])
        evaluation = self.fingerprint(scopes=[self.scope("eval:q", allowed_ids=["a", "b"])])
        changed = self.fingerprint(scopes=[self.scope("dev:q", allowed_ids=["a", "b"], exclude_ids=["unused"])])
        self.assertEqual(dev["vectors_hash"], evaluation["vectors_hash"])
        self.assertNotEqual(dev["scopes_hash"], evaluation["scopes_hash"])
        self.assertNotEqual(dev["fingerprint_hash"], evaluation["fingerprint_hash"])
        self.assertNotEqual(dev["scopes_hash"], changed["scopes_hash"])
        reordered = self.fingerprint(scopes=[self.scope("dev:q", allowed_ids=["b", "a", "b"])])
        self.assertEqual(dev, reordered)

    def test_query_order_and_sql_insertion_order_do_not_change_fingerprint(self):
        original = self.fingerprint()
        self.assertEqual(original, self.fingerprint(scopes=list(reversed(self.scopes))))
        other = DocumentStore(self.root / "other.sqlite")
        self.addCleanup(other.close)
        other.ingest(reversed(self.docs))
        for doc in reversed(self.docs):
            if doc["metadata"]["source"] != "fiqa":
                self.write_vector(doc["id"], store=other)
        self.assertEqual(original, dense_index_fingerprint(other, MODEL, scopes=self.scopes))

    def test_ten_thousand_context_scopes_merge_without_expression_limit(self):
        scopes = [self.scope(f"q-{i}", allowed_ids=["a", "b", f"missing-{i}"],
                             exclude_ids=["a"] if i % 2 else ["b"])
                  for i in range(10_000)]
        result = self.assert_same_rows(scopes, ["a", "b"])
        self.assertEqual(result["query_count"], 10_000)
        self.assertEqual(result["scope_group_count"], 1)

    def test_many_distinct_prefixes_use_bounded_depth_sql_fallback(self):
        scopes = [self.scope(f"q-{i}", id_prefix=f"absent-{i}:") for i in range(1005)]
        scopes.append(self.scope("present", id_prefix="owned:a%:"))
        self.assert_same_rows(scopes, ["owned:a%:x"])
        with patch.object(index_integrity, "_MAX_OR_GROUPS", 1):
            self.assert_same_rows([
                self.scope("one", source="scifact", exclude_ids=["b"]),
                self.scope("two", source="nfcorpus", allowed_ids=["a", "c"]),
                self.scope("empty", allowed_ids=[]),
            ], ["a", "c"])

    def test_snapshot_stays_consistent_when_another_connection_updates_between_pages(self):
        self.store.connection.execute("PRAGMA journal_mode=WAL")
        original = self.fingerprint()
        writer = sqlite3.connect(self.store.path)
        self.addCleanup(writer.close)
        original_batches, wrote = index_integrity._blob_batches, []

        def update_between_reads(rows):
            if not wrote:
                with writer:
                    writer.execute("UPDATE vectors SET embedding=? WHERE doc_id='b'", (struct.pack("<2f", 2, 3),))
                wrote.append(True)
            yield from original_batches(rows)

        with patch.object(index_integrity, "_ROW_PAGE", 1), \
                patch.object(index_integrity, "_blob_batches", side_effect=update_between_reads):
            during_update = self.fingerprint()
        self.assertEqual(original, during_update)
        self.assertNotEqual(original["vectors_hash"], self.fingerprint()["vectors_hash"])

    def test_blob_pages_are_bounded_and_invalid_lengths_fail_before_blob_materialization(self):
        original_batches, batches = index_integrity._blob_batches, []

        def observed(rows):
            for batch in original_batches(rows):
                batches.append((len(rows), sum(row[7] for row in batch)))
                yield batch

        with patch.object(index_integrity, "_ROW_PAGE", 2), \
                patch.object(index_integrity, "_BLOB_PAGE_BYTES", 8), \
                patch.object(index_integrity, "_blob_batches", side_effect=observed):
            self.assertEqual(self.fingerprint()["document_count"], 3)
        self.assertTrue(batches)
        self.assertTrue(all(rows <= 2 and size <= 8 for rows, size in batches))
        with self.store.connection:
            self.store.connection.execute("UPDATE vectors SET embedding=zeroblob(1048577) WHERE doc_id='b'")
        traced, connect = [], sqlite3.connect

        def observe_connection(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connection.set_trace_callback(traced.append)
            return connection

        with patch.object(index_integrity.sqlite3, "connect", side_effect=observe_connection):
            with self.assertRaisesRegex(ValueError, "blob length/type"):
                self.fingerprint()
        self.assertFalse(any(sql.startswith("SELECT doc_id,embedding") for sql in traced))

    def test_caller_pragmas_transactions_and_optional_dependencies_are_untouched(self):
        before = {name: self.store.connection.execute("PRAGMA " + name).fetchone()[0]
                  for name in ("temp_store", "cache_size", "mmap_size", "query_only", "journal_mode")}
        with patch.dict("sys.modules", {"numpy": None, "torch": None, "sentence_transformers": None}):
            self.fingerprint()
        after = {name: self.store.connection.execute("PRAGMA " + name).fetchone()[0] for name in before}
        self.assertEqual(before, after)
        self.store.connection.execute("UPDATE vectors SET dimensions=2 WHERE doc_id='b'")
        with self.assertRaisesRegex(ValueError, "Commit or roll back"):
            self.fingerprint()
        self.assertTrue(self.store.connection.in_transaction)
        self.store.connection.rollback()
                                                                                  
        with self.store.connection:
            self.store.connection.execute("DELETE FROM vectors WHERE doc_id='b'")
        with self.assertRaisesRegex(ValueError, "missing vector"):
            self.fingerprint()
        with closing(sqlite3.connect(self.store.path, timeout=0.1)) as writer:
            with writer:
                writer.execute("INSERT INTO metadata VALUES('after_failure','writer can commit')")

    def test_invalid_inputs_are_rejected_before_any_model_or_index_work(self):
        bad_scopes = [None, "not-scopes", [self.scope(), self.scope()],
                      [self.scope(allowed_ids="a")], [self.scope(exclude_ids=[None])],
                      [self.scope(source="")], [self.scope(id_prefix=7)],
                      [self.scope(answer="PRIVATE")], [{"source": "scifact"}]]
        for scopes in bad_scopes:
            with self.subTest(scopes=scopes), self.assertRaises(ValueError):
                self.fingerprint(scopes=scopes) if scopes is not None else dense_index_fingerprint(self.store, MODEL, scopes=None)
        for size in (0, -1, True, "2", 262145):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.fingerprint(expected_dimensions=size)
        for model in (None, "", "  "):
            with self.subTest(model=model), self.assertRaises(ValueError):
                dense_index_fingerprint(self.store, model, scopes=self.scopes)


if __name__ == "__main__":
    unittest.main()
