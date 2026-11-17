                                                                             

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from magra_rag.resources import BudgetExceeded, ResourceBudget
from magra_rag.storage import DocumentStore, tokenize


def reference(documents, question, *, title_weight=2.0):
    pass                                                                    
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE TABLE docs(id TEXT,title TEXT,text TEXT)")
        db.execute("CREATE VIRTUAL TABLE ft USING fts5(title,text,tokenize='porter unicode61')")
        for rowid, doc in enumerate(documents, 1):
            db.execute("INSERT INTO docs(rowid,id,title,text) VALUES(?,?,?,?)",
                       (rowid, doc["id"], doc["title"], doc["text"]))
            db.execute("INSERT INTO ft(rowid,title,text) VALUES(?,?,?)", (rowid, doc["title"], doc["text"]))
        terms = list(dict.fromkeys(tokenize(question)))[:64]
        query = " OR ".join('"' + term + '"' for term in terms)
        return [(doc_id, -rank) for doc_id, rank in db.execute(
            "SELECT d.id,bm25(ft,?,1.0) AS rank FROM ft JOIN docs d ON d.rowid=ft.rowid "
            "WHERE ft MATCH ? ORDER BY rank,d.id", (title_weight, query))]


class ScopedRetrievalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.budget = ResourceBudget(self.root, disk_bytes=100_000_000, reserve_bytes=0)
        self.store = DocumentStore(self.root / "docs.sqlite", budget=self.budget)
        self.addCleanup(self.store.close)
        self.documents = [
            {"id": "a:1", "title": "Running cogs", "text": "The running engine uses amber.", "metadata": {"source": "a"}},
            {"id": "a:2", "title": "Amber engines", "text": "A piston cycles. " + "long " * 40, "metadata": {"source": "a"}},
            {"id": "a:3", "title": "Research", "text": "An amber sample.", "metadata": {"source": "a"}},
            {"id": "a:4", "title": "Other", "text": "Blue crystal data.", "metadata": {"source": "a"}},
            {"id": "wiki:1", "title": "Running engine", "text": "Amber mechanisms.", "metadata": {"source": "hotpotqa"}},
            {"id": "wiki:2", "title": "Minerals", "text": "Amber and blue minerals.", "metadata": {"source": "hotpotqa"}},
            {"id": "context:1", "title": "Running", "text": "Running " * 100, "metadata": {"source": "hotpotqa"}},
        ]
        self.store.ingest(self.documents)

    def assert_scores(self, actual, expected):
        self.assertEqual([row["id"] for row in actual], [row[0] for row in expected])
        for row, (_, score) in zip(actual, expected):
            self.assertAlmostEqual(row["score"], score, places=12)

    def test_source_scores_equal_standalone_corpus_and_ignore_unrelated_additions(self):
        selected = [doc for doc in self.documents if doc["metadata"]["source"] == "a"]
        expected = reference(selected, "running amber")
        self.assert_scores(self.store.search("running amber", source="a"), expected)
        unrelated = [{"id": f"noise:{number}", "title": "Running unrelated",
                      "text": "running " * (number + 20), "metadata": {"source": "other"}}
                     for number in range(100)]
        self.store.ingest(unrelated)
        self.assert_scores(self.store.search("running amber", source="a"), expected)
        global_filtered = [row for row in self.store.search("running amber", k=200) if row["id"].startswith("a:")]
        self.assertTrue(any(abs(row["score"] - dict(expected)[row["id"]]) > 1e-8 for row in global_filtered))

    def test_prefix_and_source_combination_has_its_own_document_lengths_and_df(self):
        wiki = [doc for doc in self.documents if doc["id"].startswith("wiki:")]
        expected = reference(wiki, "running amber")
        self.assert_scores(self.store.search("running amber", source="hotpotqa", id_prefix="wiki:"), expected)
        self.assert_scores(self.store.search("running amber", id_prefix="wiki:"), expected)
        source_table = self.store.ensure_bm25_scope(source="hotpotqa")
        prefix_table = self.store.ensure_bm25_scope(source="hotpotqa", id_prefix="wiki:")
        self.assertNotEqual(source_table, prefix_table)

    def test_allowed_contexts_use_all_selected_docs_in_statistics_with_same_porter_stemming(self):
        ids = {"a:1", "a:3", "a:4", "wiki:2"}
        selected = [doc for doc in self.documents if doc["id"] in ids]
        self.assert_scores(self.store.search("runs amber", allowed_ids=ids), reference(selected, "runs amber"))
        selected_a = [doc for doc in selected if doc["metadata"]["source"] == "a" and doc["id"] != "a:1"]
        self.assert_scores(self.store.search("amber", allowed_ids=ids, source="a", id_prefix="a:",
                                            exclude_ids={"a:1"}), reference(selected_a, "amber"))
        self.store.ingest([{"id": "unrelated", "title": "Amber", "text": "Amber " * 200,
                            "metadata": {"source": "other"}}])
        self.assert_scores(self.store.search("runs amber", allowed_ids=ids), reference(selected, "runs amber"))

    def test_scoped_append_triggers_update_scores_without_rebuild_or_text_copy(self):
        table = self.store.ensure_bm25_scope(source="a")
        before_hash = self.store.corpus_hash()
        self.assertEqual(self.store.ensure_bm25_scope(source="a"), table)
        self.assertEqual(self.store.corpus_hash(), before_hash)
        added = {"id": "a:5", "title": "Running", "text": "A running amber engine.", "metadata": {"source": "a"}}
        self.store.ingest([added])
        selected = [doc for doc in [*self.documents, added] if doc["metadata"]["source"] == "a"]
        self.assert_scores(self.store.search("running amber", source="a"), reference(selected, "running amber"))
        sql = self.store.connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        self.assertIn("content=''", sql)
        self.assertEqual(len(self.store), 8)

    def test_update_move_and_delete_triggers_keep_scoped_statistics_consistent(self):
        self.store.ensure_bm25_scope(source="a")
        self.store.ensure_bm25_scope(source="b")
        moved = dict(self.documents[0], metadata={"source": "b"})
        with self.store.connection:
            self.store.connection.execute("UPDATE documents SET source=?,metadata=? WHERE id=?",
                                          ("b", json.dumps(moved["metadata"]), moved["id"]))
        remaining = [doc for doc in self.documents if doc["metadata"]["source"] == "a" and doc["id"] != moved["id"]]
        self.assert_scores(self.store.search("running amber", source="a"), reference(remaining, "running amber"))
        self.assert_scores(self.store.search("running amber", source="b"), reference([moved], "running amber"))
        with self.store.connection:
            self.store.connection.execute("DELETE FROM documents WHERE id=?", ("a:2",))
        remaining = [doc for doc in remaining if doc["id"] != "a:2"]
        self.assert_scores(self.store.search("amber", source="a"), reference(remaining, "amber"))

    def test_readonly_requires_persistent_scope_but_allowed_contexts_need_no_disk_index(self):
        self.store.ensure_bm25_scope(source="a")
        size = self.store.path.stat().st_size
        with DocumentStore(self.store.path, budget=self.budget, readonly=True) as readonly:
            expected = reference([doc for doc in self.documents if doc["metadata"]["source"] == "a"], "amber")
            self.assert_scores(readonly.search("amber", source="a"), expected)
            with self.assertRaisesRegex(ValueError, "writable"):
                readonly.search("amber", source="hotpotqa")
            selected = [doc for doc in self.documents if doc["id"] in {"wiki:1", "wiki:2"}]
            self.assert_scores(readonly.search("amber", allowed_ids={doc["id"] for doc in selected}),
                               reference(selected, "amber"))
        self.assertEqual(self.store.path.stat().st_size, size)
        self.assertFalse(any(path.suffix in {".tmp", ".temp"} for path in self.root.iterdir()))

    def test_oversized_allowed_scope_fails_before_reading_corpus_into_an_in_memory_index(self):
        with self.assertRaisesRegex(ValueError, "1000"):
            self.store.search("amber", allowed_ids={f"doc:{number}" for number in range(1001)})
        with patch("magra_rag.storage._MAX_CANDIDATE_BYTES", 10):
            with patch("magra_rag.storage.sqlite3.connect") as connect:
                with self.assertRaisesRegex(ValueError, "64 MB"):
                    self.store.search("amber", allowed_ids={"a:1"})
                connect.assert_not_called()

    def test_index_growth_is_budgeted_before_creation_and_no_partial_scope_survives(self):
        with patch.object(self.budget, "check", side_effect=BudgetExceeded("fixture full disk")) as guard:
            with self.assertRaises(BudgetExceeded):
                self.store.ensure_bm25_scope(source="a")
        self.assertGreater(guard.call_args.args[0], self.store.path.stat().st_size)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM metadata WHERE key LIKE 'bm25_scope:%'").fetchone()[0], 0)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'documents_scope_%'").fetchone()[0], 0)

    def test_hash_only_index_names_and_bound_source_literals_handle_hostile_source_names(self):
        source = "quoted'; DROP TABLE documents; --"
        doc = {"id": "odd:1", "title": "Amber", "text": "A running engine.", "metadata": {"source": source}}
        self.store.ingest([doc])
        table = self.store.ensure_bm25_scope(source=source, id_prefix="odd:")
        self.assertRegex(table, r"^documents_scope_[0-9a-f]{64}$")
        self.assert_scores(self.store.search("amber", source=source, id_prefix="odd:"), reference([doc], "amber"))
        self.assertEqual(len(self.store), 8)

    def test_global_owned_corpus_still_matches_standalone_bm25(self):
        self.assert_scores(self.store.search("running amber"), reference(self.documents, "running amber"))

    def test_result_payload_cap_retains_ranked_prefix(self):
        hits = self.store.search("running amber", source="a")
        first_two_bytes = sum(len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
                              for row in hits[:2])
        with patch("magra_rag.storage._MAX_RESULT_BYTES", first_two_bytes - 1):
            limited = self.store.search("running amber", source="a")
        self.assertEqual([row["id"] for row in limited], [hits[0]["id"]])


if __name__ == "__main__":
    unittest.main()
