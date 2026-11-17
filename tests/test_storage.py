import json
from pathlib import Path
import tempfile
import unittest

from magra_rag.resources import BudgetExceeded, ResourceBudget, tree_bytes
from magra_rag.storage import DocumentStore


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.budget = ResourceBudget(self.root, disk_bytes=100_000_000, reserve_bytes=0)
        self.store = DocumentStore(self.root / "docs.sqlite", budget=self.budget)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_sql_canonical_idempotent_and_atomic_conflicts(self):
        doc = {"id": "x", "title": "Mars", "text": "Mars is red.", "metadata": {"source": "wiki"}}
        self.assertEqual(self.store.ingest([doc])["inserted"], 1)
        self.assertEqual(self.store.ingest([doc])["already_present"], 1)
        before = self.store.corpus_hash()
        with self.assertRaises(ValueError):
            self.store.ingest([dict(doc, id="y"), dict(doc, text="Changed")])
        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.store.corpus_hash(), before)

    def test_annotation_firewall(self):
        doc = {"id": "x", "title": "", "text": "public document"}
        for extra in ({"answer": "secret"}, {"metadata": {"qrels": {"x": 1}}},
                      {"metadata": {"source": {"answer": "secret"}}}):
            with self.assertRaises(ValueError):
                self.store.ingest([{**doc, **extra}])
        self.assertEqual(len(self.store), 0)

    def test_candidate_filters_and_fts_escaping(self):
        self.store.ingest([{"id": str(i), "title": "Red Mars", "text": "Mars is a planet.",
                            "metadata": {"source": "a" if i < 2 else "b"}} for i in range(3)])
        hits = self.store.search('Mars " OR *', allowed_ids={"1", "2"}, source="a")
        self.assertEqual([d["id"] for d in hits], ["1"])
        self.assertEqual(self.store.search("Mars", allowed_ids=set()), [])
        self.assertEqual(len(self.store.search("Mars", exclude_ids={"0", "1"})), 1)

    def test_disk_checks_count_logs_and_rebuilds(self):
        (self.root / "cache.bin").write_bytes(b"x" * 5000)
        used = tree_bytes(self.root)
        self.assertGreater(used, 5000)
        self.budget.disk_limit = used + 100
        with self.assertRaises(BudgetExceeded):
            self.budget.check(101)
        with self.assertRaises(ValueError):
            self.budget.contains(self.root.parent / "outside")


if __name__ == "__main__":
    unittest.main()
