import bz2
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from magra_benchmark.datasets import (BEIR_DATASETS, HOTPOT_DATASETS, hotpot_document_id,
    iter_beir_documents, iter_hotpot, iter_hotpot_documents, iter_hotpot_wikipedia, sha256_file)
from magra_benchmark.integrity import audit_corpus, record_import
from magra_rag.storage import DocumentStore


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = DocumentStore(self.root / "corpus.sqlite")
        self.addCleanup(self.store.close)
        patcher = patch.dict(BEIR_DATASETS, {"scifact": {**BEIR_DATASETS["scifact"], "documents": 2}})
        patcher.start()
        self.addCleanup(patcher.stop)

    def beir(self, *, manifest=True):
        path = self.root / "scifact"
        path.mkdir()
        documents = [{"_id": "one", "title": "First", "text": "First public evidence."},
                     {"_id": "two", "title": "Second", "text": "Second public evidence."}]
        (path / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8")
        if manifest:
            acquisition = {"format": "magra-acquisition-v1", "dataset": "scifact", "complete_corpus": True,
                           "archive_md5": BEIR_DATASETS["scifact"]["md5"], "corpus_documents": 2,
                           "files_sha256": {"corpus.jsonl": sha256_file(path / "corpus.jsonl")}}
            (path / "acquisition.json").write_text(json.dumps(acquisition), encoding="utf-8")
        return path

    def query(self, source="scifact", **updates):
        return {"id": "private-query-id", "question": "PRIVATE_QUESTION", "answer": "PRIVATE_ANSWER",
                "source": source, "corpus_source": source, **updates}

    def test_known_beir_receipt_verifies_full_source_and_is_idempotent(self):
        path = self.beir()
        self.store.ingest(iter_beir_documents(path))
        first = record_import(self.store, format="beir", source_path=path, imported_count=2)
        second = record_import(self.store, format="beir", source_path=path, imported_count=2)
        self.assertEqual(first["receipt_hash"], second["receipt_hash"])
        audit = audit_corpus(self.store, [self.query()])
        self.assertEqual(audit["status"], "verified")
        self.assertEqual(audit["scopes"]["scifact"]["documents"], 2)
        self.assertEqual(audit["completed_import_receipts"], 1)
        metadata = " ".join(row["value"] for row in self.store.connection.execute("SELECT value FROM metadata"))
        self.assertNotIn("PRIVATE_", metadata)
        self.assertNotIn("PRIVATE_", json.dumps(audit))

    def test_count_alone_without_completed_receipt_is_insufficient(self):
        self.store.ingest(iter_beir_documents(self.beir()))
        with self.assertRaisesRegex(ValueError, "completed import receipt"):
            audit_corpus(self.store, [self.query()])

    def test_partial_committed_import_cannot_create_receipt(self):
        path = self.beir()
        self.store.ingest([next(iter_beir_documents(path))])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            record_import(self.store, format="beir", source_path=path, imported_count=1)
        self.assertEqual(self.store.connection.execute("SELECT count(*) FROM metadata").fetchone()[0], 0)
        audit = audit_corpus(self.store, [self.query()], strict=False)
        self.assertEqual(audit["status"], "failed")

    def test_known_beir_requires_acquisition_manifest_and_source_hash(self):
        path = self.beir(manifest=False)
        self.store.ingest(iter_beir_documents(path))
        with self.assertRaisesRegex(ValueError, "acquisition manifest"):
            record_import(self.store, format="beir", source_path=path)

    def test_modified_source_does_not_receive_receipt(self):
        path = self.beir()
        self.store.ingest(iter_beir_documents(path))
        with (path / "corpus.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            record_import(self.store, format="beir", source_path=path)

    def test_wrong_count_and_wrong_sql_content_refused(self):
        path = self.beir()
        documents = list(iter_beir_documents(path))
        documents[0]["text"] = "A different document with the same identifier"
        self.store.ingest(documents)
        with self.assertRaisesRegex(ValueError, "SQL content differs"):
            record_import(self.store, format="beir", source_path=path, imported_count=2)

    def test_incorrect_ingest_return_count_refused(self):
        path = self.beir()
        self.store.ingest(iter_beir_documents(path))
        with self.assertRaisesRegex(ValueError, "Accepted import count"):
            record_import(self.store, format="beir", source_path=path, imported_count=1)

    def test_deleted_document_invalidates_completed_import(self):
        path = self.beir()
        self.store.ingest(iter_beir_documents(path))
        record_import(self.store, format="beir", source_path=path)
        with self.store.connection:
            self.store.connection.execute("DELETE FROM documents WHERE id='scifact:two'")
        with self.assertRaisesRegex(ValueError, "completeness audit"):
            audit_corpus(self.store, [self.query()])

    def test_unrelated_owned_source_does_not_invalidate_benchmark_receipt(self):
        path = self.beir()
        self.store.ingest(iter_beir_documents(path))
        record_import(self.store, format="beir", source_path=path)
        self.store.ingest([{"id": "owned:one", "title": "Owned", "text": "Extra owned data", "metadata": {"source": "owned"}}])
        self.assertEqual(audit_corpus(self.store, [self.query()])["status"], "verified")

    def hotpot(self):
        path = self.root / "hotpot.json"
        rows = [{"_id": "q1", "question": "PRIVATE_QUESTION", "answer": "PRIVATE_ANSWER",
                 "supporting_facts": [["A", 0]], "context": [["A", ["Public evidence."]], ["B", ["A public distractor."]]]}]
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_all_provided_context_documents_required_and_annotations_absent(self):
        path = self.hotpot()
        documents = list(iter_hotpot_documents(path))
        self.store.ingest(documents[:1])
        queries = list(iter_hotpot(path))
        with self.assertRaisesRegex(ValueError, "missing candidate"):
            audit_corpus(self.store, queries)
        self.store.ingest(documents[1:])
        receipt = record_import(self.store, format="hotpot_context", source_path=path, imported_count=2)
        audit = audit_corpus(self.store, queries)
        self.assertEqual(audit["status"], "verified")
        self.assertEqual(receipt["documents"], 2)
        metadata = " ".join(row["value"] for row in self.store.connection.execute("SELECT value FROM metadata"))
        self.assertNotIn("PRIVATE_", metadata)
        self.assertNotIn("PRIVATE_", json.dumps(audit))

    def test_context_corpus_cannot_pass_as_full_wikipedia(self):
        path = self.hotpot()
        self.store.ingest(iter_hotpot_documents(path))
        record_import(self.store, format="hotpot_context", source_path=path)
        with self.assertRaisesRegex(ValueError, "independent Wikipedia"):
            audit_corpus(self.store, list(iter_hotpot(path, "full_corpus")))

    def test_wiki_ids_cannot_substitute_for_provided_context(self):
        self.store.ingest([{"id": hotpot_document_id("A"), "title": "A", "text": "Evidence",
                            "metadata": {"source": "hotpotqa"}}])
        query = self.query("hotpotqa", setting="provided_context", context_doc_ids=[hotpot_document_id("A")])
        with self.assertRaisesRegex(ValueError, "wrong corpus namespace"):
            audit_corpus(self.store, [query])

    def wiki(self):
        directory = self.root / "wiki"
        directory.mkdir()
        path = directory / "wiki_00.bz2"
        with bz2.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": 1, "title": "A", "text": ["Evidence."]}) + "\n")
        manifest = {"complete_corpus": True, "corpus_documents": 1,
                    "archive_md5": HOTPOT_DATASETS["wiki"]["md5"], "files_sha256": {path.name: sha256_file(path)}}
        (directory / "acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def test_full_wikipedia_requires_completed_verified_import(self):
        directory = self.wiki()
        self.store.ingest(iter_hotpot_wikipedia(directory))
        query = self.query("hotpotqa", setting="full_corpus")
        with self.assertRaisesRegex(ValueError, "import receipt"):
            audit_corpus(self.store, [query])
        record_import(self.store, format="hotpot_wiki", source_path=directory, imported_count=1)
        self.assertEqual(audit_corpus(self.store, [query])["status"], "verified")

    def test_truncated_wiki_manifest_count_cannot_be_certified(self):
        directory = self.wiki()
        manifest = json.loads((directory / "acquisition.json").read_text())
        manifest["corpus_documents"] = 2
        (directory / "acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.store.ingest(iter_hotpot_wikipedia(directory))
        with self.assertRaisesRegex(ValueError, "unexpected document count"):
            record_import(self.store, format="hotpot_wiki", source_path=directory)

    def test_owned_jsonl_records_source_digest_without_inventing_catalog_size(self):
        path = self.root / "owned.jsonl"
        document = {"id": "owned:a", "title": "Owned", "text": "Owned public evidence", "metadata": {"source": "owned"}}
        path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        self.store.ingest([document])
        receipt = record_import(self.store, format="jsonl", source_path=path, imported_count=1)
        self.assertEqual(receipt["source_sha256"], sha256_file(path))
        self.assertIsNone(receipt["catalog_documents"])
        self.assertEqual(audit_corpus(self.store, [self.query("owned")])["status"], "verified")

    def test_modified_receipt_detected(self):
        path = self.beir()
        self.store.ingest(iter_beir_documents(path))
        record_import(self.store, format="beir", source_path=path)
        with self.store.connection:
            row = self.store.connection.execute("SELECT key,value FROM metadata").fetchone()
            receipt = json.loads(row["value"])
            receipt["documents"] = 1
            self.store.connection.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps(receipt), row["key"]))
        with self.assertRaisesRegex(ValueError, "receipt integrity"):
            audit_corpus(self.store, [self.query()])


if __name__ == "__main__":
    unittest.main()
