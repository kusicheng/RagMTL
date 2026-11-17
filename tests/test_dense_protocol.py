                                                                                 

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from magra_benchmark.datasets import sha256_file
from magra_benchmark.evaluation import evaluate_run
from magra_benchmark.protocol import dense_expected_dimensions, freeze_protocol, verify_protocol
from magra_benchmark.splits import make_splits, read_split_bundle, write_split_bundle
from magra_rag.index_integrity import dense_index_fingerprint, effective_query_scope
from magra_rag.retrieval import RetrievalConfig
from magra_rag.storage import DocumentStore


MODEL = "fixture/encoder@" + "a" * 40


class DenseProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = DocumentStore(self.root / "corpus.sqlite")
        self.addCleanup(self.store.close)
        self.store.ingest([
            {"id": doc_id, "title": doc_id, "text": "Owned document " + doc_id,
             "metadata": {"source": source}}
            for doc_id, source in (("science:a", "science"), ("science:b", "science"),
                                   ("context:dev", "hotpotqa"), ("context:eval", "hotpotqa"),
                                   ("noise:a", "unrelated"))
        ])
        for doc_id in ("science:a", "science:b", "context:dev", "context:eval"):
            self.vector(doc_id)
        self.corpus_hash = self.store.corpus_hash()
        self.bundle, self.path = self.root / "bundle", self.root / "protocol.json"
        rows = [
            {"id": "science:dev", "question": "Which enzyme repairs ultraviolet damage to bacterial DNA?",
             "source": "science", "corpus_source": "science", "domain": "biology", "official_split": "dev",
             "relevance": {"science:a": 1}, "answer": "PRIVATE_DEV"},
            {"id": "science:eval", "question": "Where does ocean circulation release stored heat?",
             "source": "science", "corpus_source": "science", "domain": "biology", "official_split": "test",
             "relevance": {"science:b": 1}, "answer": "PRIVATE_EVAL"},
            {"id": "context:dev-query", "question": "Which city hosted the jazz festival in nineteen sixty?",
             "source": "hotpotqa", "domain": "reference", "setting": "provided_context", "official_split": "dev",
             "context_doc_ids": ["context:dev"], "relevance": {"context:dev": 1}},
            {"id": "context:eval-query", "question": "Who designed the first pressure gauge for Alpine expeditions?",
             "source": "hotpotqa", "domain": "reference", "setting": "provided_context", "official_split": "test",
             "context_doc_ids": ["context:eval"], "relevance": {"context:eval": 1}},
        ]
        write_split_bundle(self.bundle, make_splits(rows))
        self.dev = list(read_split_bundle(self.bundle, "dev"))
        self.evaluation = list(read_split_bundle(self.bundle, "eval"))
        self.configs = {"bm25": asdict(RetrievalConfig(strategy="bm25")),
                        "candidate": asdict(RetrievalConfig(strategy="dense"))}
        self.sparse_runtime = {"generation": False, "model": None, "dense_model": None,
                               "reranker_model": None, "max_tokens": None, "reasoning_effort": None}
        self.dense_runtime = {**self.sparse_runtime, "dense_model": MODEL,
                              "dense_index_hash": self.fingerprint(self.dev)["fingerprint_hash"]}
        patcher = patch("magra_benchmark.protocol.code_hash", return_value="fixed-source-version")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reports = self.make_reports()

    def vector(self, doc_id, values=(0.25, 0.75), *, model=MODEL):
        digest = self.store.connection.execute("SELECT content_hash FROM documents WHERE id=?", (doc_id,)).fetchone()[0]
        with self.store.connection:
            self.store.connection.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?)",
                                          (doc_id, model, digest, len(values), struct.pack("<" + "f" * len(values), *values)))

    def fingerprint(self, queries, *, model=MODEL):
        return dense_index_fingerprint(self.store, model, scopes=[effective_query_scope(q) for q in queries],
                                       expected_dimensions=dense_expected_dimensions(model))

    def make_reports(self):
        labels = list(read_split_bundle(self.bundle, "dev", include_labels=True))
        outputs = [{"query_id": row["id"], "ranked_doc_ids": list(row["relevance"]),
                    "latency_ms": 1, "tokens": 0, "cost_usd": 0} for row in labels]
        return {name: evaluate_run(labels, outputs, run_metadata={
                    "config": config, "corpus_hash": self.corpus_hash,
                    "bundle_manifest_sha256": sha256_file(self.bundle / "manifest.json"),
                    "environment": {"code_hash": "fixed-source-version"}, "status": "complete",
                    "query_ids": [q["id"] for q in self.dev],
                    **(self.sparse_runtime if name == "bm25" else self.dense_runtime),
                }) for name, config in self.configs.items()}

    def freeze(self, **kwargs):
        options = {"bundle": self.bundle, "corpus_hash": self.corpus_hash, "configs": self.configs,
                   "dev_reports": self.reports, "baseline_names": ["bm25"], "candidate_name": "candidate",
                   "store": self.store}
        options.update(kwargs)
        return freeze_protocol(self.path, **options)

    def verify(self, **kwargs):
        options = {"bundle": self.bundle, "corpus_hash": self.corpus_hash,
                   "config": self.configs["candidate"], "queries": self.evaluation,
                   "runtime": {**self.dense_runtime, "dense_index_hash": self.fingerprint(self.evaluation)["fingerprint_hash"]}}
        options.update(kwargs)
        return verify_protocol(self.path, **options)

    def test_separate_dev_and_eval_scope_hashes_are_pinned_without_labels(self):
        frozen = self.freeze()
        self.assertNotEqual(frozen["runtimes"]["candidate"]["dense_index_hash"], self.dense_runtime["dense_index_hash"])
        self.assertEqual(frozen["dense_indexes"]["development"]["candidate"]["document_count"], 3)
        self.assertEqual(frozen["dense_indexes"]["evaluation"]["candidate"]["document_count"], 3)
        self.assertEqual(frozen["runtimes"]["candidate"]["dense_index_hash"],
                         self.fingerprint(self.evaluation)["fingerprint_hash"])
        self.assertIsNone(frozen["runtimes"]["bm25"]["dense_index_hash"])
        self.assertEqual(self.verify()["protocol_hash"], frozen["protocol_hash"])
        self.assertNotIn("PRIVATE_", self.path.read_text(encoding="utf-8"))

    def test_dense_freeze_requires_store_and_matching_actual_corpus(self):
        with self.assertRaisesRegex(ValueError, "require the SQL store"):
            self.freeze(store=None)
        with self.assertRaisesRegex(ValueError, "SQL store corpus"):
            self.freeze(corpus_hash="unrelated-corpus")

    def test_missing_or_stale_dev_vector_binding_is_rejected(self):
        for value in (None, "0" * 64):
            reports = deepcopy(self.reports)
            reports["candidate"]["run_metadata"]["dense_index_hash"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Development dense index differs"):
                self.freeze(dev_reports=reports)

    def test_relevant_vector_change_before_freeze_is_detected_without_corpus_change(self):
        self.vector("science:a", (0.5, 0.5))
        self.assertEqual(self.store.corpus_hash(), self.corpus_hash)
        with self.assertRaisesRegex(ValueError, "Development dense index differs"):
            self.freeze()

    def test_eval_only_vector_changes_are_frozen_separately_from_dev(self):
        original = self.fingerprint(self.evaluation)["fingerprint_hash"]
        self.vector("context:eval", (0.125, 0.875))
        frozen = self.freeze()
        self.assertEqual(self.fingerprint(self.dev)["fingerprint_hash"], self.dense_runtime["dense_index_hash"])
        self.assertNotEqual(frozen["runtimes"]["candidate"]["dense_index_hash"], original)
        self.verify()

    def test_relevant_vector_change_after_freeze_rejects_current_runtime(self):
        self.freeze()
        self.vector("context:eval", (0.5, 0.5))
        self.assertEqual(self.store.corpus_hash(), self.corpus_hash)
        with self.assertRaisesRegex(ValueError, "generation settings"):
            self.verify()

    def test_unrelated_source_build_does_not_invalidate_scoped_lock(self):
        self.vector("noise:a")
        self.freeze()
        original = self.fingerprint(self.evaluation)
        self.vector("noise:a", (1.0, 0.0), model="another/model@" + "b" * 40)
        self.assertEqual(self.fingerprint(self.evaluation), original)
        self.verify()

    def test_missing_wrong_model_and_wrong_dimensions_fail_during_freeze(self):
        with self.store.connection:
            self.store.connection.execute("DELETE FROM vectors WHERE doc_id='science:a'")
        with self.assertRaisesRegex(ValueError, "missing vector"):
            self.freeze()
        self.vector("science:a", model="another/model@" + "b" * 40)
        with self.assertRaisesRegex(ValueError, "wrong-model"):
            self.freeze()
        self.vector("science:a", (0.1, 0.2, 0.3))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.freeze()

    def test_unknown_duplicate_or_mismatched_dev_membership_is_rejected(self):
        for ids in (["science:eval"], [self.dev[0]["id"], self.dev[0]["id"]], [self.dev[0]["id"]], []):
            reports = deepcopy(self.reports)
            reports["candidate"]["run_metadata"]["query_ids"] = ids
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "query_ids"):
                self.freeze(dev_reports=reports)
        reports = deepcopy(self.reports)
                                                                                               
        reports["candidate"]["run_metadata"]["query_ids"][0] = "science:eval"
        reports["candidate"]["per_query"][0]["query_id"] = "science:eval"
        with self.assertRaisesRegex(ValueError, "unknown public DEV"):
            self.freeze(dev_reports=reports)

    def test_duplicate_scored_ids_and_missing_dense_model_are_rejected(self):
        reports = deepcopy(self.reports)
        reports["candidate"]["per_query"][1]["query_id"] = reports["candidate"]["per_query"][0]["query_id"]
        with self.assertRaisesRegex(ValueError, "query_ids"):
            self.freeze(dev_reports=reports)
        reports = deepcopy(self.reports)
        reports["candidate"]["run_metadata"]["dense_model"] = None
        with self.assertRaisesRegex(ValueError, "missing its embedding model"):
            self.freeze(dev_reports=reports)

    def test_dev_lookup_uses_only_public_records(self):
        def public_only(*args, **kwargs):
            self.assertFalse(kwargs.get("include_labels", False))
            return read_split_bundle(*args, **kwargs)
        with patch("magra_benchmark.protocol.read_split_bundle", side_effect=public_only):
            self.freeze()

    def test_runtime_needs_eval_index_hash_even_with_identical_model(self):
        self.freeze()
        for runtime in ({**self.dense_runtime}, {key: value for key, value in self.dense_runtime.items()
                                              if key != "dense_index_hash"}):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, "generation settings"):
                self.verify(runtime=runtime)

    def test_model_revisions_must_be_40_hex_without_restricting_generation_ids(self):
        for field in ("dense_model", "reranker_model"):
            for model_id in ("fixture/model@main", "fixture/model@UNPINNED", "fixture/model@abc123", "fixture/model"):
                reports = deepcopy(self.reports)
                reports["candidate"]["run_metadata"][field] = model_id
                with self.subTest(field=field, model=model_id), self.assertRaisesRegex(ValueError, "40-hex"):
                    self.freeze(dev_reports=reports)
        for report in self.reports.values():
            report["run_metadata"].update(generation=True, model="remote/model@main", max_tokens=512)
        self.freeze()

    def test_native_qwen_identity_suffix_is_accepted(self):
        model = "Qwen/fixture@" + "a" * 40 + "#qwen-native-v1:" + "b" * 16
        for doc_id in ("science:a", "science:b", "context:dev", "context:eval"):
            self.vector(doc_id, model=model)
        self.reports["candidate"]["run_metadata"].update(
            dense_model=model, dense_index_hash=self.fingerprint(self.dev, model=model)["fingerprint_hash"])
        frozen = self.freeze()
        self.assertEqual(frozen["runtimes"]["candidate"]["dense_model"], model)

    def test_known_dimensions_are_explicit_and_generic_dimensions_remain_inferred(self):
        self.assertEqual(dense_expected_dimensions("BAAI/bge-small-en-v1.5@" + "a" * 40), 384)
        self.assertEqual(dense_expected_dimensions("Qwen/Qwen3-Embedding-0.6B@" + "a" * 40 +
                                                   "#qwen-native-v1:" + "b" * 16), 1024)
        self.assertIsNone(dense_expected_dimensions(MODEL))
        self.assertEqual(self.fingerprint(self.dev)["dimensions"], 2)

    def test_same_model_and_scope_fingerprints_are_cached_within_freeze(self):
        self.configs["alternative"] = {**self.configs["candidate"], "context_bytes": 24000}
        self.reports["alternative"] = deepcopy(self.reports["candidate"])
        self.reports["alternative"]["run_metadata"]["config"] = self.configs["alternative"]
        with patch("magra_rag.index_integrity.dense_index_fingerprint", wraps=dense_index_fingerprint) as fingerprint:
            frozen = self.freeze()
        self.assertEqual(fingerprint.call_count, 2)
        self.assertEqual(frozen["runtimes"]["candidate"]["dense_index_hash"],
                         frozen["runtimes"]["alternative"]["dense_index_hash"])


if __name__ == "__main__":
    unittest.main()
