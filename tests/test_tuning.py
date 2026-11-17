from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from magra_benchmark.datasets import sha256_file
from magra_benchmark.protocol import code_hash, freeze_protocol
from magra_benchmark.splits import write_split_bundle
from magra_benchmark.tuning import default_tuning_configs, tune_retrieval
from magra_rag.retrieval import RetrievalConfig
from magra_rag.storage import DocumentStore


class DenseStub:
    model_id = "fixture-dense@pinned"

    def __init__(self, store):
        self.store = store
        self.calls = 0

    def search(self, question, *, k, **filters):
        self.calls += 1
        return self.store.search(question, k=k, **filters)


class RerankerStub:
    model_id = "fixture-reranker@pinned"

    def __init__(self):
        self.calls = 0

    def rerank(self, question, candidates):
        self.calls += 1
        return candidates


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = DocumentStore(Path(self.temporary.name) / "documents.sqlite")
        self.store.ingest([
            {"id": "relevant", "title": "Aurora", "text": "Aurora was founded in 1998."},
            {"id": "distractor", "title": "Cars", "text": "Cars have four wheels."},
            {"id": "unjudged", "title": "Borealis", "text": "A northern region."},
        ])
        self.queries = [{"id": "dev-1", "question": "When was Aurora founded?", "split": "dev",
                         "source": "scifact", "domain": "science", "relevance": {"relevant": 1},
                         "answer": "1998", "supporting_facts": [{"doc_id": "relevant", "sentence_id": 0}]}]
        self.configs = {"base": RetrievalConfig(strategy="bm25"), "candidate": RetrievalConfig(strategy="magra")}

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def tune(self, queries=None, configs=None, **kwargs):
        return tune_retrieval(self.store, self.queries if queries is None else queries,
                              self.configs if configs is None else configs,
                              bundle_manifest_sha256="a" * 64, **kwargs)

    def test_complete_grid_retains_corpus_and_freeze_metadata(self):
        before = self.store.corpus_hash()
        result = self.tune()
        self.assertEqual(self.store.corpus_hash(), before)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 3)
        self.assertEqual(set(result["development_reports"]), {"base", "candidate"})
        report = result["development_reports"]["base"]
        metadata = report["run_metadata"]
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["config"], asdict(self.configs["base"]))
        self.assertEqual(metadata["corpus_hash"], before)
        self.assertEqual(metadata["bundle_manifest_sha256"], "a" * 64)
        self.assertEqual(metadata["environment"]["code_hash"], code_hash())
        self.assertIsNone(metadata["max_tokens"])
        self.assertFalse(metadata["generation"])
        self.assertIn("resources", metadata)
        self.assertEqual(next(iter(report["groups"].values()))["metrics"]["ndcg@10"]["value"], 1)

    def test_private_labels_are_stripped_before_engine(self):
        from magra_rag.engine import MagraEngine
        original = MagraEngine.run
        observed = []

        def audited(engine, query, *, retrieval_only=False):
            observed.append(query)
            self.assertTrue(retrieval_only)
            self.assertNotIn("answer", query)
            self.assertNotIn("relevance", query)
            self.assertNotIn("supporting_facts", query)
            return original(engine, query, retrieval_only=retrieval_only)

        public = [{key: value for key, value in self.queries[0].items()
                   if key not in {"relevance", "answer", "supporting_facts"}}]
        private = [{"id": "dev-1", "relevance": {"relevant": 1}, "answer": "1998"}]
        with patch("magra_benchmark.tuning.MagraEngine.run", audited):
            self.tune(public, private_labels=private)
        self.assertEqual(len(observed), 2)

    def test_eval_test_and_heldout_sources_cannot_enter_tuning(self):
        for extra in ({"split": "eval"}, {"split": "test"}, {"source": "arguana"},
                      {"held_out": True}, {"official_split": "test"}, {"domain": "heldout-domain"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.tune([{**self.queries[0], **extra}], heldout_domains=["heldout-domain"])

    def test_sampling_changes_queries_not_documents_and_is_order_invariant(self):
        queries = [{**self.queries[0], "id": f"dev-{i}"} for i in range(8)]
        first = self.tune(queries, limit_per_stratum=2, seed="fixed")
        second = self.tune(list(reversed(queries)), limit_per_stratum=2, seed="fixed")
        self.assertEqual(first["manifest"]["query_ids"], second["manifest"]["query_ids"])
        self.assertEqual(first["manifest"]["sampled_query_count"], 2)
        self.assertTrue(first["manifest"]["full_corpus_retained"])
        for report in first["development_reports"].values():
            self.assertEqual(report["query_count"], 2)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 3)

    def test_full_corpus_cannot_be_restricted_using_gold_document_ids(self):
        with self.assertRaises(ValueError):
            self.tune([{**self.queries[0], "allowed_doc_ids": ["relevant"]}])
        provided = {**self.queries[0], "setting": "provided_context", "context_doc_ids": ["distractor"]}
        result = self.tune([provided])
        report = result["development_reports"]["base"]
        self.assertEqual(next(iter(report["groups"].values()))["metrics"]["recall@10"]["value"], 0)

    def test_optional_backends_are_included_only_when_explicit_and_no_requested_candidate_skips(self):
        grid = default_tuning_configs()
        self.assertEqual({config.title_weight for config in grid.values() if config.strategy == "bm25"},
                         {0.5, 1, 2, 4})
        self.assertTrue({"magra", "magra_no_bridge", "magra_no_selection"}.issubset(grid))
        dense, reranker = DenseStub(self.store), RerankerStub()
        optional = default_tuning_configs(dense=dense, reranker=reranker)
        self.assertTrue({"dense", "hybrid_title_0.5", "hybrid_rerank_title_4", "magra_dense_rerank"}.issubset(optional))
        explicit = {"dense": RetrievalConfig(strategy="dense"), "rerank": RetrievalConfig(strategy="rerank")}
        with self.assertRaises(ValueError):
            self.tune(configs=explicit)
        result = self.tune(configs=explicit, dense=dense, reranker=reranker)
        self.assertEqual(set(result["development_reports"]), set(explicit))
        self.assertGreater(dense.calls, 0)
        self.assertGreater(reranker.calls, 0)
        self.assertEqual(result["development_reports"]["rerank"]["run_metadata"]["reranker_model"], reranker.model_id)

    def test_candidate_runtime_error_aborts_grid(self):
        dense = DenseStub(self.store)
        dense.search = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fixture backend failure"))
        with self.assertRaisesRegex(RuntimeError, "fixture backend failure"):
            self.tune(configs={"good": RetrievalConfig(strategy="bm25"),
                               "failing": RetrievalConfig(strategy="dense")}, dense=dense)

    def test_public_only_and_conflicting_private_labels_rejected(self):
        public = [{key: value for key, value in self.queries[0].items() if key != "relevance"}]
        with self.assertRaises(ValueError):
            self.tune(public)
        with self.assertRaises(ValueError):
            self.tune(private_labels=[{"id": "dev-1", "split": "eval", "relevance": {"relevant": 1}}])

    def test_reports_can_freeze_protocol_without_metadata_repair(self):
        bundle = Path(self.temporary.name) / "bundle"
        write_split_bundle(bundle, {"train": [], "dev": self.queries, "eval": []})
        result = tune_retrieval(self.store, self.queries, self.configs,
                                bundle_manifest_sha256=sha256_file(bundle / "manifest.json"))
        frozen = freeze_protocol(Path(self.temporary.name) / "protocol.json", bundle=bundle,
                                 corpus_hash=self.store.corpus_hash(), configs=result["configs"],
                                 dev_reports=result["development_reports"], baseline_names=["base"],
                                 candidate_name="candidate")
        self.assertEqual(frozen["baseline_names"], ["base"])

    def test_corpus_mutation_during_tuning_invalidates_results(self):
        def changed(name, index, total):
            if name == "base":
                self.store.ingest([{"id": "late-document", "title": "Late", "text": "Changed corpus."}])

        with self.assertRaisesRegex(RuntimeError, "SQL corpus changed"):
            self.tune(on_progress=changed)


if __name__ == "__main__":
    unittest.main()
