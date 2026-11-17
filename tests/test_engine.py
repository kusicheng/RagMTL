from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from magra_rag.engine import MagraEngine
from magra_rag.generation import validate_answer, SpendLedger, GenerationError
from magra_rag.resources import ResourceBudget
from magra_rag.retrieval import Retriever, RetrievalConfig, reciprocal_rank_fusion
from magra_rag.storage import DocumentStore


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = DocumentStore(self.root / "docs.sqlite")
        self.store.ingest([
            {"id": "a", "title": "Aurora", "text": "Aurora was founded in 1998. The city is in Borealis.",
             "metadata": {"sentences": ["Aurora was founded in 1998.", "The city is in Borealis."]}},
            {"id": "b", "title": "Borealis", "text": "Borealis is a northern region."},
            {"id": "c", "title": "Cars", "text": "Cars have four wheels."},
        ])

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_context_is_bounded_and_citations_traceable(self):
        engine = MagraEngine(Retriever(self.store, RetrievalConfig(context_bytes=300)))
        result = engine.run({"id": "q", "question": "When was Aurora founded?"})
        self.assertIn("a", result["ranked_doc_ids"])
        self.assertLessEqual(result["context_bytes"], 300)
        self.assertNotIn("answer", result)
        for u in result["evidence"]:
            self.assertEqual(self.store.get(u["doc_id"])["metadata"]["sentences"][u["sentence_id"]], u["text"])

    def test_provided_context_cannot_search_other_documents(self):
        engine = MagraEngine(Retriever(self.store))
        query = {"id": "q", "question": "Aurora founded?", "setting": "provided_context"}
        with self.assertRaises(ValueError):
            engine.run(query)
        result = engine.run({**query, "allowed_doc_ids": ["c"]})
        self.assertEqual(result["ranked_doc_ids"], [])
        with self.assertRaises(ValueError):
            engine.run({**query, "answer": "1998"})

    def test_structural_grounding_rejects_unsupplied_citations(self):
        evidence = [{"doc_id": "a", "sentence_id": 0}]
        answer = {"answer": "1998", "citations": [{"doc_id": "a", "sentence_id": 0}],
                  "confidence": 0.8, "abstained": False, "reason": "Supported"}
        self.assertEqual(validate_answer(answer, evidence)["answer"], "1998")
        with self.assertRaises(ValueError):
            validate_answer({**answer, "citations": [{"doc_id": "a", "sentence_id": 1}]}, evidence)
        with self.assertRaises(ValueError):
            validate_answer({**answer, "confidence": float("nan")}, evidence)
        with self.assertRaises(ValueError):
            validate_answer({**answer, "abstained": True}, evidence)

    def test_persistent_spend_reservations(self):
        budget = ResourceBudget(self.root)
        ledger = SpendLedger(self.root / "spend.sqlite", 1.0, budget=budget)
        first = ledger.reserve(0.6)
        ledger = SpendLedger(self.root / "spend.sqlite", 1.0, budget=budget)
        with self.assertRaises(GenerationError):
            ledger.reserve(0.5)
        ledger.settle(first, 0.1)
        ledger.reserve(0.5)
        self.assertAlmostEqual(ledger.summary()["accounted_usd"], 0.6)

    def test_rrf_deduplicates_each_list(self):
        result = reciprocal_rank_fusion([[{"id": "a"}, {"id": "a"}], [{"id": "b"}]])
        self.assertEqual(result[0]["score"], result[1]["score"])

    def test_local_retrieval_timing_is_separate_from_remote_generation(self):
        retriever = Mock(store=Mock(budget=Mock()))
        retriever.retrieve.return_value = {"ranked_doc_ids": [], "evidence": [], "context_bytes": 2,
                                           "coverage": 0.0, "trace": [], "config": {}}
        generator = Mock()
        generator.generate.return_value = {"answer": "", "abstained": True, "latency_ms": 8000}
        with patch("magra_rag.engine.time.perf_counter", side_effect=[0, 2, 2, 10, 10]):
            result = MagraEngine(retriever, generator).run({"id": "q", "question": "Several related questions?"})
        self.assertEqual(result["retrieval_latency_ms"], 2000)
        self.assertEqual(result["generation_latency_ms"], 8000)
        self.assertEqual(result["latency_ms"], 10000)
        with patch("magra_rag.engine.time.perf_counter", side_effect=[0, 2, 2]):
            retrieval_only = MagraEngine(retriever, generator).run(
                {"id": "q", "question": "Question?"}, retrieval_only=True)
        self.assertIsNone(retrieval_only["generation_latency_ms"])
        self.assertEqual(generator.generate.call_count, 1)

    def test_total_latency_includes_final_resource_check(self):
        for retrieval_only in (True, False):
            with self.subTest(retrieval_only=retrieval_only):
                clock = {"seconds": 0.0}
                resources = {"scope": "final resource check"}
                retriever = Mock(store=Mock(budget=Mock()))
                generator = Mock()

                def retrieve(*args, **kwargs):
                    clock["seconds"] += 2.0
                    return {"ranked_doc_ids": [], "evidence": [], "context_bytes": 2,
                            "coverage": 0.0, "trace": [], "config": {}}

                def generate(*args, **kwargs):
                    clock["seconds"] += 8.0
                    return {"answer": "", "abstained": True, "latency_ms": 8000}

                def check():
                    clock["seconds"] += 3.0
                    return resources

                retriever.retrieve.side_effect = retrieve
                retriever.store.budget.check.side_effect = check
                generator.generate.side_effect = generate
                with patch("magra_rag.engine.time.perf_counter", side_effect=lambda: clock["seconds"]):
                    result = MagraEngine(retriever, generator).run(
                        {"id": "q", "question": "Question?"}, retrieval_only=retrieval_only)
                self.assertEqual(result["retrieval_latency_ms"], 2000)
                self.assertEqual(result["generation_latency_ms"], None if retrieval_only else 8000)
                self.assertEqual(result["latency_ms"], 5000 if retrieval_only else 13000)
                self.assertIs(result["resources"], resources)
                retriever.store.budget.check.assert_called_once_with()
                self.assertEqual(generator.generate.call_count, 0 if retrieval_only else 1)


if __name__ == "__main__":
    unittest.main()
