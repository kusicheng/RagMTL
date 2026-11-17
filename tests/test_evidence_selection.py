from collections import Counter
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch

from magra_rag import retrieval
from magra_rag.retrieval import (
    RetrievalConfig, Retriever, evidence_context, fixed_evidence,
    query_focused_evidence, select_evidence,
)
from magra_rag.storage import sentences


def document(doc_id, units, *, title="Reference", metadata=None):
    return {"id": doc_id, "title": title, "text": "\n".join(units),
            "metadata": {"sentences": units, **(metadata or {})}}


class EvidenceSelectionTests(unittest.TestCase):
    def config(self, **kwargs):
        return RetrievalConfig(top_k=1, candidate_k=1, initial_k=1,
                               bridge_terms=0, coverage_stop=0, **kwargs)

    def assert_exact_size(self, result, limit):
        self.assertEqual(result["context_bytes"], len(evidence_context(result["evidence"]).encode()))
        self.assertLessEqual(result["context_bytes"], limit)
        keys = [(unit["doc_id"], unit["sentence_id"]) for unit in result["evidence"]]
        self.assertEqual(len(keys), len(set(keys)))

    def test_streaming_fallback_and_metadata_preserve_canonical_ids(self):
        text = " \n\nFirst!  Second?\r\n \nThird without punctuation\n\n第四。\nLast. "
        doc = {"id": "d", "title": "Title", "text": text, "metadata": {}}
        self.assertEqual(list(retrieval._sentence_texts(doc)), list(enumerate(sentences(text))))
        doc["metadata"]["sentences"] = ["", "  First! ", "\n", "Second?", " ", "Tail."]
        self.assertEqual(list(retrieval._sentence_texts(doc)),
                         [(1, "  First! "), (3, "Second?"), (5, "Tail.")])
        self.assertEqual([unit["sentence_id"] for unit in retrieval._units(doc)], [1, 3, 5])

    def test_late_support_in_last_document_is_reachable_for_both_methods(self):
        docs = [document(f"early-{i}", ["Unrelated filler."] * 250) for i in range(10)]
        tail = ["Unrelated filler."] * 2505
        tail[2502], tail[2503] = "Quartz divisor calibration constants.", "1.2 and 0.75 respectively."
        docs.append(document("last", tail))
        c = self.config(context_bytes=1500)
        for selector in (query_focused_evidence, select_evidence):
            with self.subTest(selector=selector.__name__):
                result = selector("quartz divisor calibration constants", docs, c)
                ids = {(u["doc_id"], u["sentence_id"]) for u in result["evidence"]}
                self.assertIn(("last", 2502), ids)
                self.assertIn(("last", 2503), ids)
                self.assertEqual(result["coverage"], 1)
                self.assert_exact_size(result, c.context_bytes)

    def test_zero_overlap_value_neighbor_is_atomic_including_metadata_gaps(self):
        doc = document("d", ["Intro.", "", "Quartz constants.", " ", "1.2 and 0.75."])
        expected = list(retrieval._units(doc))
        budget = len(evidence_context(expected).encode())
        for selector in (query_focused_evidence, select_evidence):
            with self.subTest(selector=selector.__name__):
                result = selector("quartz constants", [doc], self.config(context_bytes=budget))
                self.assertEqual([u["sentence_id"] for u in result["evidence"]], [0, 2, 4])
                self.assert_exact_size(result, budget)
                                                                             
                                                                             
                small = selector("quartz constants", [doc], self.config(context_bytes=budget - 1))
                self.assertEqual(small["evidence"], [])
                self.assertEqual(small["coverage"], 0)

    def test_title_only_matches_never_increase_coverage(self):
        doc = document("d", [f"Quiet unrelated prose {i}." for i in range(500)], title="Quartz constants")
        for selector in (query_focused_evidence, select_evidence):
            result = selector("quartz constants", [doc], self.config(context_bytes=1500))
            self.assertEqual(result["coverage"], 0)
            self.assertFalse(result["coverage_is_confidence"])
            self.assert_exact_size(result, 1500)

    def test_full_scan_is_bounded_and_deterministic_without_prefix_ties(self):
        docs = [document(f"doc-{i}", ["Shared body term."] * 350) for i in range(12)]
        original, seen = retrieval._sentence_texts, Counter()

        def observed(doc):
            for sid, text in original(doc):
                seen[doc["id"]] += 1
                yield sid, text

        with patch.object(retrieval, "_sentence_texts", side_effect=observed):
            pool = retrieval._evidence_pool("shared term", docs, 1000)
        _, units, _, _, _, windows = pool
        self.assertLessEqual(len(windows), 666)
        self.assertLessEqual(len(units), 2000)
        self.assertTrue(all(n <= 200 for n in Counter(key[0] for key in units).values()))
        self.assertTrue(all(seen[doc["id"]] >= 700 for doc in docs))
        self.assertTrue(any(key[1] > 200 for key in units))
        c = self.config(context_bytes=1500)
        first = query_focused_evidence("shared term", docs, c)
        second = query_focused_evidence("shared term", list(reversed(docs)), c)
        self.assertEqual(first, second)
        self.assert_exact_size(first, 1500)

    def test_overlapping_windows_deduplicate_and_unicode_size_is_exact(self):
        doc = document("unicode", ["Préface.", "石英 quartz constant.", "Quartz value: 雪.", "尾部."])
        for limit in (128, 350, 800, 1500):
            for selector in (query_focused_evidence, select_evidence):
                with self.subTest(limit=limit, selector=selector.__name__):
                    result = selector("quartz constant value", [doc], self.config(context_bytes=limit))
                    self.assert_exact_size(result, limit)
                    for unit in result["evidence"]:
                        self.assertEqual(unit["text"], doc["metadata"]["sentences"][unit["sentence_id"]])

    def test_oversized_neighbor_is_not_truncated_or_silently_dropped(self):
        doc = document("d", ["Quartz constants.", "x" * 10_000])
        for selector in (query_focused_evidence, select_evidence):
            result = selector("quartz constants", [doc], self.config(context_bytes=500))
            self.assertEqual(result["evidence"], [])
            self.assert_exact_size(result, 500)

    def test_labels_answers_and_model_metadata_do_not_affect_selection(self):
        doc = document("d", ["Intro.", "Quartz constants.", "1.2 and 0.75."])
        polluted = {**doc, "answer": "invented", "supporting_facts": [["d", 0]],
                    "metadata": {**doc["metadata"], "labels": [1], "model": "secret-model", "answer": "wrong"}}
        for selector in (query_focused_evidence, select_evidence):
            self.assertEqual(selector("quartz constants", [doc], self.config()),
                             selector("quartz constants", [polluted], self.config()))

    def test_empty_candidates_and_duplicate_documents(self):
        doc = document("d", ["Quartz constants.", "1.2 and 0.75."])
        for selector in (query_focused_evidence, select_evidence):
            self.assertEqual(selector("quartz", [], self.config())["evidence"], [])
            self.assertEqual(selector("quartz", [doc], self.config()),
                             selector("quartz", [doc, doc], self.config()))

    def test_auto_strong_baseline_and_explicit_prefix_ablation(self):
        units = ["Unrelated filler."] * 500 + ["Quartz constants.", "1.2 and 0.75."]
        doc = document("d", units)
        store = Mock()
        store.search.return_value = [doc]
        base = self.config(strategy="bm25", context_bytes=1500)
        results = {}
        for name, config in {
            "baseline": base,
            "magra": replace(base, strategy="magra"),
            "prefix": replace(base, evidence_selection="prefix"),
            "no_selection": replace(base, strategy="magra_no_selection", evidence_selection="query_focused"),
            "explicit_marginal": replace(base, evidence_selection="marginal_gain"),
        }.items():
            results[name] = Retriever(store, config).retrieve("quartz constants")
            self.assertEqual(results[name]["ranked_doc_ids"], ["d"])
            self.assert_exact_size(results[name], 1500)
        self.assertIn(500, [u["sentence_id"] for u in results["baseline"]["evidence"]])
        self.assertIn(500, [u["sentence_id"] for u in results["magra"]["evidence"]])
        self.assertEqual(results["baseline"]["trace"][-1]["method"], "query_focused")
        self.assertEqual(results["magra"]["trace"][-1]["method"], "marginal_gain")
        for name in ("prefix", "no_selection"):
            self.assertEqual(results[name]["trace"][-1]["method"], "prefix")
            self.assertTrue(all(u["sentence_id"] < 200 for u in results[name]["evidence"]))
        self.assertEqual(results["explicit_marginal"]["trace"][-1]["method"], "marginal_gain")
        with self.assertRaisesRegex(ValueError, "evidence selection"):
            replace(base, evidence_selection="unknown")

    def test_legacy_prefix_keeps_its_caps_and_exact_bytes(self):
        docs = [document(str(i), ["x"] * 250) for i in range(12)]
        result = fixed_evidence(docs, 1_000_000)
        self.assertEqual(len(result["evidence"]), 2000)
        self.assertEqual(max(u["sentence_id"] for u in result["evidence"]), 199)
        self.assertIsNone(result["coverage"])
        self.assert_exact_size(result, 1_000_000)

    def reuse_config(self, **kwargs):
        return RetrievalConfig(top_k=2, candidate_k=3, initial_k=2,
                               context_bytes=1500, **kwargs)

    def reuse_documents(self):
        return [document("a", ["Quartz constants.", "Values are 1.2 and 0.75."], title="Zephyr"),
                document("b", ["Quartz constants are documented."], title="Reference")]

    def assert_real_final_pack(self, result, question, config, selector=select_evidence):
                                                                               
                                                                               
        packed = selector(question, result["documents"], config)
        for key, value in packed.items():
            self.assertEqual(result[key], value)
        self.assert_exact_size(result, config.context_bytes)

    def test_unchanged_initial_pack_is_reused_per_request(self):
        docs, store = self.reuse_documents(), Mock()
        store.search.return_value = docs
        config = self.reuse_config()
        with patch.object(retrieval, "select_evidence", wraps=select_evidence) as selector:
            first = Retriever(store, config).retrieve("quartz constants")
            self.assertEqual(selector.call_count, 1)
            second = Retriever(store, config).retrieve("quartz constants")
            self.assertEqual(selector.call_count, 2)
        self.assertEqual(json.dumps(first, ensure_ascii=False), json.dumps(second, ensure_ascii=False))
        self.assertEqual(first["trace"], [{"stage": "bm25", "candidates": 2},
                                         {"stage": "evidence_selection", "method": "marginal_gain"}])
        self.assertEqual(set(first), {"ranked_doc_ids", "documents", "evidence", "coverage",
                                     "context_bytes", "coverage_is_confidence", "trace", "config"})
        self.assert_real_final_pack(first, "quartz constants", config)

    def test_expansion_invalidates_pack_even_when_search_mutates_same_documents(self):
        docs, store = self.reuse_documents(), Mock()
        config = self.reuse_config(bridge_terms=0)
        def search(*_, **kwargs):
            if kwargs["k"] == config.candidate_k:
                docs[0]["metadata"]["sentences"][0] = "Quartz constants and missing values."
                docs[0]["text"] = "\n".join(docs[0]["metadata"]["sentences"])
            return docs
        store.search.side_effect = search
        with patch.object(retrieval, "select_evidence", wraps=select_evidence) as selector:
            result = Retriever(store, config).retrieve("quartz constants missing")
        self.assertEqual(selector.call_count, 2)
        self.assertIn("expand", [stage["stage"] for stage in result["trace"]])
        self.assertEqual(result["coverage"], 1)
        self.assert_real_final_pack(result, "quartz constants missing", config)

    def test_bridge_recomputes_with_same_document_ids(self):
        docs, store = self.reuse_documents(), Mock()
        config = self.reuse_config()
        def search(question, **_):
            if "zephyr" in question:
                docs[1]["metadata"]["sentences"][0] = "Quartz constants and missing values are documented."
                docs[1]["text"] = docs[1]["metadata"]["sentences"][0]
            return docs
        store.search.side_effect = search
        with patch.object(retrieval, "select_evidence", wraps=select_evidence) as selector:
            result = Retriever(store, config).retrieve("quartz constants missing")
        self.assertEqual(selector.call_count, 2)
        self.assertEqual(store.search.call_count, 3)
        self.assertIn("bridge", [stage["stage"] for stage in result["trace"]])
        self.assert_real_final_pack(result, "quartz constants missing", config)

    def test_reranker_invalidates_pack_for_in_place_mutation_without_reordering(self):
        docs, store, reranker = self.reuse_documents(), Mock(), Mock()
        store.search.return_value = docs
        config = self.reuse_config(use_reranker=True)
        def rerank(_, candidates):
            candidates[0]["metadata"]["sentences"][1] = "Corrected values are 3.0 and 4.0."
            candidates[0]["text"] = "\n".join(candidates[0]["metadata"]["sentences"])
            return candidates
        reranker.rerank.side_effect = rerank
        with patch.object(retrieval, "select_evidence", wraps=select_evidence) as selector:
            result = Retriever(store, config, reranker=reranker).retrieve("quartz constants")
        self.assertEqual(selector.call_count, 2)
        self.assertEqual(result["ranked_doc_ids"], ["a", "b"])
        self.assertTrue(any("Corrected" in unit["text"] for unit in result["evidence"]))
        self.assert_real_final_pack(result, "quartz constants", config)

    def test_explicit_prefix_and_query_focused_selection_keep_their_final_method(self):
        docs, store = self.reuse_documents(), Mock()
        store.search.return_value = docs
        for method in ("prefix", "query_focused"):
            with self.subTest(method=method):
                config = self.reuse_config(evidence_selection=method)
                with patch.object(retrieval, "select_evidence", wraps=select_evidence) as initial:
                    result = Retriever(store, config).retrieve("quartz constants")
                self.assertEqual(initial.call_count, 1)
                self.assertEqual(result["trace"][-1]["method"], method)
                if method == "prefix":
                    expected = fixed_evidence(result["documents"], config.context_bytes)
                    for key, value in expected.items():
                        self.assertEqual(result[key], value)
                else:
                    self.assert_real_final_pack(result, "quartz constants", config,
                                                selector=query_focused_evidence)


if __name__ == "__main__":
    unittest.main()
