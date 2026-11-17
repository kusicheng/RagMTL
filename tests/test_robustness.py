import copy
import unittest

from magra_benchmark.robustness import CASES, build_failure_case, build_failure_suite, materialize_overlay


class RobustnessTests(unittest.TestCase):
    def test_suite_is_deterministic_explicitly_synthetic_and_complete(self):
        first = build_failure_suite(seed=7, as_of="2025-01-02")
        self.assertEqual(first, build_failure_suite(seed=7, as_of="2025-01-02"))
        self.assertNotEqual(first, build_failure_suite(seed=8, as_of="2025-01-02"))
        self.assertEqual({case["case"] for case in first}, set(CASES))
        for case in first:
            self.assertTrue(case["synthetic"])
            self.assertEqual(case["track"], "synthetic_robustness")
            self.assertEqual(case["query"]["track"], "synthetic_robustness")
            for document in case["documents"]:
                self.assertTrue(document["metadata"]["synthetic"])

    def test_missing_evidence_overlay_removes_all_judged_and_exact_copies(self):
        documents = [
            {"id": "support", "text": "A fact.  With whitespace."},
            {"id": "copy", "text": "a fact. with whitespace."},
            {"id": "background", "text": "Unrelated background."},
        ]
        original = copy.deepcopy(documents)
        case = build_failure_case("missing_evidence", documents, "Natural query", ["support"])
        self.assertEqual(documents, original)
        indexed_ids = {document["id"] for document in case["documents"]}
        self.assertNotIn("support", indexed_ids)
        self.assertNotIn("copy", indexed_ids)
        self.assertIn("background", indexed_ids)
        self.assertFalse(case["labels"]["answerable"])
        self.assertEqual(case["labels"]["expected_behavior"], "abstain")
        self.assertTrue(case["labels"]["evidence_absent_by_construction"])
        withheld = case["labels"]["withheld_evidence"][0]
        self.assertNotIn(withheld["id"], indexed_ids)
        self.assertTrue(all(withheld["text"] != document["text"] for document in case["documents"]))
        self.assertEqual(case["documents"], materialize_overlay(documents, case["overlay"]))
        self.assertEqual(case["provenance"]["source_query"], "Natural query")

    def test_conflicting_sources_require_conflict_reporting(self):
        case = build_failure_case("conflicting_sources")
        labels = case["labels"]
        self.assertEqual(labels["expected_behavior"], "report_conflict")
        self.assertIsNone(labels["expected_answer"])
        self.assertEqual(len(labels["conflicting_ids"]), 2)
        self.assertEqual(len(set(labels["conflicting_values"])), 2)
        for identifier, value in zip(labels["conflicting_ids"], labels["conflicting_values"]):
            document = next(document for document in case["documents"] if document["id"] == identifier)
            self.assertIn(value, document["text"])
            self.assertEqual(document["metadata"]["document_date"], "2025-01-01")

    def test_stale_fact_has_explicit_as_of_and_dated_superseding_evidence(self):
        case = build_failure_case("stale_facts", as_of="2025-07-10")
        labels = case["labels"]
        by_id = {document["id"]: document for document in case["documents"]}
        current = by_id[labels["supporting_ids"][0]]
        old = by_id[labels["stale_ids"][0]]
        self.assertLess(old["metadata"]["document_date"], current["metadata"]["document_date"])
        self.assertLess(current["metadata"]["document_date"], case["query"]["as_of"])
        self.assertIn("superseding", current["text"])
        self.assertIn(labels["expected_answer"], current["text"])
        self.assertIn("2025-07-10", case["query"]["question"])

    def test_lexical_mismatch_and_irrelevant_context_have_known_support(self):
        mismatch = build_failure_case("lexical_mismatch")
        self.assertIn("car", mismatch["query"]["text"])
        self.assertIn("automobile", mismatch["documents"][0]["text"])
        self.assertIn(mismatch["labels"]["expected_answer"], mismatch["documents"][0]["text"])
        noisy = build_failure_case("irrelevant_context")
        self.assertEqual(len(noisy["labels"]["irrelevant_ids"]), 5)
        for document in noisy["documents"]:
            if document["id"] in noisy["labels"]["irrelevant_ids"]:
                self.assertNotIn(noisy["labels"]["expected_answer"], document["text"])

    def test_labels_never_enter_documents_and_background_metadata_is_preserved(self):
        background = [{"id": "natural", "title": "Owned", "text": "Owned evidence", "metadata": {"source": "owned", "synthetic": False}}]
        for case in build_failure_suite(background):
            self.assertEqual(case["documents"][0], background[0])
            for document in case["documents"]:
                self.assertEqual(set(document), {"id", "title", "text", "metadata"})
                self.assertTrue({"labels", "answers", "qrels", "expected_answer"}.isdisjoint(document["metadata"]))

    def test_overlay_is_independent_and_rejects_ambiguous_ids_and_annotations(self):
        base = [{"id": "a", "text": "A", "metadata": {"tags": ["x"]}}]
        output = materialize_overlay(base, {})
        output[0]["metadata"]["tags"].append("y")
        self.assertEqual(base[0]["metadata"]["tags"], ["x"])
        with self.assertRaises(ValueError):
            materialize_overlay(base + base, {})
        with self.assertRaises(ValueError):
            materialize_overlay(base, {"add_documents": [{"id": "a", "text": "B"}]})
        with self.assertRaises(ValueError):
            build_failure_suite([{"id": "a", "text": "A", "metadata": {"answers": ["secret"]}}])
        with self.assertRaises(ValueError):
            build_failure_case("missing_evidence", base, relevant_ids=["unknown"])
        with self.assertRaises(ValueError):
            build_failure_case("unknown")


if __name__ == "__main__":
    unittest.main()
