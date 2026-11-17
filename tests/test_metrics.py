import math
import unittest

from magra_benchmark.metrics import (
    abstention_metrics, answer_metrics, brier_score, evidence_metrics,
    expected_calibration_error, paired_bootstrap, retrieval_metrics,
    selective_metrics, token_scores,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_graded_trec_gain_and_map_denominator(self):
        result = retrieval_metrics(["medium", "unjudged", "best"],
                                   {"best": 3, "medium": 1, "other": 1, "zero": 0}, k=3)
        expected_ndcg = (1 + 3 / math.log2(4)) / (3 + 1 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(result["ndcg@3"], expected_ndcg)
        self.assertAlmostEqual(result["map@3"], (1 + 2 / 3) / 3)
        self.assertAlmostEqual(result["recall@3"], 2 / 3)
        self.assertEqual(result["mrr@3"], 1)
        self.assertAlmostEqual(result["unjudged_fraction@3"], 1 / 3)

    def test_cutoff_ap_uses_all_relevant_not_cutoff(self):
        score = retrieval_metrics(["a"], {"a": 1, "b": 1, "c": 1}, 1)
        self.assertAlmostEqual(score["map@1"], 1 / 3)
        self.assertEqual(score["ndcg@1"], 1)

    def test_zero_grade_is_a_judgment_and_absent_judgments_are_unknown(self):
        result = retrieval_metrics(["zero", "unseen"], {"zero": 0}, 2)
        self.assertEqual(result["ndcg@2"], 0)
        self.assertEqual(result["judged_fraction@2"], 0.5)
        self.assertIsNone(retrieval_metrics(["a"], {}, 2)["recall@2"])
        self.assertIsNone(retrieval_metrics(["a"], {"a": -1}, 2)["ndcg@2"])

    def test_duplicate_documents_cannot_inflate_scores(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(["a", "a"], {"a": 1})


class AnswerEvidenceTests(unittest.TestCase):
    def test_hotpot_normalization_and_binary_guard(self):
        score = answer_metrics("The Hague!", ["Amsterdam", "hague"])
        self.assertEqual(score["answer_em"], 1)
        self.assertEqual(score["answer_f1"], 1)
        self.assertEqual(token_scores("yes indeed", "yes"), (0, 0, 0))
        self.assertEqual(token_scores("", ""), (0, 0, 0))
        self.assertIsNone(answer_metrics("x", None)["answer_f1"])

    def test_sentence_citations_do_not_get_document_only_credit(self):
        support = [{"doc_id": "a", "sentence_id": 1}, {"doc_id": "b", "sentence_id": 2}]
        result = evidence_metrics([{"doc_id": "a", "sentence_id": 1}, "b", "b"], support)
        self.assertEqual(result["citation_precision"], 1)
        self.assertEqual(result["citation_recall"], 1)
        self.assertEqual(result["evidence_sentence_f1"], 0.5)
        self.assertIsNone(evidence_metrics(["a"], ["a"])["evidence_sentence_f1"])
        self.assertIsNone(evidence_metrics([], None)["citation_precision"])


class ReliabilityTests(unittest.TestCase):
    def test_calibration_bins_include_one(self):
        self.assertEqual(brier_score([0, 1], [0, 1]), 0)
        self.assertEqual(expected_calibration_error([0, 1], [0, 1]), 0)
        self.assertAlmostEqual(brier_score([0.8, 0.2], [1, 1]), 0.34)
        self.assertIsNone(brier_score([], []))
        with self.assertRaises(ValueError):
            brier_score([float("nan")], [1])

    def test_abstention_and_selective_accuracy_denominators(self):
        score = selective_metrics([1, 0, 1], [False, True, True])
        self.assertAlmostEqual(score["coverage"], 1 / 3)
        self.assertEqual(score["selective_accuracy"], 1)
        score = abstention_metrics([False, False, True], [True, False, True])
        self.assertEqual(score["abstention_precision"], 0.5)
        self.assertEqual(score["unsupported_answer_rate"], 0.5)
        self.assertEqual(score["false_abstention_rate"], 1)

    def test_paired_cluster_bootstrap_and_missing_interval(self):
        result = paired_bootstrap([2, 2, 2], [1, 1, 1], samples=100, seed=4)
        self.assertEqual((result["delta"], result["ci_low"], result["ci_high"]), (1, 1, 1))
        self.assertEqual(result, paired_bootstrap([2, 2, 2], [1, 1, 1], samples=100, seed=4))
        result = paired_bootstrap([2, 3], [1, 1], cluster_ids=["duplicate", "duplicate"])
        self.assertEqual(result["status"], "insufficient_pairs")
        self.assertIsNone(result["ci_low"])


if __name__ == "__main__":
    unittest.main()
