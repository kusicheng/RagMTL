import unittest

from magra_benchmark.calibration import fit_calibrator
from magra_benchmark.evaluation import (
    compare_runs, evaluate_judge_calibration, evaluate_run, select_baselines,
)


def label(qid, split="eval", **kwargs):
    return {"id": qid, "split": split, "domain": "science", "source": "fixture",
            "answer": "Ada Lovelace", "relevance": {"doc": 1},
            "supporting_facts": [{"doc_id": "doc", "sentence_id": 0}], **kwargs}


def prediction(qid, correct=True, **kwargs):
    return {"query_id": qid, "ranked_doc_ids": ["doc"], "answer": "Ada Lovelace" if correct else "wrong",
            "citations": [{"doc_id": "doc", "sentence_id": 0}], "confidence": 0.8,
            "abstained": False, "cost_usd": 0.01, "tokens": 100, "latency_ms": 50, **kwargs}


class EvaluationTests(unittest.TestCase):
    def test_never_pool_context_synthetic_or_web_settings(self):
        queries = [label("a", setting="provided_context"), label("b", setting="full_corpus"),
                   label("c", synthetic=True), label("d", web_mode="live")]
        report = evaluate_run(queries, [prediction(q["id"]) for q in queries])
        self.assertEqual(len(report["groups"]), 4)
        self.assertNotIn("overall", report)
        self.assertFalse(report["generality"]["claim_supported"])
        for group in report["groups"].values():
            self.assertEqual(group["metrics"]["joint_f1"]["value"], 1)

    def test_missing_predictions_penalize_quality_and_coverage(self):
        report = evaluate_run([label("a"), label("b")], [prediction("a")])
        metrics = next(iter(report["groups"].values()))["metrics"]
        self.assertEqual(metrics["answer_em"]["value"], 0.5)
        self.assertEqual(metrics["coverage"]["value"], 0.5)
        self.assertEqual(metrics["cost_usd"]["missing"], 1)
        self.assertEqual(report["missing_predictions"], ["b"])

    def test_retrieval_only_does_not_invent_answer_or_cost_metrics(self):
        report = evaluate_run([{"id": "a", "relevance": {"doc": 1}}],
                              [{"query_id": "a", "ranked_doc_ids": ["doc"]}])
        metrics = next(iter(report["groups"].values()))["metrics"]
        self.assertEqual(metrics["answer_f1"]["status"], "unavailable")
        self.assertIsNone(metrics["cost_usd"]["value"])
        self.assertEqual(metrics["ndcg@10"]["value"], 1)

    def test_explicit_retrieval_only_run_leaves_all_generation_metrics_unavailable(self):
        queries = [label("a", answerable=True), label("missing", answerable=False)]
        output = {"query_id": "a", "ranked_doc_ids": ["doc"], "latency_ms": 5,
                  "evidence": [{"doc_id": "doc", "sentence_id": 0, "text": "Ada Lovelace"}]}
        report = evaluate_run(queries, [output], run_metadata={"generation": False})
        metrics = next(iter(report["groups"].values()))["metrics"]
        self.assertEqual(metrics["ndcg@10"]["value"], 0.5)
        self.assertEqual(metrics["latency_ms"]["value"], 5)
        self.assertEqual(report["missing_predictions"], ["missing"])
        generation_metrics = (
            "answer_em", "answer_f1", "answer_precision", "answer_recall",
            "citation_precision", "citation_recall", "citation_f1", "citation_em",
            "evidence_sentence_precision", "evidence_sentence_recall", "evidence_sentence_f1",
            "evidence_sentence_em", "joint_em", "joint_f1", "coverage", "selective_accuracy",
            "selective_risk", "raw_brier", "raw_ece", "calibrated_brier", "calibrated_ece",
            "abstention_precision", "missing_evidence_abstention_recall", "unsupported_answer_rate",
            "false_abstention_rate", "answerability_decision_accuracy",
        )
        for name in generation_metrics:
            with self.subTest(metric=name):
                self.assertIsNone(metrics[name]["value"])
                self.assertEqual(metrics[name]["status"], "unavailable")
                self.assertEqual(metrics[name]["count"], 0)
        self.assertFalse(report["generation_scoring_enabled"])
        for row in report["per_query"]:
            self.assertIsNone(row["abstained"])
            self.assertIsNone(row["answerable"])
            self.assertIsNone(row["correct"])
            self.assertIsNone(row["confidence"])
            self.assertIsNone(row["calibrated_confidence"])

    def test_generated_bad_answers_and_real_abstentions_still_receive_generation_penalties(self):
        queries = [label("a", answerable=True)]
        bad = prediction("a", correct=False, citations=[])
        for metadata in (None, {}, {"generation": True}):
            with self.subTest(metadata=metadata):
                report = evaluate_run(queries, [bad], run_metadata=metadata)
                metrics = next(iter(report["groups"].values()))["metrics"]
                for name in ("answer_em", "answer_f1", "citation_f1", "evidence_sentence_f1", "joint_f1"):
                    self.assertEqual(metrics[name]["value"], 0)
                    self.assertEqual(metrics[name]["status"], "available")
                self.assertEqual(metrics["coverage"]["value"], 1)
                self.assertEqual(metrics["selective_accuracy"]["value"], 0)
                self.assertAlmostEqual(metrics["raw_brier"]["value"], 0.64)
        abstention = prediction("a", answer="", citations=[], confidence=0.0, abstained=True)
        report = evaluate_run(queries, [abstention], run_metadata={"generation": True})
        metrics = next(iter(report["groups"].values()))["metrics"]
        self.assertEqual(metrics["answer_em"]["value"], 0)
        self.assertEqual(metrics["citation_f1"]["value"], 0)
        self.assertEqual(metrics["coverage"]["value"], 0)
        self.assertEqual(metrics["false_abstention_rate"]["value"], 1)

    def test_selected_evidence_is_scored_without_generation_and_missing_telemetry_is_unknown(self):
        queries = [label("selected"), label("unknown"), label("missing")]
        outputs = [{"query_id": "selected", "ranked_doc_ids": ["doc"], "context_bytes": 100,
                    "evidence": [{"doc_id": "doc", "sentence_id": 0},
                                 {"doc_id": "doc", "sentence_id": 1}]},
                   {"query_id": "unknown", "ranked_doc_ids": ["doc"]}]
        report = evaluate_run(queries, outputs, run_metadata={"generation": False})
        rows = {row["query_id"]: row["metrics"] for row in report["per_query"]}
        self.assertEqual(rows["selected"]["selected_document_recall"], 1)
        self.assertEqual(rows["selected"]["selected_sentence_recall"], 1)
        self.assertEqual(rows["selected"]["selected_sentence_precision"], 0.5)
        self.assertEqual(rows["missing"]["selected_sentence_recall"], 0)
        self.assertIsNone(rows["unknown"]["selected_sentence_recall"])
        self.assertIsNone(rows["selected"]["citation_recall"])
        self.assertEqual(rows["selected"]["context_bytes"], 100)

    def test_explicit_retrieval_mode_does_not_score_incidental_answer_fields(self):
        report = evaluate_run([label("a", answerable=True)], [prediction("a")],
                              run_metadata={"generation": False})
        metrics = next(iter(report["groups"].values()))["metrics"]
        self.assertIsNone(metrics["answer_em"]["value"])
        self.assertIsNone(metrics["citation_f1"]["value"])
        self.assertIsNone(metrics["raw_brier"]["value"])
        self.assertEqual(metrics["ndcg@10"]["value"], 1)

    def test_unknown_duplicate_and_calibration_overlap_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_run([label("a")], [prediction("outside")])
        with self.assertRaises(ValueError):
            evaluate_run([label("a")], [prediction("a"), prediction("a")])
        fitted = fit_calibrator([{"query_id": "a", "split": "dev", "confidence": 0.5, "correct": 1}],
                                provenance={"dataset_hash": "a" * 64, "split_hash": "b" * 64,
                                            "predictions_hash": "c" * 64})
        with self.assertRaises(ValueError):
            evaluate_run([label("a")], [prediction("a")], calibration=fitted)

    def test_missing_evidence_abstention_is_scored_separately(self):
        report = evaluate_run([label("a", answerable=False), label("b", answerable=False)],
                              [prediction("a", abstained=True), prediction("b")])
        metrics = next(iter(report["groups"].values()))["metrics"]
        self.assertEqual(metrics["missing_evidence_abstention_recall"]["value"], 0.5)
        self.assertEqual(metrics["selective_accuracy"]["value"], 0)
        self.assertEqual(metrics["selective_accuracy"]["count"], 1)
        self.assertEqual(metrics["abstention_precision"]["count"], 1)

    def test_coverage_does_not_require_answer_reference_labels(self):
        report = evaluate_run([{"id": "a", "answerable": True}], [prediction("a")])
        metrics = next(iter(report["groups"].values()))["metrics"]
        self.assertEqual(metrics["coverage"]["value"], 1)
        self.assertIsNone(metrics["selective_accuracy"]["value"])
        self.assertEqual(metrics["selective_accuracy"]["count"], 0)

    def test_baseline_is_selected_on_dev_not_eval(self):
        dev = [label("dev-a", "dev"), label("dev-b", "dev")]
        reports = {
            "strong-dev": evaluate_run(dev, [prediction(q["id"]) for q in dev]),
            "weak-dev": evaluate_run(dev, [prediction(q["id"], False) for q in dev]),
        }
        evaluated = [label("eval-a"), label("eval-b")]
        runs = {"candidate": [prediction(q["id"]) for q in evaluated],
                "strong-dev": [prediction(q["id"], False) for q in evaluated],
                "weak-dev": [prediction(q["id"]) for q in evaluated]}
        comparison = compare_runs(evaluated, runs, baseline_names=["strong-dev", "weak-dev"],
                                  candidate_name="candidate", development_reports=reports,
                                  bootstrap_samples=100)
        group = next(iter(comparison["groups"].values()))
        self.assertEqual(group["baseline"], "strong-dev")
        self.assertEqual(group["paired_deltas"]["answer_f1"]["ci_low"], 1)
        self.assertFalse(comparison["generality_claim_supported"])
        self.assertTrue(group["observations"]["cost_difference_ci_at_or_below_zero"])

    def test_dev_eval_group_leak_and_missing_feasibility_rejected(self):
        dev = [label("dev-a", "dev", group_id="same-question")]
        report = evaluate_run(dev, [prediction("dev-a")])
        selected = select_baselines({"base": report}, ["base"], constraints={"peak_host_ram_bytes": 32_000_000_000})
        self.assertIsNone(next(iter(selected.values()))["baseline"])
        with self.assertRaises(ValueError):
            compare_runs([label("eval-a", group_id="same-question")],
                         {"base": [prediction("eval-a")], "new": [prediction("eval-a")]},
                         baseline_names=["base"], candidate_name="new", development_reports={"base": report})

    def test_unseen_domain_uses_frozen_dev_macro_baseline(self):
        dev = [label("dev-a", "dev", domain="science"),
               label("dev-b", "dev", domain="finance", source="other")]
        reports = {
            "a": evaluate_run(dev, [prediction("dev-a"), prediction("dev-b")]),
            "b": evaluate_run(dev, [prediction("dev-a", False), prediction("dev-b", False)]),
        }
        labels = [label("eval-a", domain="arguments", source="heldout"),
                  label("eval-b", domain="arguments", source="heldout")]
        comparison = compare_runs(labels,
                                  {name: [prediction(q["id"]) for q in labels] for name in ("a", "b", "new")},
                                  baseline_names=["a", "b"], candidate_name="new", development_reports=reports,
                                  bootstrap_samples=20)
        group = next(iter(comparison["groups"].values()))
        self.assertEqual(group["baseline"], "a")
        self.assertEqual(group["selection"]["policy"], "dev_macro_for_unseen_source")
        self.assertEqual(group["selection"]["development_macro_quality"]["a"], 1)
                                                                                
        for query in labels:
            query["setting"] = "provided_context"
        separate = compare_runs(labels,
                                {name: [prediction(q["id"]) for q in labels] for name in ("a", "b", "new")},
                                baseline_names=["a", "b"], candidate_name="new", development_reports=reports)
        self.assertEqual(next(iter(separate["groups"].values()))["status"], "no_dev_selected_baseline")

    def test_engine_calibrated_confidence_is_not_calibrated_twice(self):
        fitted = fit_calibrator(
            [{"query_id": "dev", "split": "dev", "confidence": 0.5, "correct": 1}],
            provenance={"dataset_hash": "a" * 64, "split_hash": "b" * 64, "predictions_hash": "c" * 64})
        report = evaluate_run([label("eval")],
                              [prediction("eval", raw_confidence=0.5, confidence=1.0,
                                          calibration_hash=fitted.artifact_hash)], calibration=fitted)
        self.assertEqual(report["per_query"][0]["confidence"], 0.5)
        self.assertEqual(report["per_query"][0]["calibrated_confidence"], 1)


class JudgeAuditTests(unittest.TestCase):
    def audit_data(self):
        humans, judges = [], []
        for split in ("dev", "eval"):
            for i in range(10):
                qid, truth = f"{split}-{i}", i % 2
                humans.append({"query_id": qid, "split": split, "label": truth})
                judges.append({"query_id": qid, "groundedness_probability": 0.9 if truth else 0.1})
        return humans, judges

    def test_human_judge_audit_measures_discrimination_and_agreement(self):
        humans, judges = self.audit_data()
        report = evaluate_judge_calibration(humans, judges)
        self.assertEqual(report["status"], "audit_passed")
        self.assertEqual(report["evaluation_metrics"]["balanced_accuracy"], 1)
        self.assertEqual(report["evaluation_metrics"]["auroc"], 1)
        self.assertEqual(report["evaluation_metrics"]["cohen_kappa"], 1)
        self.assertEqual(report["calibration_artifact"]["fit_count"], 10)

    def test_bad_heldout_judgments_do_not_change_dev_threshold(self):
        humans, judges = self.audit_data()
        original = evaluate_judge_calibration(humans, judges)
        for row in judges:
            if row["query_id"].startswith("eval"):
                row["groundedness_probability"] = 1 - row["groundedness_probability"]
        degraded = evaluate_judge_calibration(humans, judges)
        self.assertEqual(original["threshold"], degraded["threshold"])
        self.assertEqual(original["locked_judge_hash"], degraded["locked_judge_hash"])
        self.assertTrue(degraded["recommend_abstention"])
        self.assertEqual(degraded["evaluation_metrics"]["balanced_accuracy"], 0)

    def test_small_audit_and_duplicate_leak_fail_closed(self):
        humans, judges = self.audit_data()
        report = evaluate_judge_calibration(humans[:3], judges[:3])
        self.assertEqual(report["status"], "insufficient_audit")
        self.assertTrue(report["recommend_abstention"])
        humans[0]["group_id"] = "duplicated"
        humans[10]["group_id"] = "duplicated"
        with self.assertRaises(ValueError):
            evaluate_judge_calibration(humans, judges)


if __name__ == "__main__":
    unittest.main()
