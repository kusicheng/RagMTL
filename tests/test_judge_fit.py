                                                                              

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from magra_benchmark.evaluation import evaluate_judge_calibration, fit_judge_dev
from magra_benchmark.protocol import json_hash


def audit_rows(metadata=True):
    humans, judgments = [], []
    for split in ("dev", "eval"):
        for index in range(6):
            qid, label = f"{split}-{index}", index % 2
            humans.append({"query_id": qid, "group_id": qid, "split": split, "label": label})
            row = {"query_id": qid, "groundedness_probability": 0.9 if label else 0.1}
            if metadata:
                row.update(model="fixture/model", judge_version="groundedness-v2",
                           model_sha256=json_hash("fixture/model"), rubric_sha256="b" * 64,
                           runtime_sha256="c" * 64, reasoning_effort=None)
            judgments.append(row)
    return humans, judgments


class JudgeDevFitTests(unittest.TestCase):
    def setUp(self):
        self.humans, self.judgments = audit_rows()
        self.dev_humans, self.dev_judgments = self.humans[:6], self.judgments[:6]

    def fit(self, **kwargs):
        return fit_judge_dev(self.dev_humans, self.dev_judgments, min_per_class=1, **kwargs)

    def evaluate(self, artifact, **kwargs):
        return evaluate_judge_calibration(self.humans, self.judgments, min_per_class=1,
                                          frozen_dev_fit=artifact, **kwargs)

    def test_reuse_matches_original_algorithm_and_has_no_eval_inputs(self):
        artifact = self.fit()
        original = evaluate_judge_calibration(self.humans, self.judgments, min_per_class=1)
        locked = self.evaluate(artifact)
        self.assertEqual(artifact["status"], "dev_frozen")
        self.assertFalse(artifact["use_for_acceptance"])
        body = {key: value for key, value in artifact.items() if key != "artifact_hash"}
        self.assertEqual(artifact["artifact_hash"], json_hash(body))
        self.assertEqual(artifact["fit_query_ids"], sorted(row["query_id"] for row in self.dev_humans))
        for field in ("threshold", "calibration_artifact", "dev_metrics", "locked_judge_hash"):
            self.assertEqual(artifact[field], original[field])
            self.assertEqual(locked[field], original[field])
        self.assertTrue(locked["fit_locked"])
        self.assertFalse(original["fit_locked"])
        self.assertEqual(locked["frozen_dev_fit_hash"], artifact["artifact_hash"])
        self.assertEqual(locked["status"], "audit_passed")
        self.assertEqual(artifact, fit_judge_dev(reversed(self.dev_humans), reversed(self.dev_judgments),
                                               min_per_class=1))

    def test_metadata_free_fixtures_remain_compatible(self):
        humans, judgments = audit_rows(metadata=False)
        artifact = fit_judge_dev(humans[:6], judgments[:6], min_per_class=1)
        self.assertEqual(artifact["judge_identity"], {})
        report = evaluate_judge_calibration(humans, judgments, min_per_class=1, frozen_dev_fit=artifact)
        self.assertTrue(report["fit_locked"])

    def test_fit_and_frozen_fit_can_be_loaded_from_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            human_file, prediction_file, frozen_file = (directory / name for name in
                                                        ("human.jsonl", "judgments.jsonl", "fit.json"))
            human_file.write_text("\n".join(json.dumps(row) for row in self.dev_humans), encoding="utf-8")
            prediction_file.write_text("\n".join(json.dumps(row) for row in self.dev_judgments), encoding="utf-8")
            artifact = fit_judge_dev(human_file, prediction_file, min_per_class=1)
            frozen_file.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertTrue(self.evaluate(frozen_file)["fit_locked"])

    def test_fit_refuses_every_non_dev_split_and_prediction_split_mismatch(self):
        for split in ("eval", "test", "locked_eval", "evaluation", "train", None):
            humans = deepcopy(self.dev_humans)
            humans[-1]["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "restricted to split='dev'"):
                fit_judge_dev(humans, self.dev_judgments, min_per_class=1)
        self.dev_judgments[-1]["split"] = "eval"
        with self.assertRaisesRegex(ValueError, "prediction split"):
            self.fit()

    def test_fit_rejects_duplicate_query_ids_and_paraphrase_groups(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_judge_dev(self.dev_humans + self.dev_humans[:1], self.dev_judgments, min_per_class=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_judge_dev(self.dev_humans, self.dev_judgments + self.dev_judgments[:1], min_per_class=1)
        self.dev_humans[-1]["group_id"] = self.dev_humans[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "group IDs"):
            self.fit()

    def test_fit_requires_class_counts_coverage_and_known_prediction_ids(self):
        self.dev_judgments[0].update(abstained=True, groundedness_probability=None)
        self.dev_judgments[1].update(abstained=True, groundedness_probability=None)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.fit()
        artifact = self.fit(min_coverage=0.5)
        self.assertEqual(artifact["counts"]["judged"], 4)
        with self.assertRaisesRegex(ValueError, "each human class"):
            fit_judge_dev(self.dev_humans, self.dev_judgments, min_per_class=3, min_coverage=0.5)
        with self.assertRaisesRegex(ValueError, "without human"):
            fit_judge_dev(self.dev_humans, self.judgments, min_per_class=1)

    def test_artifact_tampering_and_resigned_parameters_are_rejected(self):
        for field in ("threshold", "calibration_artifact", "provenance", "locked_judge_hash"):
            changed = deepcopy(self.fit())
            if field == "threshold":
                changed[field] = 0.123
            elif field == "calibration_artifact":
                changed[field]["blocks"][0][1] = 0.25
            elif field == "provenance":
                changed[field]["dataset_hash"] = "0" * 64
            else:
                changed[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "artifact hash"):
                self.evaluate(changed)
            changed["artifact_hash"] = json_hash({key: value for key, value in changed.items()
                                                  if key != "artifact_hash"})
            with self.subTest(resigned=field), self.assertRaisesRegex(ValueError, "differs from current"):
                self.evaluate(changed)

    def test_changed_dev_label_prediction_or_group_is_rejected(self):
        for target, field, value in (("human", "label", 1), ("human", "group_id", "new-group"),
                                     ("judgment", "groundedness_probability", 0.2)):
            artifact = self.fit()
            humans, judgments = deepcopy(self.humans), deepcopy(self.judgments)
            (humans if target == "human" else judgments)[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs from current"):
                evaluate_judge_calibration(humans, judgments, min_per_class=1, frozen_dev_fit=artifact)

    def test_unjudged_dev_row_is_also_bound(self):
        self.dev_judgments[0].update(abstained=True, groundedness_probability=None)
        artifact = self.fit()
                                                                                        
        self.humans[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "differs from current"):
            self.evaluate(artifact)

    def test_changed_model_rubric_or_runtime_is_rejected(self):
        for field, value in (("model", "other/model"), ("rubric_sha256", "d" * 64),
                             ("runtime_sha256", "e" * 64), ("reasoning_effort", "low")):
            artifact, judgments = self.fit(), deepcopy(self.judgments)
            for row in judgments:
                row[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs from current"):
                evaluate_judge_calibration(self.humans, judgments, min_per_class=1, frozen_dev_fit=artifact)

    def test_missing_or_mixed_identity_is_rejected_in_dev_and_eval(self):
        artifact = self.fit()
        for field in ("model", "rubric_sha256", "runtime_sha256"):
            judgments = deepcopy(self.judgments)
            judgments[-1].pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                evaluate_judge_calibration(self.humans, judgments, min_per_class=1, frozen_dev_fit=artifact)
        self.dev_judgments[-1]["runtime_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            self.fit()

    def test_eval_outcome_changes_never_change_fit(self):
        artifact = self.fit()
        original = self.evaluate(artifact)
        for row in self.judgments[6:]:
            row["groundedness_probability"] = 1 - row["groundedness_probability"]
        changed = self.evaluate(artifact)
        self.assertTrue(changed["fit_locked"])
        self.assertEqual(changed["locked_judge_hash"], original["locked_judge_hash"])
        self.assertEqual(changed["frozen_dev_fit_hash"], original["frozen_dev_fit_hash"])
        self.assertEqual(changed["evaluation_metrics"]["balanced_accuracy"], 0)
        self.assertEqual(changed["status"], "audit_failed")

    def test_fit_policy_and_cross_split_group_overlap_are_rejected(self):
        artifact = self.fit()
        with self.assertRaisesRegex(ValueError, "differs from current"):
            self.evaluate(artifact, min_coverage=0.7)
        self.humans[-1]["group_id"] = self.dev_humans[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "cross dev/evaluation"):
            self.evaluate(artifact)

    def test_incomplete_eval_keeps_verified_fit_but_refuses_acceptance(self):
        artifact = self.fit()
        report = evaluate_judge_calibration(self.humans, self.dev_judgments,
                                           min_per_class=1, frozen_dev_fit=artifact)
        self.assertTrue(report["fit_locked"])
        self.assertEqual(report["locked_judge_hash"], artifact["locked_judge_hash"])
        self.assertEqual(report["calibration_artifact"], artifact["calibration_artifact"])
        self.assertEqual(report["status"], "insufficient_audit")
        self.assertFalse(report["use_for_acceptance"])


if __name__ == "__main__":
    unittest.main()
