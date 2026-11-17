import json
from pathlib import Path
import tempfile
import unittest

from magra_benchmark.calibration import fit_calibrator, load_calibrator


PROVENANCE = {"dataset_hash": "a" * 64, "split_hash": "b" * 64, "predictions_hash": "c" * 64}


def rows():
    return [{"query_id": "dev-" + str(i), "split": "dev", "confidence": score, "correct": correct}
            for i, (score, correct) in enumerate([(0.1, 0), (0.2, 1), (0.3, 0), (0.9, 1)])]


class CalibrationTests(unittest.TestCase):
    def test_pava_is_monotone_and_pools_adjacent_violators(self):
        calibrated = fit_calibrator(rows(), provenance=PROVENANCE, min_accepted=1)
        self.assertEqual(calibrated.predict(0.1), 0)
        self.assertEqual(calibrated.predict(0.2), 0.5)
        self.assertEqual(calibrated.predict(0.3), 0.5)
        self.assertEqual(calibrated.predict(0.9), 1)
        self.assertTrue(calibrated.should_abstain(0.2))
        self.assertFalse(calibrated.should_abstain(0.9))
        self.assertEqual(calibrated.to_dict()["fit_diagnostics"]["scope"], "in_sample_dev_only")

    def test_tied_scores_do_not_depend_on_input_order(self):
        tied = [{"query_id": "x", "split": "dev", "confidence": 0.5, "correct": 1},
                {"query_id": "y", "split": "dev", "confidence": 0.5, "correct": 0}]
        self.assertEqual(fit_calibrator(tied, provenance=PROVENANCE).artifact_hash,
                         fit_calibrator(reversed(tied), provenance=PROVENANCE).artifact_hash)

    def test_eval_data_and_missing_provenance_are_rejected(self):
        invalid = rows()
        invalid[0]["split"] = "test"
        with self.assertRaises(ValueError):
            fit_calibrator(invalid, provenance=PROVENANCE)
        with self.assertRaises(ValueError):
            fit_calibrator(rows(), provenance={})
        duplicated = rows() + [rows()[0]]
        with self.assertRaises(ValueError):
            fit_calibrator(duplicated, provenance=PROVENANCE)

    def test_artifact_is_create_once_pinned_and_integrity_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibrator.json"
            fitted = fit_calibrator(rows(), provenance=PROVENANCE, output_path=path)
            loaded = load_calibrator(path, expected_hash=fitted.artifact_hash)
            self.assertEqual(loaded.predict(0.2), fitted.predict(0.2))
            with self.assertRaises(FileExistsError):
                fitted.save(path)
            with self.assertRaises(ValueError):
                load_calibrator(path, expected_hash="0" * 64)
            document = json.loads(path.read_text())
            document["threshold"] = 0
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_calibrator(path)

    def test_insufficient_accepted_sample_count_means_abstain(self):
        fitted = fit_calibrator(rows(), provenance=PROVENANCE, min_accepted=100)
        self.assertTrue(fitted.should_abstain(1))


if __name__ == "__main__":
    unittest.main()
