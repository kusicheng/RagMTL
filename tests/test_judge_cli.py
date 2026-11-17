                                                                             

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from magra_rag.cli import main


class JudgeCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.humans, self.judgments = {}, {}
        for split in ("dev", "eval"):
            humans, judgments = [], []
            for i, label in enumerate((0, 0, 1, 1)):
                qid = f"{split}:{i}"
                humans.append({"query_id": qid, "group_id": qid, "split": split, "label": label})
                judgments.append({"query_id": qid, "groundedness_probability": 0.9 if label else 0.1})
            self.humans[split] = self.write(f"{split}.human.jsonl", humans)
            self.judgments[split] = self.write(f"{split}.judge.jsonl", judgments)
        self.combined = self.root / "humans.jsonl"
        self.combined.write_bytes(b"".join(p.read_bytes() for p in self.humans.values()))

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(["--data-root", str(self.root), *map(str, args)])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_fit_dev_then_audit_separate_completed_inputs(self):
        fit = self.root / "fit.json"
        status, _, error = self.invoke("judge-fit-dev", "--human-labels", self.humans["dev"],
                                      "--judgments", self.judgments["dev"], "--min-per-class", 1, "--out", fit)
        self.assertEqual(status, 0, error)
        frozen_bytes = fit.read_bytes()
        report = self.root / "report.json"
        status, _, error = self.invoke("judge-audit", "--human-labels", self.combined, "--judgments",
                                      *self.judgments.values(), "--frozen-dev-fit", fit,
                                      "--min-per-class", 1, "--out", report)
        self.assertEqual(status, 0, error)
        result = json.loads(report.read_text(encoding="utf-8"))
        self.assertTrue(result["fit_locked"])
        self.assertTrue(result["use_for_acceptance"])
        self.assertEqual(result["evaluation_metrics"]["balanced_accuracy"], 1.0)
        self.assertEqual(fit.read_bytes(), frozen_bytes)

    def test_development_fit_cannot_read_evaluation_labels(self):
        target = self.root / "rejected.json"
        status, _, error = self.invoke("judge-fit-dev", "--human-labels", self.combined,
                                      "--judgments", *self.judgments.values(), "--min-per-class", 1, "--out", target)
        self.assertEqual(status, 1)
        self.assertTrue(error)
        self.assertFalse(target.exists())

    def test_failed_or_changed_judge_runs_cannot_be_audited(self):
        path = self.judgments["dev"]
        manifest_path = Path(str(path) + ".manifest.json")
        manifest = {"status": "failed", "judgment_count": 4,
                    "query_ids": [f"dev:{i}" for i in range(4)],
                    "judgments_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for state in ("failed", "changed"):
            if state == "changed":
                manifest["status"] = "complete"
                manifest["judgments_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            status, _, error = self.invoke("judge-fit-dev", "--human-labels", self.humans["dev"],
                                          "--judgments", path, "--min-per-class", 1,
                                          "--out", self.root / f"{state}.json")
            self.assertEqual(status, 1)
            self.assertIn("incomplete or judgment file changed", error)


if __name__ == "__main__":
    unittest.main()
