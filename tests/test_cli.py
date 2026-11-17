                                                                                      

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from magra_benchmark.splits import read_split_bundle


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = Path(__file__).resolve().parent.parent
        self.data = self.root / "data"
        self.data.mkdir()
        self.fixture = self.data / "sources/fixture"
        (self.fixture / "qrels").mkdir(parents=True)
        documents = [
            {"_id": "d1", "title": "Saturn", "text": "Saturn is a planet with spectacular rings of ice."},
            {"_id": "d2", "title": "Halite", "text": "The mineral halite forms ordinary table salt."},
            {"_id": "d3", "title": "Wetlands", "text": "Wetland reeds clean runoff by absorbing contaminants."},
            {"_id": "d4", "title": "Aquifers", "text": "Mineralized water moves through porous rocks in aquifers."},
            {"_id": "d5", "title": "Unrelated", "text": "Unjudged vases display wild flowers."},
        ]
        self.questions = [
            {"_id": "train", "text": "How do wetland reeds clean runoff?"},
            {"_id": "dev", "text": "Which planet carries spectacular rings?"},
            {"_id": "eval", "text": "What mineral forms ordinary table salt?"},
            {"_id": "duplicate-train", "text": "Which processes explain mineralized water movement through porous rocks?"},
            {"_id": "duplicate-eval", "text": "WHICH processes explain mineralized water movement through porous rocks"},
        ]
        self.write_jsonl(self.fixture / "corpus.jsonl", documents)
        self.write_jsonl(self.fixture / "queries.jsonl", self.questions)
        for split, pairs in {"train": [("train", "d3"), ("duplicate-train", "d4")],
                             "dev": [("dev", "d1")],
                             "test": [("eval", "d2"), ("duplicate-eval", "d4")]}.items():
            (self.fixture / "qrels" / f"{split}.tsv").write_text(
                "query-id\tcorpus-id\tscore\n" + "".join(f"{query}\t{doc}\t1\n" for query, doc in pairs), encoding="utf-8")
        self.bundle = self.data / "bundle"

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def cli(self, *arguments, success=True):
        process = subprocess.run(
            [sys.executable, "-m", "magra_rag", "--data-root", str(self.data), *map(str, arguments)],
            cwd=self.repository, capture_output=True, text=True, encoding="utf-8", timeout=40)
        if success:
            self.assertEqual(process.returncode, 0, process.stderr)
            return json.loads(process.stdout)
        self.assertNotEqual(process.returncode, 0, "Command unexpectedly succeeded")
        return process

    def ingest_and_prepare(self):
        self.cli("ingest", "--format", "beir", self.fixture)
        return self.cli("prepare", "--beir", self.fixture, "--out", self.bundle)

    def test_complete_offline_workflow_dev_selection_and_locked_eval(self):
        prepared = self.ingest_and_prepare()
        self.assertEqual(prepared["counts"], {"train": 1, "dev": 1, "eval": 3})
        configurations = {}
        reports = {}
        for strategy in ("bm25", "magra"):
            configuration = self.data / f"{strategy}.config.json"
            configuration.write_text(json.dumps({"strategy": strategy}), encoding="utf-8")
            configurations[strategy] = configuration
            predictions = self.data / f"{strategy}.dev.jsonl"
            run = self.cli("run", "--bundle", self.bundle, "--split", "dev", "--config", configuration,
                           "--out", predictions)
            self.assertEqual(run["query_count"], 1)
            report = self.data / f"{strategy}.dev.report.json"
            self.cli("evaluate", "--bundle", self.bundle, "--predictions", predictions, "--out", report)
            reports[strategy] = report
            evaluated = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(evaluated["splits"], ["dev"])
            self.assertEqual(evaluated["missing_predictions"], [])
            group = next(iter(evaluated["groups"].values()))
            self.assertEqual(group["metrics"]["ndcg@10"]["value"], 1)
            self.assertIsNone(group["metrics"]["answer_em"]["value"])
        frozen_path = self.data / "preregistered.json"
        frozen = self.cli("freeze", "--bundle", self.bundle, "--configs",
                          *[f"{name}={path}" for name, path in configurations.items()],
                          "--dev-reports", *[f"{name}={path}" for name, path in reports.items()],
                          "--baselines", "bm25", "--candidate", "magra", "--out", frozen_path)
        self.assertEqual(len(frozen["eval_query_ids"]), 3)
        self.assertTrue(any(row.get("baseline") == "bm25" for row in frozen["baseline_selection"].values()))
        output = self.data / "magra.eval.jsonl"
        self.cli("run", "--bundle", self.bundle, "--split", "eval", "--config", configurations["magra"],
                 "--freeze", frozen_path, "--out", output)
        evaluation_path = self.data / "magra.eval.report.json"
        self.cli("evaluate", "--bundle", self.bundle, "--predictions", output, "--out", evaluation_path)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        self.assertEqual(evaluation["query_count"], 3)
        self.assertEqual(evaluation["splits"], ["eval"])
        self.assertFalse(evaluation["generality"]["claim_supported"])
        manifest = json.loads(Path(str(output) + ".manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["generation"])
        self.assertEqual(manifest["prediction_count"], 3)

    def test_eval_requires_preregistration_and_existing_outputs_are_preserved(self):
        self.ingest_and_prepare()
        rejected = self.cli("run", "--bundle", self.bundle, "--split", "eval",
                            "--out", self.data / "eval.jsonl", success=False)
        self.assertIn("Locked eval requires", rejected.stderr)
        self.assertFalse((self.data / "eval.jsonl").exists())
        output = self.data / "dev.jsonl"
        self.cli("run", "--bundle", self.bundle, "--out", output)
        original = output.read_bytes()
        self.cli("run", "--bundle", self.bundle, "--out", output, success=False)
        self.assertEqual(output.read_bytes(), original)

    def test_duplicate_queries_cannot_cross_splits_and_full_corpus_remains_searchable(self):
        self.ingest_and_prepare()
        train = list(read_split_bundle(self.bundle, "train"))
        evaluation = list(read_split_bundle(self.bundle, "eval"))
        self.assertNotIn("fixture:duplicate-train", {query["id"] for query in train})
        duplicates = [query for query in evaluation if "duplicate" in query["id"]]
        self.assertEqual(len(duplicates), 2)
        self.assertEqual(duplicates[0]["group_id"], duplicates[1]["group_id"])
        with closing(sqlite3.connect(self.data / "corpus.sqlite")) as connection:
            rows = connection.execute("SELECT id,text,metadata FROM documents").fetchall()
        self.assertEqual(len(rows), 5)
        self.assertIn("fixture:d5", {row[0] for row in rows})
        serialized = json.dumps(rows)
        self.assertFalse(any(question["text"] in serialized for question in self.questions))
        self.assertNotIn('"relevance"', serialized)

    def test_hotpot_private_annotations_never_enter_corpus_or_public_queries(self):
        source = self.data / "hotpot_dev.json"
        secret = "PRIVATE_GOLD_28c9aee4"
        records = [{"_id": "hotpot-dev-1", "question": "Where was the observatory telescope assembled?",
                    "answer": secret, "supporting_facts": [["Observatory", 0]],
                    "context": [["Observatory", ["The telescope was assembled in Chile."]],
                                ["Other", ["The aircraft landed in Peru."]]]}]
        source.write_text(json.dumps(records), encoding="utf-8")
        self.cli("ingest", "--format", "hotpot_context", source)
        self.cli("prepare", "--hotpot-dev", source, "--hotpot-settings", "provided_context", "--out", self.bundle)
        public = list(read_split_bundle(self.bundle, "dev"))
        private = list(read_split_bundle(self.bundle, "dev", include_labels=True))
        self.assertNotIn(secret, json.dumps(public))
        self.assertEqual(private[0]["answer"], secret)
        with closing(sqlite3.connect(self.data / "corpus.sqlite")) as connection:
            dump = "\n".join(connection.iterdump())
        self.assertNotIn(secret, dump)
        self.assertNotIn("supporting_facts", dump)
        self.assertNotIn(records[0]["question"], dump)
        output = self.data / "hotpot.dev.predictions.jsonl"
        self.cli("run", "--bundle", self.bundle, "--out", output)
        predictions = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(predictions[0]["ranked_doc_ids"])
        self.assertLessEqual(set(predictions[0]["ranked_doc_ids"]), set(public[0]["context_doc_ids"]))

    def test_broad_judge_audit_requires_prior_group_exclusion(self):
        output = self.data / "audit"
        common = ("prepare-judge-audit", "--source", self.fixture, "--out", output)
        rejected = self.cli(*common, "--all-tasks", success=False)
        self.assertIn("requires --exclude-audit", rejected.stderr)
        self.assertFalse(output.exists())
        rejected = self.cli(*common, "--exclude-audit", self.bundle, success=False)
        self.assertIn("requires --all-tasks", rejected.stderr)
        self.assertFalse(output.exists())

    def test_prediction_tampering_cannot_be_scored(self):
        self.ingest_and_prepare()
        output = self.data / "dev.jsonl"
        self.cli("run", "--bundle", self.bundle, "--out", output)
        with output.open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        rejected = self.cli("evaluate", "--bundle", self.bundle, "--predictions", output,
                            "--out", self.data / "tampered.report.json", success=False)
        self.assertIn("prediction file changed", rejected.stderr)
        self.assertFalse((self.data / "tampered.report.json").exists())


if __name__ == "__main__":
    unittest.main()
