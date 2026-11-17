                                                                               

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from magra_benchmark.calibration import load_calibrator
from magra_benchmark.datasets import sha256_file
from magra_benchmark.protocol import load_json, read_jsonl
from magra_benchmark.splits import read_split_bundle, verify_split_bundle
from magra_rag.cli import build_parser, main
from magra_rag.storage import DocumentStore
from magra_rag.web import SnapshotStore, WebNetworkError, WebResponse


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "benchmark"
        self.db = self.root / "corpus.sqlite"
        for target in ("urllib.request.urlopen", "socket.create_connection", "socket.getaddrinfo"):
            blocker = patch(target, side_effect=AssertionError("Workflow tests forbid network access"))
            blocker.start()
            self.addCleanup(blocker.stop)
                                                                           
                                                                          
        for target in ("magra_benchmark.protocol.code_hash", "magra_benchmark.tuning.code_hash"):
            version = patch(target, return_value="a" * 64)
            version.start()
            self.addCleanup(version.stop)
        self.questions = [
            ("dev-aurora", "When was Aurora founded?", "1998", "dev"),
            ("dev-diamond", "Which mineral defines Mohs hardness ten?", "diamond", "dev"),
            ("dev-composer", "Who composed the Moonlight Sonata?", "Beethoven", "dev"),
            ("dev-jupiter", "Where is the Great Red Spot located?", "Jupiter", "dev"),
            ("eval-fusion", "What process powers the solar core?", "fusion", "test"),
            ("eval-hexagon", "How many sides form a hexagon?", "six", "test"),
        ]
        documents, labels = [], []
        for identifier, question, answer, split in self.questions:
            document_id = "owned:" + identifier
            documents.append({"id": document_id, "title": question,
                              "text": question + " The answer is " + answer + ".",
                              "metadata": {"source": "fixture", "domain": "owned"}})
            labels.append({"id": identifier, "question": question, "official_split": split,
                           "source": "fixture", "domain": "owned", "answer": answer,
                           "relevance": {document_id: 1},
                           "supporting_facts": [{"doc_id": document_id, "sentence_id": 0}]})
        documents_path, questions_path = self.root / "documents.jsonl", self.root / "questions.jsonl"
        self.write_jsonl(documents_path, documents)
        self.write_jsonl(questions_path, labels)
        self.invoke("ingest", "--format", "jsonl", documents_path)
        self.invoke("prepare", "--queries", questions_path, "--out", self.bundle)

    def invoke(self, *arguments, db=None):
        argv = ["--data-root", str(self.root)]
        if db is not None:
            argv += ["--db", str(db)]
        argv += [str(argument) for argument in arguments]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(argv)
        self.assertEqual(status, 0, f"CLI failed: {stderr.getvalue()}")
        self.assertEqual(stderr.getvalue(), "")
        return json.loads(stdout.getvalue())

    @staticmethod
    def write_jsonl(path, rows):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_hotpot_download_accepts_no_positional_beir_dataset(self):
        parsed = build_parser().parse_args(["download", "--hotpot", "train"])
        self.assertEqual(parsed.datasets, [])
        self.assertEqual(parsed.hotpot, "train")
        local = self.root / "sources" / "hotpot_train_v1.1.json"
        with patch("magra_benchmark.datasets.download_hotpot", return_value=local) as acquire:
            result = self.invoke("download", "--hotpot", "train", "--max-download-bytes", 10000,
                                 "--max-extracted-bytes", 10000)
        self.assertEqual(result["acquired"], [str(local)])
        self.assertEqual(acquire.call_args.args[0], "train")

    def test_calibrate_consumes_real_run_artifacts_and_emits_reloadable_v1_artifact(self):
        class OfflineGenerator:
            def __init__(self, **kwargs):
                self.model_id = kwargs["model_id"]
                self.max_tokens = kwargs["max_tokens"]
                self.reasoning_effort = kwargs["reasoning_effort"]
                self.ledger = SimpleNamespace(summary=lambda: {"accounted_usd": 0.004,
                                                               "unconfirmed_requests": 0})

            def generate(self, question, evidence, *, as_of=None):
                good = question.startswith(("When", "Which"))
                answer = "1998" if question.startswith("When") else "diamond"
                citation = {field: evidence[0][field] for field in ("doc_id", "sentence_id")}
                return {"answer": answer if good else "incorrect", "citations": [citation],
                        "confidence": 0.9 if good else 0.1, "abstained": False, "reason": "Offline fixture",
                        "tokens": 20, "cost_usd": 0.001, "model": self.model_id}

        predictions = self.root / "runs" / "m3-dev.jsonl"
        env = self.root / "offline.env"
        env.write_text("ai-model-key=offline-fixture-secret\nOPENROUTER_MODEL=example/offline\n", encoding="utf-8")
        with patch.dict("os.environ", {}, clear=True), patch("magra_rag.generation.OpenRouterM3", OfflineGenerator):
            self.invoke("run", "--bundle", self.bundle, "--split", "dev", "--strategy", "bm25",
                        "--generate", "--env-file", env, "--model", "example/offline", "--max-usd", 1, "--prompt-price", 2,
                        "--completion-price", 4, "--out", predictions)
        self.assertEqual(len(list(read_jsonl(predictions))), 4)
        self.assertEqual(load_json(str(predictions) + ".manifest.json")["status"], "complete")
        artifact_path = self.root / "runs" / "calibration.json"
        returned = self.invoke("calibrate", "--bundle", self.bundle, "--predictions", predictions,
                               "--target-accuracy", 0.9, "--min-accepted", 1, "--out", artifact_path)
        fitted = load_calibrator(artifact_path, expected_hash=returned["artifact_hash"])
        self.assertEqual(returned["version"], 1)
        self.assertEqual(fitted.to_dict()["fit_count"], 4)
        self.assertEqual(fitted.to_dict()["fit_split"], "dev")
        self.assertEqual(fitted.to_dict()["provenance"]["predictions_hash"], sha256_file(predictions))
        self.assertEqual(fitted.predict(0.9), 1)
        self.assertEqual(fitted.predict(0.1), 0)
        self.assertTrue(fitted.should_abstain(0.1))

    def test_free_generation_uses_env_model_without_a_project_spending_cap(self):
        calls = []

        class OfflineGenerator:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.model_id = kwargs["model_id"]
                self.max_tokens = kwargs["max_tokens"]
                self.reasoning_effort = kwargs["reasoning_effort"]
                self.ledger = SimpleNamespace(summary=lambda: {"limit_usd": None,
                                                               "accounted_usd": 0,
                                                               "unconfirmed_requests": 0})

            def generate(self, question, evidence, *, as_of=None):
                citation = {field: evidence[0][field] for field in ("doc_id", "sentence_id")}
                return {"answer": "fixture", "citations": [citation], "confidence": 0.2,
                        "abstained": False, "reason": "Offline fixture", "tokens": 20,
                        "cost_usd": 0, "model": calls[-1]["model_id"]}

        env = self.root / "private.env"
        env.write_text("ai-model-key=fixture-secret\nOPENROUTER_MODEL=example/future:free\n", encoding="utf-8")
        predictions = self.root / "runs" / "free-dev.jsonl"
        with patch.dict("os.environ", {}, clear=True), patch("magra_rag.generation.OpenRouterM3", OfflineGenerator):
            self.invoke("run", "--bundle", self.bundle, "--split", "dev", "--strategy", "bm25",
                        "--generate", "--env-file", env, "--reasoning-effort", "low",
                        "--max-output-tokens", 2048, "--out", predictions)
            doctor = self.invoke("doctor", "--env-file", env)
        self.assertIsNone(calls[0]["max_total_usd"])
        self.assertEqual(calls[0]["prompt_usd_per_million"], 0)
        self.assertEqual(calls[0]["completion_usd_per_million"], 0)
        metadata = load_json(str(predictions) + ".manifest.json")
        self.assertEqual(metadata["model"], "example/future:free")
        self.assertEqual(metadata["max_tokens"], 2048)
        self.assertEqual(metadata["reasoning_effort"], "low")
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(doctor["model"], "example/future:free")
        self.assertTrue(doctor["openrouter_key_present"])
        self.assertNotIn("fixture-secret", json.dumps(doctor))
        self.assertNotIn("fixture-secret", json.dumps(metadata))

    def test_judge_audit_cli_scores_independent_dev_and_eval_humans(self):
        humans, judgments = [], []
        for split in ("dev", "eval"):
            for number in range(10):
                qid, truth = f"human-{split}-{number}", number % 2
                humans.append({"query_id": qid, "split": split, "label": truth, "group_id": qid})
                judgments.append({"query_id": qid, "groundedness_probability": 0.9 if truth else 0.1})
        human_path, judgment_path = self.root / "audit" / "humans.jsonl", self.root / "audit" / "judgments.jsonl"
        self.write_jsonl(human_path, humans)
        self.write_jsonl(judgment_path, judgments)
        out = self.root / "audit" / "report.json"
        result = self.invoke("judge-audit", "--human-labels", human_path, "--judgments", judgment_path,
                             "--out", out)
        self.assertTrue(result["use_for_acceptance"])
        report = load_json(out)
        self.assertEqual(report["evaluation_metrics"]["balanced_accuracy"], 1)
        self.assertEqual(report["calibration_artifact"]["fit_count"], 10)
        self.assertTrue(all(qid.startswith("human-dev") for qid in
                            report["calibration_artifact"]["fit_query_ids"]))

    def test_judge_inputs_cli_pins_public_bytes_and_writes_uncalibrated_observations(self):
        from magra_rag.generation import OpenRouterM3
        public = next(read_split_bundle(self.bundle, "dev"))
        record = {"query": public,
                  "prediction": {"query_id": public["id"], "answer": "A fixture response.",
                                 "abstained": False, "citations": []},
                  "evidence": [{"doc_id": "fixture", "sentence_id": 0, "title": "Fixture",
                                "text": "A fixture response."}]}
        inputs = self.root / "audit" / "public-inputs.jsonl"
        self.write_jsonl(inputs, [record])
        out = self.root / "audit" / "raw-judge.jsonl"
        client = OpenRouterM3(data_root=self.root, api_key="offline-workflow-secret", env_file=None)

        def verdict(query, prediction, evidence):
            self.assertNotIn("answer", query)
            self.assertNotIn("relevance", query)
            return {"query_id": query["id"], "groundedness_probability": 0.9, "abstained": False,
                    "calibration_status": "not_human_audited", "model": client.model_id}

        with patch("magra_rag.cli._judge_client", return_value=client), patch(
                "magra_benchmark.judge_workflow.M3GroundednessJudge.judge", side_effect=verdict):
            result = self.invoke("judge-inputs", "--inputs", inputs, "--inputs-sha256", sha256_file(inputs),
                                 "--split", "dev", "--out", out)
        self.assertEqual(result["judgment_count"], 1)
        judgment = next(read_jsonl(out))
        self.assertEqual(judgment["query_id"], public["id"])
        self.assertEqual(judgment["group_id"], public["group_id"])
        self.assertEqual(judgment["source_inputs_sha256"], sha256_file(inputs))
        self.assertEqual(judgment["calibration_status"], "not_human_audited")
        self.assertEqual(load_json(str(out) + ".manifest.json")["status"], "complete")

    def test_failure_cli_keeps_private_annotations_outside_sql_and_public_questions(self):
        out = self.root / "failures"
        result = self.invoke("failures", "--seeds", 2, "--as-of", "2026-10-20", "--out", out)
        self.assertTrue(result["synthetic"])
        self.assertFalse(result["natural_coverage"])
        manifest = verify_split_bundle(out / "benchmark")
        self.assertEqual(sum(manifest["counts"].values()), 10)
        cases = set()
        with DocumentStore(out / "corpus.sqlite", readonly=True) as store:
            corpus = list(store.connection.execute("SELECT id,text,metadata FROM documents"))
        self.assertGreater(len(corpus), 0)
        for row in corpus:
            metadata = json.loads(row["metadata"])
            self.assertNotIn("answer", metadata)
            self.assertNotIn("relevance", metadata)
            self.assertNotIn("expected_behavior", metadata)
        for split in ("train", "dev", "eval"):
            for query in read_split_bundle(out / "benchmark", split):
                self.assertTrue(query["synthetic"])
                self.assertIn("allowed_doc_ids", query)
                self.assertNotIn("answers", query)
                self.assertNotIn("relevance", query)
                cases.add(query["scenario"])
        self.assertEqual(cases, {"missing_evidence", "conflicting_sources", "stale_facts",
                                "lexical_mismatch", "irrelevant_context"})
        self.assertTrue((out / "private_provenance.json").exists())

    def test_mocked_snapshot_and_live_commands_keep_tracks_separate(self):
        url = "https://example.org/source"
        response = WebResponse(url, url, 200, b"<title>Owned snapshot</title><p>Evidence.</p>",
                               "text/html; charset=utf-8", "2026-10-20T12:00:00+00:00")
        snapshot_path = self.root / "snapshots" / "example"
        with patch("magra_rag.web.BoundedWebClient.fetch", return_value=response) as fetch:
            snapshot = self.invoke("web-snapshot", url, "--allowed-domains", "example.org",
                                   "--max-bytes", 10000, "--out", snapshot_path)
        fetch.assert_called_once_with(url)
        document = snapshot["document"]
        self.assertEqual(document["metadata"]["track"], "frozen_web")
        replay = SnapshotStore(snapshot_path).load(document["metadata"]["snapshot_id"])
        self.assertEqual(replay["text"], document["text"])
        live_path = self.root / "live" / "run.json"
        with patch("magra_rag.web.BoundedWebClient.fetch",
                   side_effect=[response, WebNetworkError("Offline fixture failure")]):
            live = self.invoke("web-live", url, "https://example.org/failure", "--allowed-domains", "example.org",
                               "--max-bytes", 10000, "--out", live_path)
        self.assertEqual(live["summary"]["network_availability"], 0.5)
        report = load_json(live_path)
        self.assertEqual(report["track"], "live_web")
        self.assertFalse(report["summary"]["replayable"])
        self.assertEqual(report["acquisitions"][0]["document"]["metadata"]["track"], "live_web")

    def test_tune_freeze_locked_eval_evaluate_and_compare_cli_flow(self):
        tuning = self.root / "tuning"
        tuned = self.invoke("tune", "--bundle", self.bundle, "--limit-per-stratum", 4, "--out", tuning)
        self.assertEqual(len(tuned["candidates"]), 7)
        names = ("bm25_title_2", "magra")
        dev_reports = [f"{name}={tuning / (name + '.report.json')}" for name in names]
        configs = [f"{name}={tuning / (name + '.config.json')}" for name in names]
        frozen = self.root / "runs" / "protocol.json"
        self.invoke("freeze", "--bundle", self.bundle, "--dev-reports", *dev_reports,
                    "--configs", *configs, "--baselines", names[0], "--candidate", names[1],
                    "--limit-per-stratum", 2, "--out", frozen)
        locked = load_json(frozen)
        self.assertEqual(len(locked["eval_query_ids"]), 2)
        runs = []
        for name in names:
            predictions, report = self.root / "runs" / (name + "-eval.jsonl"), self.root / "runs" / (name + "-report.json")
            run = self.invoke("run", "--bundle", self.bundle, "--split", "eval", "--config",
                              tuning / (name + ".config.json"), "--limit-per-stratum", 2,
                              "--freeze", frozen, "--out", predictions)
            self.assertEqual(run["query_count"], 2)
            scored = self.invoke("evaluate", "--bundle", self.bundle, "--predictions", predictions, "--out", report)
            self.assertEqual(scored["missing_predictions"], 0)
            self.assertEqual(load_json(report)["splits"], ["eval"])
            runs.append(f"{name}={predictions}")
        comparison = self.root / "runs" / "comparison.json"
        self.invoke("compare", "--bundle", self.bundle, "--freeze", frozen, "--runs", *runs,
                    "--dev-reports", *dev_reports, "--out", comparison)
        compared = load_json(comparison)
        self.assertFalse(compared["generality_claim_supported"])
        self.assertEqual(compared["acceptance_status"], "not_assessed")
        self.assertTrue(all(group["complete_predictions"] for group in compared["groups"].values()))
        with DocumentStore(self.db, readonly=True) as store:
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 6)


if __name__ == "__main__":
    unittest.main()
