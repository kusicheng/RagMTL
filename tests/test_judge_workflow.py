                                                                                   

import copy
import json
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import patch

from magra_benchmark.datasets import sha256_file
from magra_benchmark.judge import JudgeError
from magra_benchmark import judge_workflow as workflow
from magra_benchmark.judge_workflow import JudgeWorkflowError, judge_inputs, judge_run
from magra_benchmark.protocol import load_json, read_jsonl
from magra_benchmark.splits import PUBLIC_FIELDS, make_splits, read_split_bundle, write_split_bundle
from magra_rag.generation import MODEL, IncompleteResponseError, OpenRouterM3, PaidHTTPError, RateLimitError
from magra_rag.resources import ResourceBudget


SECRET = "offline-only-workflow-key"


class Response:
    def __init__(self, value):
        self.raw = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.raw[:limit]


def offline_judgment(query, prediction, evidence):
    return {"query_id": query["id"], "groundedness_probability": 0.9, "abstained": False,
            "explanation": "Offline workflow fixture", "unsupported_claims": [],
            "model": MODEL, "cost_usd": 0, "tokens": 20,
            "confidence_kind": "uncalibrated_model_self_report", "calibration_status": "not_human_audited"}


class JudgeWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.budget = ResourceBudget(self.root)
        self.bundle = self.root / "benchmark"
        rows = [
            {"id": "aurora", "question": "When was Aurora founded?", "official_split": "dev",
             "answer": "PRIVATE_REFERENCE_ONE", "relevance": {"doc:aurora": 1},
             "source": "fixture", "domain": "history", "setting": "full_corpus"},
            {"id": "composer", "question": "Who composed the Moonlight Sonata?", "official_split": "dev",
             "answer": "PRIVATE_REFERENCE_TWO", "relevance": {"doc:composer": 1},
             "source": "fixture", "domain": "music", "setting": "full_corpus"},
            {"id": "solar", "question": "Which process powers the solar core?", "official_split": "test",
             "answer": "PRIVATE_REFERENCE_THREE", "source": "fixture", "domain": "physics"},
        ]
        write_split_bundle(self.bundle, make_splits(rows))
        self.public = list(read_split_bundle(self.bundle, "dev", include_labels=False))
        self.rows = [{"query_id": query["id"], "answer": "A generated answer.", "abstained": False,
                      "citations": [{"doc_id": "doc:" + query["id"], "sentence_id": 0}],
                      "evidence": [{"doc_id": "doc:" + query["id"], "sentence_id": 0,
                                    "title": "Fixture", "text": "A supplied evidence sentence."}],
                      "model": MODEL} for query in self.public]
        self.predictions = self.artifact()
        self.out = self.root / "runs" / "judgments.jsonl"
        self.client = OpenRouterM3(data_root=self.root, api_key=SECRET, model_id=MODEL,
                                   env_file=None, budget=self.budget)
        network = patch("urllib.request.urlopen", side_effect=AssertionError("Offline tests forbid network"))
        self.urlopen = network.start()
        self.addCleanup(network.stop)
        environment = patch("magra_benchmark.judge_workflow.environment", return_value={"code_hash": "offline"})
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def write_rows(path, rows):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def artifact(self, rows=None, *, basename="predictions", **changes):
        path = self.root / "runs" / (basename + ".jsonl")
        values = copy.deepcopy(self.rows if rows is None else rows)
        self.write_rows(path, values)
        metadata = {"format": "magra-run-v1", "status": "complete", "generation": True,
                    "split": "dev", "model": MODEL, "prediction_count": len(values),
                    "query_ids": [row["query_id"] for row in values],
                    "bundle_manifest_sha256": sha256_file(self.bundle / "manifest.json"),
                    "predictions_sha256": sha256_file(path)}
        metadata.update(changes)
        Path(str(path) + ".manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def run_artifact(self, **changes):
        args = {"bundle": self.bundle, "predictions": self.predictions, "out": self.out,
                "client": self.client, "budget": self.budget}
        args.update(changes)
        return judge_run(**args)

    def inputs(self, rows=None):
        path = self.root / "public" / "dev.inputs.jsonl"
        values = rows if rows is not None else [
            {"query": query, "prediction": {key: value for key, value in row.items() if key != "evidence"},
             "evidence": row["evidence"]} for query, row in zip(self.public, self.rows, strict=True)]
        self.write_rows(path, values)
        return path

    def run_inputs(self, path, **changes):
        args = {"inputs": path, "expected_sha256": sha256_file(path), "out": self.out,
                "client": self.client, "budget": self.budget}
        args.update(changes)
        return judge_inputs(**args)

    def mocked_judge(self, **changes):
        return patch("magra_benchmark.judge_workflow.M3GroundednessJudge.judge",
                     side_effect=offline_judgment, **changes)

    def test_complete_run_namespaces_observations_and_preserves_public_groups(self):
        with self.mocked_judge() as judge:
            result = self.run_artifact()
        rows = list(read_jsonl(self.out))
        manifest = load_json(str(self.out) + ".manifest.json")
        namespace = "prediction:" + sha256_file(self.predictions) + ":"
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["observation_namespace"], namespace)
        self.assertEqual(judge.call_count, 2)
        for row, query in zip(rows, self.public, strict=True):
            self.assertEqual(row["query_id"], namespace + query["id"])
            self.assertEqual(row["observation_id"], row["query_id"])
            self.assertEqual(row["source_query_id"], query["id"])
            self.assertEqual(row["group_id"], query["group_id"])
            self.assertEqual(row["split"], "dev")
            self.assertEqual(row["calibration_status"], "not_human_audited")
        self.assertEqual(manifest["judgments_sha256"], sha256_file(self.out))
        self.assertEqual(manifest["judgment_count"], 2)
        self.assertEqual(manifest["status"], "complete")
        self.assertIn("resources", manifest)
        self.assertIn("spend_before", manifest)
        self.assertIn("spend", manifest)
        self.assertIsNone(manifest["reasoning_effort"])
        self.assertIsNone(manifest["runtime"]["reasoning_effort"])
        self.assertEqual(manifest["runtime"]["model"], MODEL)
        self.assertRegex(manifest["runtime_sha256"], r"^[a-f0-9]{64}$")
        self.assertNotIn(SECRET, self.out.read_text() + json.dumps(manifest))

    def test_different_predictions_have_different_observations_with_same_groups(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["answer"] = "A different generated answer."
        other = self.artifact(changed, basename="other")
        with self.mocked_judge():
            self.run_artifact()
            self.run_artifact(predictions=other, out=self.root / "other-judgments.jsonl")
        first = list(read_jsonl(self.out))[0]
        second = list(read_jsonl(self.root / "other-judgments.jsonl"))[0]
        self.assertNotEqual(first["query_id"], second["query_id"])
        self.assertEqual(first["source_query_id"], second["source_query_id"])
        self.assertEqual(first["group_id"], second["group_id"])

    def test_bundle_and_prediction_hash_tampering_never_reaches_judge(self):
        with self.mocked_judge() as judge:
            self.predictions.write_text(self.predictions.read_text() + "\n")
            with self.assertRaisesRegex(JudgeWorkflowError, "hash"):
                self.run_artifact()
            self.predictions = self.artifact()
            public = self.bundle / "public/dev.jsonl"
            public.write_text(public.read_text() + "\n")
            with self.assertRaises(JudgeWorkflowError):
                self.run_artifact()
        judge.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_changed_bundle_manifest_is_rejected_before_inference(self):
        manifest = self.bundle / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        with self.mocked_judge() as judge:
            with self.assertRaisesRegex(JudgeWorkflowError, "bundle"):
                self.run_artifact()
        judge.assert_not_called()

    def test_only_complete_generated_runs_are_accepted(self):
        for changes in ({"status": "running"}, {"generation": False}, {"generation": 1},
                        {"prediction_count": 1}, {"split": "unknown"}):
            with self.subTest(changes=changes):
                path = self.artifact(**changes)
                with self.mocked_judge() as judge:
                    with self.assertRaises(JudgeWorkflowError):
                        self.run_artifact(predictions=path)
                judge.assert_not_called()

    def test_late_extra_duplicate_and_missing_ids_are_rejected_before_first_call(self):
        cases = []
        extra = copy.deepcopy(self.rows)
        extra[-1]["query_id"] = "not-in-manifest"
        cases.append((extra, {"query_ids": [row["query_id"] for row in self.rows]}))
        duplicate = [copy.deepcopy(self.rows[0]), copy.deepcopy(self.rows[0])]
        cases.append((duplicate, {"query_ids": [row["query_id"] for row in self.rows]}))
        cases.append((self.rows[:1], {"query_ids": [row["query_id"] for row in self.rows], "prediction_count": 2}))
        cases.append((self.rows, {"query_ids": [self.rows[0]["query_id"], "absent-from-bundle"]}))
        for rows, changes in cases:
            with self.subTest(ids="invalid"):
                path = self.artifact(rows, **changes)
                with self.mocked_judge() as judge:
                    with self.assertRaises(JudgeWorkflowError):
                        self.run_artifact(predictions=path)
                judge.assert_not_called()

    def test_late_annotation_or_bad_evidence_is_validated_before_any_request(self):
        for field, value in (("label", 1), ("evidence", None), ("answer", 5)):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[-1][field] = value
                path = self.artifact(rows)
                with self.mocked_judge() as judge:
                    with self.assertRaises((JudgeError, JudgeWorkflowError)):
                        self.run_artifact(predictions=path)
                judge.assert_not_called()

    def test_private_labels_are_never_decoded_or_sent_in_actual_judge_prompt(self):
        catalog = {"data": [{"id": MODEL, "supported_parameters": ["response_format"],
                             "pricing": {"prompt": "0", "completion": "0"}}]}
        completion = {"model": MODEL, "usage": {"cost": 0, "total_tokens": 50},
                      "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                          "groundedness_probability": 0.5, "abstained": False,
                          "explanation": "Offline", "unsupported_claims": []})}}]}
        self.urlopen.side_effect = [Response(value) for value in (catalog, completion, catalog, completion)]
        with patch("magra_benchmark.judge_workflow.read_split_bundle", wraps=read_split_bundle) as read:
            self.run_artifact()
        self.assertTrue(all(call.kwargs["include_labels"] is False for call in read.call_args_list))
        requests = [call.args[0] for call in self.urlopen.call_args_list if call.args[0].get_method() == "POST"]
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertNotIn("PRIVATE_REFERENCE", request.data.decode())
            body = json.loads(request.data)
            query = json.loads(body["messages"][1]["content"])["query"]
            self.assertTrue(set(query) <= PUBLIC_FIELDS)
            self.assertNotIn("answer", query)

    def test_partial_failure_keeps_raw_rows_and_failed_hashed_manifest(self):
        with patch("magra_benchmark.judge_workflow.M3GroundednessJudge.judge",
                   side_effect=[offline_judgment(self.public[0], {}, []), JudgeError("Offline failure")]) as judge:
            with self.assertRaises(JudgeError):
                self.run_artifact()
        manifest = load_json(str(self.out) + ".manifest.json")
        self.assertEqual(judge.call_count, 2)
        self.assertEqual(len(list(read_jsonl(self.out))), 1)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["judgment_count"], 1)
        self.assertEqual(manifest["judgments_sha256"], sha256_file(self.out))
        self.assertEqual(manifest["failure"], {"type": "JudgeError"})

    def test_arbitrary_failure_does_not_copy_key_or_raw_body_into_diagnostics(self):
        with patch("magra_benchmark.judge_workflow.M3GroundednessJudge.judge",
                   side_effect=RuntimeError("RAW-BODY-" + SECRET)):
            with self.assertRaises(JudgeWorkflowError) as caught:
                self.run_artifact()
        manifest = load_json(str(self.out) + ".manifest.json")
        diagnostic = "".join(traceback.format_exception(caught.exception)) + json.dumps(manifest)
        self.assertNotIn(SECRET, diagnostic)
        self.assertNotIn("RAW-BODY", diagnostic)
        self.assertEqual(manifest["status"], "failed")

    def test_reasoning_effort_is_persisted_and_changes_runtime_identity(self):
        path = self.inputs()
        default_out, low_out = self.root / "default.jsonl", self.root / "low.jsonl"
        low = OpenRouterM3(data_root=self.root, api_key=SECRET, model_id=MODEL,
                           env_file=None, budget=self.budget, reasoning_effort="low")
        with self.mocked_judge():
            self.run_inputs(path, out=default_out)
            self.run_inputs(path, out=low_out, client=low)
        default = load_json(str(default_out) + ".manifest.json")
        explicit = load_json(str(low_out) + ".manifest.json")
        self.assertIsNone(default["reasoning_effort"])
        self.assertEqual(explicit["reasoning_effort"], "low")
        self.assertEqual(explicit["runtime"]["reasoning_effort"], "low")
        self.assertEqual(explicit["runtime"]["model"], MODEL)
        self.assertNotEqual(default["runtime_sha256"], explicit["runtime_sha256"])

    def test_incomplete_response_writes_safe_finish_usage_and_partial_failure(self):
        self.client = OpenRouterM3(data_root=self.root, api_key=SECRET, model_id=MODEL,
                                    env_file=None, budget=self.budget, max_tokens=4096,
                                    reasoning_effort="low")
        catalog = {"data": [{"id": MODEL, "supported_parameters": ["response_format", "reasoning"],
                             "reasoning": {"supported_efforts": ["low", "high"], "mandatory": False},
                             "pricing": {"prompt": "0", "completion": "0"}}]}
        finished = {"model": MODEL, "usage": {"cost": 0}, "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"groundedness_probability": 0.5, "abstained": False,
                                   "explanation": "Offline", "unsupported_claims": []})}}]}
        incomplete = {"model": MODEL, "usage": {"cost": 0, "prompt_tokens": 100,
                      "completion_tokens": 4096, "total_tokens": 4196,
                      "completion_tokens_details": {"reasoning_tokens": 4096, "raw_body": SECRET}},
                      "choices": [{"finish_reason": "length", "message": {
                          "content": SECRET, "reasoning": SECRET}}]}
        self.urlopen.side_effect = [Response(value) for value in (catalog, finished, catalog, incomplete)]
        with self.assertRaises(IncompleteResponseError) as caught:
            self.run_artifact()
        manifest = load_json(str(self.out) + ".manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["judgment_count"], 1)
        self.assertEqual(manifest["reasoning_effort"], "low")
        self.assertEqual(manifest["failure"], {"type": "incomplete_response", "finish_reason": "length",
                         "token_usage": {"prompt_tokens": 100, "completion_tokens": 4096,
                                         "total_tokens": 4196, "reasoning_tokens": 4096,
                                         "visible_completion_tokens": 0}})
        self.assertEqual(manifest["failed_source_query_id"], self.public[1]["id"])
        self.assertNotIn(SECRET, json.dumps(manifest) + self.out.read_text()
                         + "".join(traceback.format_exception(caught.exception)))
        posts = [call for call in self.urlopen.call_args_list if call.args[0].get_method() == "POST"]
        self.assertEqual(len(posts), 2)

    def test_rate_limit_is_a_failed_partial_artifact_with_safe_hints_and_no_retry(self):
        with patch("magra_benchmark.judge_workflow.M3GroundednessJudge.judge",
                   side_effect=RateLimitError(retry_after_seconds=23,
                                              rate_limit={"x-ratelimit-remaining": 0})) as judge:
            with self.assertRaises(RateLimitError):
                self.run_artifact()
        self.assertEqual(judge.call_count, 1)
        manifest = load_json(str(self.out) + ".manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["judgment_count"], 0)
        self.assertEqual(manifest["failure"]["retry_after_seconds"], 23)

    def test_outputs_are_exclusive_and_must_stay_inside_accounted_root(self):
        with self.mocked_judge():
            self.run_artifact()
        before = self.out.read_bytes()
        with self.mocked_judge() as judge:
            with self.assertRaises(FileExistsError):
                self.run_artifact()
            with self.assertRaises(ValueError):
                self.run_artifact(out=self.root.parent / "outside-judge.jsonl")
        judge.assert_not_called()
        self.assertEqual(self.out.read_bytes(), before)

    def test_existing_sidecar_also_blocks_output_reuse(self):
        Path(str(self.out) + ".manifest.json").write_text("{}")
        with self.mocked_judge() as judge:
            with self.assertRaises(FileExistsError):
                self.run_artifact()
        judge.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_source_mutation_during_inference_cannot_produce_complete_manifest(self):
        def mutate(query, prediction, evidence):
            self.predictions.write_text(self.predictions.read_text() + "\n")
            return offline_judgment(query, prediction, evidence)
        with patch("magra_benchmark.judge_workflow.M3GroundednessJudge.judge", side_effect=mutate):
            with self.assertRaises(JudgeWorkflowError):
                self.run_artifact()
        self.assertEqual(load_json(str(self.out) + ".manifest.json")["status"], "failed")

    def test_public_inputs_keep_identity_and_group_for_independent_human_rows(self):
        path = self.inputs()
        with self.mocked_judge() as judge:
            result = self.run_inputs(path)
        self.assertEqual(judge.call_count, 2)
        self.assertIsNone(result["observation_namespace"])
        for row, query in zip(read_jsonl(self.out), self.public, strict=True):
            self.assertEqual(row["query_id"], query["id"])
            self.assertEqual(row["observation_id"], query["id"])
            self.assertEqual(row["source_query_id"], query["id"])
            self.assertEqual(row["group_id"], query["group_id"])
            self.assertEqual(row["split"], "dev")
        manifest = load_json(str(self.out) + ".manifest.json")
        self.assertEqual(manifest["inputs_sha256"], sha256_file(path))
        self.assertEqual(manifest["source_kind"], "public_inputs")
        self.assertEqual(manifest["splits"], ["dev"])

    def test_public_inputs_require_correct_explicit_hash_before_calls(self):
        path = self.inputs()
        for expected in (None, "", "a" * 64):
            with self.subTest(pin=expected):
                with self.mocked_judge() as judge:
                    with self.assertRaises(JudgeWorkflowError):
                        self.run_inputs(path, expected_sha256=expected)
                judge.assert_not_called()

    def test_public_inputs_prevalidate_late_duplicate_private_field_and_split(self):
        original = list(read_jsonl(self.inputs()))
        variants = []
        duplicate = copy.deepcopy(original)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        variants.append(duplicate)
        private = copy.deepcopy(original)
        private[-1]["query"]["answer"] = "PRIVATE_REFERENCE"
        variants.append(private)
        extra = copy.deepcopy(original)
        extra[-1]["label"] = 1
        variants.append(extra)
        wrong_split = copy.deepcopy(original)
        wrong_split[-1]["query"]["split"] = "eval"
        variants.append(wrong_split)
        for values in variants:
            with self.subTest(input="invalid-late-row"):
                path = self.inputs(values)
                with self.mocked_judge() as judge:
                    with self.assertRaises((JudgeWorkflowError, JudgeError)):
                        self.run_inputs(path, split="dev")
                judge.assert_not_called()

    def test_public_input_row_count_and_bytes_are_bounded_before_calls(self):
        path = self.inputs()
        with self.mocked_judge() as judge:
            with patch("magra_benchmark.judge_workflow.MAX_AUDIT_INPUT_BYTES", 1):
                with self.assertRaisesRegex(JudgeWorkflowError, "64 MB"):
                    self.run_inputs(path)
            with patch("magra_benchmark.judge_workflow.MAX_AUDIT_INPUT_ROWS", 1):
                with self.assertRaisesRegex(JudgeWorkflowError, "1000"):
                    self.run_inputs(path)
        judge.assert_not_called()

    def test_public_manifest_can_bind_basename_hash_without_private_metadata(self):
        path = self.inputs()
        manifest = path.parent / "manifest.json"
        manifest.write_text(json.dumps({"format": "magra-ragtruth-public-inputs-v1", "files": {
            path.name: {"sha256": sha256_file(path), "records": 2}}}))
        with self.mocked_judge():
            self.run_inputs(path, input_manifest=manifest)
        output = load_json(str(self.out) + ".manifest.json")
        self.assertEqual(output["input_manifest_sha256"], sha256_file(manifest))

    def test_wrong_public_manifest_hash_is_rejected_without_requests(self):
        path = self.inputs()
        manifest = path.parent / "manifest.json"
        manifest.write_text(json.dumps({"format": "magra-ragtruth-public-inputs-v1", "files": {
            path.name: {"sha256": "a" * 64, "records": 2}}}))
        with self.mocked_judge() as judge:
            with self.assertRaises(JudgeWorkflowError):
                self.run_inputs(path, input_manifest=manifest)
        judge.assert_not_called()

    def test_interval_paces_call_starts_and_records_queue_wait_separately(self):
        path = self.inputs()
        for generic in (False, True):
            with self.subTest(generic=generic):
                clock, starts, waits = {"now": 0.0}, [], []

                def sleep(seconds):
                    self.assertGreaterEqual(seconds, 0)
                    waits.append(seconds)
                    clock["now"] += seconds

                def judge(query, prediction, evidence):
                    starts.append(clock["now"])
                    clock["now"] += 1
                    return {**offline_judgment(query, prediction, evidence), "latency_ms": 1000}

                out = self.root / ("paced-inputs.jsonl" if generic else "paced-run.jsonl")
                with patch("magra_benchmark.judge_workflow.time.monotonic", side_effect=lambda: clock["now"]), \
                        patch("magra_benchmark.judge_workflow.time.sleep", side_effect=sleep), \
                        patch("magra_benchmark.judge_workflow.M3GroundednessJudge.judge", side_effect=judge):
                    if generic:
                        self.run_inputs(path, out=out, min_request_interval_seconds=5)
                    else:
                        self.run_artifact(out=out, min_request_interval_seconds=5)
                self.assertEqual(starts, [0, 5])
                self.assertEqual(waits, [4])
                rows = list(read_jsonl(out))
                self.assertEqual([row["queue_wait_ms"] for row in rows], [0, 4000])
                self.assertEqual([row["latency_ms"] for row in rows], [1000, 1000])
                self.assertEqual(load_json(str(out) + ".manifest.json")["min_request_interval_seconds"], 5)

    def test_invalid_interval_is_rejected_before_any_calls_or_output(self):
        path = self.inputs()
        for interval in (True, None, -1, 61, float("inf"), float("nan")):
            with self.subTest(interval=interval):
                with self.mocked_judge() as judge:
                    with self.assertRaisesRegex(JudgeWorkflowError, "interval"):
                        self.run_artifact(min_request_interval_seconds=interval)
                    with self.assertRaisesRegex(JudgeWorkflowError, "interval"):
                        self.run_inputs(path, min_request_interval_seconds=interval)
                judge.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_paid402_survives_secondary_summary_hash_and_manifest_failures(self):
        path = self.inputs()
        original_hash, original_summary = workflow.sha256_file, self.client.ledger.summary
        for site in ("summary", "hash", "manifest"):
            with self.subTest(reporting_failure=site):
                out = self.root / ("paid402-" + site + ".jsonl")
                state = {"failed": False}
                primary = PaidHTTPError(402)

                def judge(*_):
                    state["failed"] = True
                    raise primary

                def summary():
                    if state["failed"] and site == "summary":
                        raise OSError(SECRET)
                    return original_summary()

                def digest(file):
                    if state["failed"] and site == "hash" and Path(file).resolve() == out.resolve():
                        raise OSError(SECRET)
                    return original_hash(file)

                def manifest(*args):
                    if state["failed"] and site == "manifest":
                        raise OSError(SECRET)
                    return original_replace(*args)

                original_replace = workflow._replace_manifest
                with patch.object(workflow.M3GroundednessJudge, "judge", side_effect=judge), \
                        patch.object(self.client.ledger, "summary", side_effect=summary), \
                        patch.object(workflow, "sha256_file", side_effect=digest), \
                        patch.object(workflow, "_replace_manifest", side_effect=manifest):
                    with self.assertRaises(PaidHTTPError) as caught:
                        self.run_inputs(path, out=out)
                self.assertIs(caught.exception, primary)
                self.assertEqual(caught.exception.status_code, 402)
                self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))
                if site != "manifest":
                    metadata = load_json(str(out) + ".manifest.json")
                    self.assertEqual(metadata["status"], "failed")
                    self.assertEqual(metadata["failure"]["status_code"], 402)
                    self.assertTrue(metadata["failure"]["goal_stop_required"])
                    self.assertTrue(metadata["failure_reporting_incomplete"])
        self.urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
