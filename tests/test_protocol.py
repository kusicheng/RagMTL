                                                                                  

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from magra_benchmark.datasets import sha256_file
from magra_benchmark.evaluation import evaluate_run
from magra_benchmark.protocol import freeze_protocol, sample_queries, verify_protocol
from magra_benchmark.splits import make_splits, read_split_bundle, write_split_bundle
from magra_rag.retrieval import RetrievalConfig


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundle"
        self.path = self.root / "preregistration.json"
                                                                              
                                                                              
        patcher = patch("magra_benchmark.protocol.code_hash", return_value="code-version-one")
        self.code_hash = patcher.start()
        self.addCleanup(patcher.stop)
        rows = [
            {"id": "fixture:dev", "question": "Which enzyme repairs ultraviolet damage to bacterial DNA?",
             "source": "fixture", "domain": "biology", "official_split": "dev",
             "answer": "PRIVATE_REFERENCE", "relevance": {"fixture:d1": 1}},
            {"id": "fixture:eval", "question": "Where does the Atlantic current release stored heat?",
             "source": "fixture", "domain": "biology", "official_split": "test",
             "answer": "PRIVATE_REFERENCE_TWO", "relevance": {"fixture:d2": 1}},
        ]
        write_split_bundle(self.bundle, make_splits(rows))
        self.configs = {"bm25": asdict(RetrievalConfig(strategy="bm25")),
                        "magra": asdict(RetrievalConfig(strategy="magra"))}
        self.runtime = {"generation": False, "model": None, "dense_model": None,
                        "reranker_model": None, "max_tokens": None}
        self.reports = {}
        for name, config in self.configs.items():
            self.reports[name] = evaluate_run(
                read_split_bundle(self.bundle, "dev", include_labels=True),
                [{"query_id": "fixture:dev", "ranked_doc_ids": ["fixture:d1"],
                  "latency_ms": 1, "cost_usd": 0, "tokens": 0}],
                run_metadata={"config": config, "corpus_hash": "owned-corpus-one",
                              "status": "complete",
                              **self.runtime,
                              "bundle_manifest_sha256": sha256_file(self.bundle / "manifest.json"),
                              "environment": {"code_hash": "code-version-one"},
                              "resources": {"process_peak_rss_bytes": 1_000_000,
                                            "peak_observed_physical_bytes": 2_000_000}})

    def freeze(self, **overrides):
        arguments = {"bundle": self.bundle, "corpus_hash": "owned-corpus-one",
                     "configs": self.configs, "dev_reports": self.reports,
                     "baseline_names": ["bm25"], "candidate_name": "magra"}
        arguments.update(overrides)
        return freeze_protocol(self.path, **arguments)

    def verify(self, **overrides):
        arguments = {"bundle": self.bundle, "corpus_hash": "owned-corpus-one",
                     "config": self.configs["magra"],
                     "runtime": self.runtime,
                     "queries": list(read_split_bundle(self.bundle, "eval"))}
        arguments.update(overrides)
        return verify_protocol(self.path, **arguments)

    def test_freeze_and_verify_measured_development_system(self):
        frozen = self.freeze()
        self.assertEqual(self.verify()["protocol_hash"], frozen["protocol_hash"])
        self.assertEqual(frozen["eval_query_ids"], ["fixture:eval"])
        selection = frozen["baseline_selection"]
        self.assertTrue(selection)
        self.assertTrue(any(row.get("baseline") == "bm25" for row in selection.values()))
        self.assertNotIn("PRIVATE_REFERENCE", self.path.read_text(encoding="utf-8"))

    def test_corpus_change_refused(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Corpus changed"):
            self.verify(corpus_hash="owned-corpus-two")

    def test_configuration_change_refused(self):
        self.freeze()
        changed = {**self.configs["magra"], "context_bytes": 13000}
        with self.assertRaisesRegex(ValueError, "Configuration"):
            self.verify(config=changed)

    def test_generation_reasoning_and_output_limit_are_frozen(self):
        self.runtime.update(generation=True, model="example/model:free", max_tokens=4096,
                            reasoning_effort="low")
        for report in self.reports.values():
            report["run_metadata"].update(self.runtime)
        self.freeze()
        self.verify()
        for runtime in ({**self.runtime, "reasoning_effort": "high"},
                        {**self.runtime, "max_tokens": 8192}):
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, "generation settings"):
                self.verify(runtime=runtime)

    def test_query_sample_change_refused(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "query"):
            self.verify(queries=[])

    def test_query_content_change_with_same_id_refused(self):
        self.freeze()
        queries = list(read_split_bundle(self.bundle, "eval"))
        queries[0]["question"] = "An altered question with the frozen ID"
        with self.assertRaisesRegex(ValueError, "[Qq]uer"):
            self.verify(queries=queries)

    def test_code_change_refused(self):
        self.freeze()
        self.code_hash.return_value = "code-version-two"
        with self.assertRaisesRegex(ValueError, "Implementation changed"):
            self.verify()

    def test_calibration_change_refused(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Calibration"):
            self.verify(calibration_hash="a-different-calibrator")

    def test_generation_or_model_revision_change_refused(self):
        self.freeze()
        for changed in ({"generation": True, "model": "minimax/minimax-m3", "max_tokens": 512},
                        {"dense_model": "same-model@different-commit"},
                        {"reranker_model": "same-model@different-commit"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "generation settings"):
                self.verify(runtime={**self.runtime, **changed})

    def test_development_measurements_from_old_code_refused(self):
        reports = deepcopy(self.reports)
        reports["bm25"]["run_metadata"]["environment"]["code_hash"] = "old-code"
        with self.assertRaisesRegex(ValueError, "implementation differs"):
            self.freeze(dev_reports=reports)

    def test_bundle_question_edit_refused(self):
        self.freeze()
        path = self.bundle / "public/eval.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.verify(queries=[])

    def test_bundle_manifest_edit_refused(self):
        self.freeze()
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["provenance"]["modified"] = True
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.verify()

    def test_preregistration_edit_refused(self):
        self.freeze()
        frozen = json.loads(self.path.read_text(encoding="utf-8"))
        frozen["corpus_hash"] = "edited-after-freeze"
        self.path.write_text(json.dumps(frozen), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.verify()

    def test_preregistration_refuses_overwrite(self):
        self.freeze()
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_eval_report_cannot_select_baselines(self):
        reports = deepcopy(self.reports)
        reports["bm25"]["splits"] = ["eval"]
        with self.assertRaisesRegex(ValueError, "[Dd]ev"):
            self.freeze(dev_reports=reports)

    def test_development_report_cannot_describe_different_config(self):
        reports = deepcopy(self.reports)
        reports["bm25"]["run_metadata"]["config"]["context_bytes"] += 1
        with self.assertRaises(ValueError):
            self.freeze(dev_reports=reports)

    def test_development_report_cannot_describe_different_corpus(self):
        reports = deepcopy(self.reports)
        reports["bm25"]["run_metadata"]["corpus_hash"] = "different-corpus"
        with self.assertRaises(ValueError):
            self.freeze(dev_reports=reports)

    def test_development_report_cannot_describe_different_bundle(self):
        reports = deepcopy(self.reports)
        reports["bm25"]["run_metadata"]["bundle_manifest_sha256"] = "different-bundle"
        with self.assertRaises(ValueError):
            self.freeze(dev_reports=reports)

    def test_sampling_is_deterministic_per_stratum_and_preserves_input(self):
        records = [{"id": f"domain{domain}:{setting}:{number}", "source": str(domain), "domain": str(domain),
                    "setting": setting, "track": "natural"}
                   for domain in range(2) for setting in ("provided_context", "full_corpus")
                   for number in range(8)]
        before = deepcopy(records)
        sampled = sample_queries(records, 2)
        self.assertEqual(sampled, sample_queries(reversed(records), 2))
        self.assertEqual(len(sampled), 8)
        self.assertEqual(records, before)


if __name__ == "__main__":
    unittest.main()
