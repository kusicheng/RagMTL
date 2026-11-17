                                                                                

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from magra_benchmark.datasets import sha256_file
from magra_benchmark.protocol import read_jsonl
from magra_benchmark.ragtruth import (
    REPOSITORY, RAGTRUTH_REVISION, _RAGTruthRedirect, acquire_ragtruth, prepare_ragtruth_audit,
)
from magra_rag.resources import BudgetExceeded, ResourceBudget


REVISION = "a" * 40


def source(identifier):
    return {"source_id": identifier, "task_type": "QA", "source": "MARCO",
            "source_info": {"question": "Question about " + identifier + "?",
                            "passages": (identifier + " unique context ") * 3},
            "prompt": "PRIVATE_GENERATION_PROMPT_SENTINEL"}


def response(identifier, source_id, split, label, *, implicit_true=False):
    answer = "The context supplies the answer." if label else "The answer adds unsupported information."
    start = answer.find("unsupported")
    spans = [] if label else [{"start": start, "end": start + len("unsupported"), "text": "unsupported",
                               "label_type": "Evident Baseless Info", "implicit_true": implicit_true,
                               "due_to_null": False, "meta": "PRIVATE_ANNOTATOR_SENTINEL"}]
    return {"id": identifier, "source_id": source_id, "model": "PRIVATE_ORIGINAL_MODEL_SENTINEL",
            "quality": "good", "split": split, "response": answer, "labels": spans, "temperature": 0.7}


class RAGTruthTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_root = self.root / "publisher"
        self.dataset = self.source_root / "dataset"
        self.dataset.mkdir(parents=True)
        self.budget = ResourceBudget(self.root / "outputs", disk_bytes=100_000_000, reserve_bytes=0)
        self.sources, self.responses = [], []
        for split, prefix in (("train", "dev"), ("test", "eval")):
            for label in (0, 1):
                identifier = prefix + ("negative" if label == 0 else "positive")
                self.sources.append(source(identifier))
                self.responses.append(response("response-" + identifier, identifier, split, label))
        self.write_inputs()

    def write_inputs(self):
        records = {}
        for name, rows in (("source_info.jsonl", self.sources), ("response.jsonl", self.responses)):
            path = self.dataset / name
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            raw = path.read_bytes()
            records["dataset/" + name] = {
                "sha256": sha256_file(path), "bytes": len(raw),
                "url": f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{REVISION}/dataset/{name}",
                "git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(),
            }
        (self.source_root / "acquisition.json").write_text(json.dumps({
            "format": "magra-ragtruth-acquisition-v1", "repository": REPOSITORY,
            "revision": REVISION, "files": records,
        }), encoding="utf-8")

    def prepare(self, name="audit", **kwargs):
        return prepare_ragtruth_audit(self.source_root, self.budget.root / name, self.budget,
                                     dev_per_class=1, eval_per_class=1, **kwargs)

    def test_human_label_mapping_includes_implicit_true_as_unsupported(self):
        self.responses[0]["labels"][0]["implicit_true"] = True
        self.responses[0]["labels"][0]["meta"] = None                                                    
        self.write_inputs()
        result = self.prepare()
        humans = list(read_jsonl(result["human_labels"]))
        self.assertEqual(result["selected_counts"], {"dev": {"0": 1, "1": 1}, "eval": {"0": 1, "1": 1}})
        for human in humans:
            provenance = human["annotation_provenance"]
            self.assertEqual(human["label"], int(not human["annotation_spans"]))
            self.assertEqual(human["split"], "dev" if provenance["official_split"] == "train" else "eval")
            if provenance["response_id"] == "response-devnegative":
                self.assertEqual(human["label"], 0)
                self.assertTrue(provenance["contains_implicit_true"])
        self.assertEqual(json.loads(Path(result["manifest"]).read_text())["status"], "prepared_not_judged")

    def test_annotations_model_quality_and_original_prompt_never_enter_judge_inputs(self):
        result = self.prepare()
        forbidden_keys = {"labels", "label", "quality", "meta", "model", "original_model", "annotation_spans",
                          "label_type", "implicit_true", "due_to_null", "start", "end"}

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden_keys)
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        contexts = {row["source_info"]["passages"] for row in self.sources}
        for path in result["public_inputs"].values():
            raw = Path(path).read_text(encoding="utf-8")
            for sentinel in ("PRIVATE_ANNOTATOR_SENTINEL", "PRIVATE_ORIGINAL_MODEL_SENTINEL", "PRIVATE_GENERATION_PROMPT_SENTINEL"):
                self.assertNotIn(sentinel, raw)
            for row in read_jsonl(path):
                self.assertEqual(set(row), {"query", "prediction", "evidence"})
                self.assertEqual(row["query"]["id"], row["prediction"]["query_id"])
                self.assertFalse(row["prediction"]["abstained"])
                self.assertEqual(row["prediction"]["citations"], [])
                self.assertIn(row["evidence"][0]["text"], contexts)
                inspect(row)
        public_manifest = json.loads(Path(result["public_manifest"]).read_text())
        self.assertEqual(set(public_manifest["files"]), {"dev.inputs.jsonl", "eval.inputs.jsonl"})
        self.assertNotIn("human-labels", Path(result["public_manifest"]).read_text())

    def test_shared_context_across_official_splits_excludes_train_without_promoting_it(self):
        spare = source("sparepositive")
        self.sources.append(spare)
        self.responses.append(response("response-spare", spare["source_id"], "train", 1))
        dev_positive = next(row for row in self.sources if row["source_id"] == "devpositive")
        eval_negative = next(row for row in self.sources if row["source_id"] == "evalnegative")
        dev_positive["source_info"]["passages"] = eval_negative["source_info"]["passages"]
        self.write_inputs()
        result = self.prepare()
        humans = list(read_jsonl(result["human_labels"]))
        ids = {row["annotation_provenance"]["source_id"] for row in humans}
        self.assertNotIn("devpositive", ids)
        self.assertIn("evalnegative", ids)
        dev_groups = {row["group_id"] for row in humans if row["split"] == "dev"}
        eval_groups = {row["group_id"] for row in humans if row["split"] == "eval"}
        self.assertFalse(dev_groups & eval_groups)
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["exclusions"]["train_group_touches_official_test"], 1)

    def test_near_duplicate_questions_also_protect_official_split_boundary(self):
        self.sources.append(source("sparepositive"))
        self.responses.append(response("response-spare", "sparepositive", "train", 1))
        shared = " ".join(f"token{number}" for number in range(20))
        for row in self.sources:
            if row["source_id"] in {"devpositive", "evalnegative"}:
                row["source_info"]["question"] = shared + " " + row["source_id"]
        self.write_inputs()
        result = self.prepare()
        selected = [row["annotation_provenance"]["source_id"] for row in read_jsonl(result["human_labels"])]
        self.assertNotIn("devpositive", selected)

    def test_one_response_per_group_balanced_selection_and_seed_are_deterministic(self):
        self.responses.append(response("another-dev-positive", "devnegative", "train", 1))
        self.responses.append(response("another-eval-positive", "evalnegative", "test", 1))
        self.write_inputs()
        first, second = self.prepare("first", seed="fixed"), self.prepare("second", seed="fixed")
        for split in ("dev", "eval"):
            self.assertEqual(Path(first["public_inputs"][split]).read_bytes(), Path(second["public_inputs"][split]).read_bytes())
            self.assertEqual(Path(first["private_labels"][split]).read_bytes(), Path(second["private_labels"][split]).read_bytes())
        humans = list(read_jsonl(first["human_labels"]))
        self.assertEqual(len({row["group_id"] for row in humans}), len(humans))
        self.assertEqual(len({row["annotation_provenance"]["source_id"] for row in humans}), len(humans))

    def test_output_is_immutable_and_tampered_acquisition_is_rejected(self):
        result = self.prepare()
        before = Path(result["manifest"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(Path(result["manifest"]).read_bytes(), before)
        with (self.dataset / "response.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "acquisition manifest"):
            self.prepare("tampered")
        self.assertFalse((self.budget.root / "tampered").exists())

    def test_nonqa_bad_quality_and_ambiguous_spans_are_excluded_explicitly(self):
        summary = {"source_id": "summary", "task_type": "Summary", "source": "CNN/DM", "source_info": "Article.", "prompt": "Summarize"}
        self.sources.append(summary)
        self.responses.append(response("summary-answer", "summary", "train", 1))
        self.responses.append({**response("refusal", "devpositive", "train", 1), "quality": "incorrect_refusal"})
        self.responses.append({**response("truncated", "devpositive", "train", 1), "quality": "truncated"})
        bad = response("ambiguous", "devpositive", "train", 0)
        bad["labels"][0]["start"] = False
        self.responses.append(bad)
        self.write_inputs()
        result = self.prepare()
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["exclusions"]["non_qa_source"], 1)
        self.assertEqual(manifest["exclusions"]["response_quality_not_good"], 2)
        self.assertEqual(manifest["exclusions"]["ambiguous_or_invalid_annotations"], 1)
        selected = {row["annotation_provenance"]["response_id"] for row in read_jsonl(result["human_labels"])}
        self.assertFalse(selected & {"summary-answer", "refusal", "truncated", "ambiguous"})

    def test_insufficient_independent_groups_fail_without_partial_output(self):
        self.responses = [row for row in self.responses if row["split"] != "test" or row["labels"]]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Insufficient independent"):
            self.prepare()
        self.assertFalse((self.budget.root / "audit").exists())


class PublicResponse:
    def __init__(self, url, body, *, declared_bytes=None, status=200, encoding="identity", final_url=None):
        self.url = final_url or url
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body) if declared_bytes is None else declared_bytes),
                        "Content-Encoding": encoding}
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.body.close()

    def geturl(self):
        return self.url

    def read(self, size):
        self.read_sizes.append(size)
        return self.body.read(size)

    read1 = read


class RAGTruthAcquisitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.budget = ResourceBudget(self.root, disk_bytes=1_000_000, reserve_bytes=0)
        self.destination = self.root / "ragtruth"
        self.payloads = {"LICENSE": b"Fixture license.\n", "README.md": b"Fixture documentation.\n",
                         "dataset/response.jsonl": b'{"fixture":"response"}\n',
                         "dataset/source_info.jsonl": b'{"fixture":"source"}\n'}
        self.pins = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                            "git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()}
                     for name, raw in self.payloads.items()}
        self.requests, self.response_overrides = [], {}

        def open_fixture(request, *, timeout):
            self.requests.append((request, timeout))
            relative = request.full_url.split(RAGTRUTH_REVISION + "/", 1)[1]
            headers = {name.lower(): value for name, value in request.header_items()}
            self.assertNotIn("authorization", headers)
            self.assertNotIn("proxy-authorization", headers)
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(headers["accept-encoding"], "identity")
            return self.response_overrides.get(relative) or PublicResponse(request.full_url, self.payloads[relative])

        self.opener = type("FixtureOpener", (), {"open": staticmethod(open_fixture)})()
        for patcher in (patch("magra_benchmark.ragtruth.RAGTRUTH_FILES", self.pins),
                        patch("magra_benchmark.ragtruth.urllib.request.build_opener", return_value=self.opener),
                        patch("urllib.request.urlopen", side_effect=AssertionError("Acquisition tests forbid live network"))):
            patched = patcher.start()
            self.addCleanup(patcher.stop)
            if hasattr(patched, "call_count") and patched is not None and patcher.attribute == "build_opener":
                self.build_opener = patched

    def acquire(self, **kwargs):
        return acquire_ragtruth(self.destination, self.budget, **kwargs)

    def assert_no_partial_acquisition(self):
        self.assertFalse(self.destination.exists())
        self.assertFalse(any(path.name.startswith(".ragtruth-acquire-") or path.name.endswith(".acquire.lock")
                             for path in self.root.iterdir()))

    def test_public_pinned_files_are_verified_and_published_atomically(self):
        result = self.acquire(max_bytes=200_000_000)
        self.assertEqual(result, self.destination.resolve())
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(RAGTRUTH_REVISION, "c103204b9ce28d6bbad859304bf30de72b8ed8fe")
        for relative, expected in self.payloads.items():
            self.assertEqual((result / relative).read_bytes(), expected)
        manifest = json.loads((result / "acquisition.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["downloaded_bytes"], sum(map(len, self.payloads.values())))
        for request, timeout in self.requests:
            self.assertTrue(request.full_url.startswith("https://raw.githubusercontent.com/ParticleMedia/RAGTruth/" + RAGTRUTH_REVISION + "/"))
            self.assertGreater(timeout, 0)
        handlers = self.build_opener.call_args.args
        self.assertTrue(any(isinstance(handler, urllib.request.ProxyHandler) and not handler.proxies for handler in handlers))
        self.assertFalse(any(path.name.startswith(".ragtruth-acquire-") for path in self.root.iterdir()))

    def test_existing_complete_legacy_manifest_is_revalidated_without_network(self):
        result = self.acquire()
        manifest_path = result / "acquisition.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("status")                                                              
        manifest_path.write_text(json.dumps(manifest))
        self.requests.clear()
        self.build_opener.reset_mock()
        self.assertEqual(self.acquire(), result)
        self.assertEqual(self.requests, [])
        self.build_opener.assert_not_called()

    def test_tampered_existing_bytes_or_manifest_are_never_overwritten(self):
        result = self.acquire()
        self.requests.clear()
        readme = result / "README.md"
        original = readme.read_bytes()
        readme.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.acquire()
        self.assertEqual(readme.read_bytes(), b"X" + original[1:])
        readme.write_bytes(original)
        manifest_path = result / "acquisition.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["revision"] = "main"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.acquire()
        self.assertEqual(self.requests, [])

    def test_partial_existing_target_is_preserved_and_rejected(self):
        self.destination.mkdir()
        marker = self.destination / "unfinished"
        marker.write_text("operator-owned partial acquisition")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.acquire()
        self.assertTrue(marker.exists())
        self.assertEqual(self.requests, [])

    def test_combined_byte_bound_and_disk_preflight_fail_before_requests(self):
        with self.assertRaisesRegex(ValueError, "combined byte budget"):
            self.acquire(max_bytes=sum(map(len, self.payloads.values())) - 1)
        self.assertEqual(self.requests, [])
        with patch.object(self.budget, "check", side_effect=BudgetExceeded("fixture full disk")) as guard:
            with self.assertRaises(BudgetExceeded):
                self.acquire()
        self.assertGreaterEqual(guard.call_args.args[0], 2 * sum(map(len, self.payloads.values())))
        self.assertEqual(self.requests, [])
        self.assert_no_partial_acquisition()

    def test_corrupt_or_truncated_download_never_publishes_partial_target(self):
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/LICENSE"
        original = self.payloads["LICENSE"]
        for raw in (b"X" + original[1:], original[:-1], original + b"X"):
            with self.subTest(kind="invalid-publisher-bytes"):
                self.requests.clear()
                self.response_overrides["LICENSE"] = PublicResponse(url, raw, declared_bytes=len(original))
                with self.assertRaises(ValueError):
                    self.acquire()
                self.assertEqual(len(self.requests), 1)
                self.assert_no_partial_acquisition()

    def test_wrong_final_url_encoded_content_status_and_declared_length_are_rejected(self):
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/LICENSE"
        original = self.payloads["LICENSE"]
        options = ({"final_url": "https://example.org/LICENSE"}, {"encoding": "gzip"},
                   {"status": 206}, {"declared_bytes": len(original) + 1})
        for override in options:
            with self.subTest(override=override):
                fake = PublicResponse(url, original, **override)
                self.response_overrides["LICENSE"] = fake
                with self.assertRaises(ValueError):
                    self.acquire()
                self.assertEqual(fake.read_sizes, [])
                self.assert_no_partial_acquisition()

    def test_redirects_cannot_escape_host_revision_or_file_pin(self):
        handler = _RAGTruthRedirect()
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/LICENSE"
        request = urllib.request.Request(url)
        destinations = ("http://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/LICENSE",
                        "https://example.org/LICENSE",
                        "https://raw.githubusercontent.com/Other/Repo/main/LICENSE",
                        url.replace(RAGTRUTH_REVISION, "main"), url.replace("LICENSE", "README.md"))
        for destination in destinations:
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "redirect", {}, destination)
        self.assertEqual(self.requests, [])

    def test_total_deadline_expires_before_body_read_and_cleans_staging(self):
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/LICENSE"
        response = PublicResponse(url, self.payloads["LICENSE"])
        self.response_overrides["LICENSE"] = response
        with patch("magra_benchmark.ragtruth.time.monotonic", side_effect=[0, 0, 2]):
            with self.assertRaises(TimeoutError):
                self.acquire(deadline_seconds=1)
        self.assertEqual(response.read_sizes, [])
        self.assert_no_partial_acquisition()


if __name__ == "__main__":
    unittest.main()
