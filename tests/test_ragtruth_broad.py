                                                                                

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from magra_benchmark.datasets import sha256_file
from magra_benchmark.protocol import read_jsonl
from magra_benchmark.ragtruth import (
    REPOSITORY, _broad_groups, _broad_source, prepare_ragtruth_audit,
    prepare_ragtruth_broad_audit,
)
from magra_rag.resources import ResourceBudget


REVISION = "a" * 40


def fixture_source(identifier, task, publisher):
    word = identifier.replace("-", "_")
    if task == "QA":
        info = {"question": word + "?", "passages": "Provided " + word + ".\n\n"}
        prompt = ("Briefly answer the following question:\n" + info["question"] + "\n\n" + info["passages"]
                  + 'In case the passages lack information, reply "Unable to answer based on given passages."\noutput:')
    elif task == "Summary":
        info = word + " reports specific unique events.\n"
        prompt = "Summarize the following news within 100 words:\n" + info + "\noutput:"
    else:
        info = {"name": word, "address": word + " Street", "city": word + " City", "state": "CA",
                "categories": word, "hours": {"Monday": word}, "attributes": {"feature": word},
                "business_stars": 3, "review_info": [{"review_stars": 3, "review_date": "2020-01-01",
                                                       "review_text": " ".join(word + f"_detail{i}" for i in range(20))}]}
        prompt = ("Instruction:\nWrite an objective overview from only these facts. Structured data:\n"
                  + str(info) + "\nOverview:")
    return {"source_id": identifier, "task_type": task, "source": publisher, "source_info": info, "prompt": prompt}


def fixture_response(identifier, source_id, split, label):
    answer = "A supported answer." if label else "An unsupported answer."
    start = answer.find("unsupported")
    labels = [] if label else [{"start": start, "end": start + 11, "text": "unsupported",
                               "label_type": "Evident Baseless Info", "implicit_true": True,
                               "due_to_null": False, "meta": "PRIVATE_ANNOTATION_SENTINEL"}]
    return {"id": identifier, "source_id": source_id, "model": "PRIVATE_MODEL_SENTINEL",
            "quality": "good", "split": split, "response": answer, "labels": labels, "temperature": 0.7}


class BroadRAGTruthTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "publisher"
        (self.raw / "dataset").mkdir(parents=True)
        self.budget = ResourceBudget(self.root / "outputs", disk_bytes=100_000_000, reserve_bytes=0)
        self.sources, self.responses = [], []
        for task, publisher, prefix in (("QA", "MARCO", "qa"), ("Summary", "CNN/DM", "cnn"),
                                        ("Summary", "Recent News", "recent"), ("Data2txt", "Yelp", "yelp")):
            for split in ("train", "test"):
                for label in (0, 1):
                    for replica in range(2 if task == "QA" else 1):
                        identifier = f"{prefix}-{split}-{label}-{replica}"
                        self.sources.append(fixture_source(identifier, task, publisher))
                        self.responses.append(fixture_response("r-" + identifier, identifier, split, label))
        self.write_raw()
        self.prior = self.make_prior("prior")
        for target in ("urllib.request.urlopen", "magra_benchmark.ragtruth.urllib.request.build_opener"):
            patcher = patch(target, side_effect=AssertionError("No network is authorized for preparation"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_raw(self):
        files = {}
        for name, rows in (("source_info.jsonl", self.sources), ("response.jsonl", self.responses)):
            path = self.raw / "dataset" / name
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            raw = path.read_bytes()
            files["dataset/" + name] = {"bytes": len(raw), "sha256": sha256_file(path),
                "git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(),
                "url": f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{REVISION}/dataset/{name}"}
        (self.raw / "acquisition.json").write_text(json.dumps({"format": "magra-ragtruth-acquisition-v1",
            "repository": REPOSITORY, "revision": REVISION, "files": files}), encoding="utf-8")

    def make_prior(self, name):
        result = prepare_ragtruth_audit(self.raw, self.budget.root / name, self.budget,
                                       dev_per_class=1, eval_per_class=1)
        return Path(result["directory"])

    def prepare(self, name="broad", **kwargs):
        return prepare_ragtruth_broad_audit(self.raw, self.budget.root / name, self.budget,
            exclude_audit=self.prior, dev_per_class=1, eval_per_class=1, **kwargs)

    def test_four_strata_balanced_fresh_unique_and_original_evidence(self):
        result = self.prepare()
        humans = list(read_jsonl(result["human_labels"]))
        self.assertEqual(len(humans), 16)
        self.assertEqual(len({r["group_id"] for r in humans}), 16)
        prior_sources = {r["annotation_provenance"]["source_id"] for r in read_jsonl(self.prior / "private/human-labels.jsonl")}
        self.assertFalse(prior_sources & {r["annotation_provenance"]["source_id"] for r in humans})
        original = {r["source_id"]: _broad_source(r) for r in self.sources}
        public = {r["query"]["id"]: r for path in result["public_inputs"].values() for r in read_jsonl(path)}
        for row in humans:
            source = original[row["annotation_provenance"]["source_id"]]
            self.assertEqual(public[row["query_id"]]["evidence"][0]["text"], source["evidence"])
            self.assertEqual(public[row["query_id"]]["query"]["question"], source["instruction"])
            self.assertEqual(row["split"], "dev" if row["annotation_provenance"]["official_split"] == "train" else "eval")
            self.assertEqual(row["label"], int(not row["annotation_spans"]))
        for split in result["selected_counts"].values():
            self.assertEqual(len(split), 4)
            self.assertTrue(all(counts == {"0": 1, "1": 1} for counts in split.values()))

    def test_same_summary_instruction_does_not_merge_distinct_articles(self):
        parsed = {r["source_id"]: _broad_source(r) for r in self.sources}
        groups, members = _broad_groups(parsed, 0.85)
        news = [r for r in self.sources if r["task_type"] == "Summary"]
        self.assertEqual(len({groups[r["source_id"]] for r in news}), len(news))
                                                                           
        left, right = news[0], next(r for r in news if r["source"] == "Recent News")
        parsed[right["source_id"]]["evidence"] = parsed[left["source_id"]]["evidence"]
        groups, _ = _broad_groups(parsed, 0.85)
        self.assertEqual(groups[left["source_id"]], groups[right["source_id"]])

    def test_yelp_complete_business_identity_groups_different_review_contexts(self):
        parsed = {r["source_id"]: _broad_source(r) for r in self.sources}
        ids = [i for i, r in parsed.items() if r["task_type"] == "Data2txt"]
        parsed[ids[1]]["business_identity"] = parsed[ids[0]]["business_identity"]
        groups, _ = _broad_groups(parsed, 0.85)
        self.assertEqual(groups[ids[0]], groups[ids[1]])

    def test_nonstring_business_identity_field_keeps_context_only_grouping(self):
        for value in (None, False, ""):
            row = fixture_source("incomplete", "Data2txt", "Yelp")
            row["source_info"]["address"] = value
            row["prompt"] = ("Instruction:\nUse the provided facts. Structured data:\n"
                             + str(row["source_info"]) + "\nOverview:")
            parsed = _broad_source(row)
            self.assertIsNone(parsed["business_identity"])
            self.assertEqual(json.loads(parsed["evidence"])["address"], value)

    def test_cross_split_context_drops_train_never_promotes_it(self):
        train = next(r for r in self.sources if r["source_id"] == "cnn-train-1-0")
        test = next(r for r in self.sources if r["source_id"] == "cnn-test-0-0")
        train.update(source_info=test["source_info"], prompt=test["prompt"])
        self.sources.append(fixture_source("cnn-spare", "Summary", "CNN/DM"))
        self.responses.append(fixture_response("r-spare", "cnn-spare", "train", 1))
        self.write_raw()
        self.prior = self.make_prior("prior-updated")
        result = self.prepare()
        humans = list(read_jsonl(result["human_labels"]))
        self.assertNotIn("cnn-train-1-0", {r["annotation_provenance"]["source_id"] for r in humans})
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["exclusions"]["train_group_touches_official_test"], 1)

    def test_all_members_of_prior_groups_are_excluded(self):
        aliases = []
        for row in self.sources:
            if row["task_type"] == "QA":
                alias = {**row, "source_id": row["source_id"] + "-alias"}
                aliases.append(alias)
                original = next(r for r in self.responses if r["source_id"] == row["source_id"])
                self.responses.append({**original, "id": original["id"] + "-alias", "source_id": alias["source_id"]})
        self.sources.extend(aliases)
        self.write_raw()
        self.prior = self.make_prior("prior-aliases")
        result = self.prepare()
        grouping = json.loads(Path(result["grouping_audit"]).read_text())
        self.assertEqual(grouping["prior_audit"]["prior_group_count"], 4)
        self.assertEqual(len(grouping["prior_audit"]["excluded_source_ids"]), 8)
        selected = {r["annotation_provenance"]["source_id"] for r in read_jsonl(result["human_labels"])}
        self.assertFalse(selected & set(grouping["prior_audit"]["excluded_source_ids"]))

    def test_annotations_and_model_identity_stay_private(self):
        result = self.prepare()
        forbidden = {"label", "labels", "meta", "model", "original_model", "annotation_spans", "quality", "implicit_true", "due_to_null"}

        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & set(value))
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        for path in result["public_inputs"].values():
            raw = Path(path).read_text()
            self.assertNotIn("PRIVATE_ANNOTATION_SENTINEL", raw)
            self.assertNotIn("PRIVATE_MODEL_SENTINEL", raw)
            for row in read_jsonl(path):
                inspect(row)
        private = Path(result["human_labels"]).read_text()
        self.assertIn("PRIVATE_ANNOTATION_SENTINEL", private)
        self.assertIn("PRIVATE_MODEL_SENTINEL", private)

    def test_deterministic_immutable_inputs_and_frozen_proposed_gates(self):
        first, second = self.prepare("first"), self.prepare("second")
        for split in ("dev", "eval"):
            self.assertEqual(Path(first["public_inputs"][split]).read_bytes(), Path(second["public_inputs"][split]).read_bytes())
        with self.assertRaises(FileExistsError):
            self.prepare("first")
        plan = json.loads(Path(first["proposed_plan"]).read_text())
        self.assertEqual(plan["status"], "prepared_not_judged")
        self.assertEqual(plan["proposed_gates"]["aggregate"], {"min_balanced_accuracy": 0.8, "min_coverage": 0.8})
        self.assertEqual(len(plan["proposed_gates"]["per_stratum"]), 4)
        self.assertEqual(plan["confidence_intervals"]["level"], 0.95)
        manifest = json.loads(Path(first["manifest"]).read_text())
        for relative, pin in manifest["files"].items():
            self.assertEqual(sha256_file(Path(first["directory"]) / relative), pin["sha256"])

    def test_missing_original_instruction_and_tampered_prior_fail_closed(self):
        target = next(r for r in self.sources if r["task_type"] == "Summary")
        target["prompt"] = "Invent an instruction"
        self.write_raw()
        with self.assertRaisesRegex(ValueError, "summary"):
            self.prepare()
        self.assertFalse((self.budget.root / "broad").exists())

    def test_prior_public_hash_tampering_is_rejected_without_labels_read(self):
        path = self.prior / "public/dev.inputs.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Prior QA public input hash"):
            self.prepare()
        self.assertFalse((self.budget.root / "broad").exists())

    def test_prior_private_labels_are_never_opened(self):
        original = Path.open
        prior_private = (self.prior / "private").resolve()

        def guarded(path, mode="r", *args, **kwargs):
            if path.resolve().is_relative_to(prior_private) and "r" in mode:
                raise AssertionError("Broad preparation must use prior public groups, not labels")
            return original(path, mode, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            self.assertEqual(self.prepare()["selected_total"], 16)

    def test_full_evidence_over_judge_bound_fails_without_truncating(self):
        row = next(r for r in self.sources if r["task_type"] == "Summary")
        row["source_info"] = "x" * 1_000_001 + "\n"
        row["prompt"] = "Summarize the following news within 100 words:\n" + row["source_info"] + "\noutput:"
        self.write_raw()
        self.prior = self.make_prior("prior-large")
        with self.assertRaisesRegex(ValueError, "1 MB judge bound"):
            self.prepare()
        self.assertFalse((self.budget.root / "broad").exists())

    def test_count_bounds_fail_before_preparation(self):
        for invalid in (False, 0, 101):
            with self.assertRaises(ValueError):
                prepare_ragtruth_broad_audit(self.raw, self.budget.root / "bad", self.budget,
                                            exclude_audit=self.prior, dev_per_class=invalid)


if __name__ == "__main__":
    unittest.main()
