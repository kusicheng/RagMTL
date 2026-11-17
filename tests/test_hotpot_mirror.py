import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from magra_benchmark.datasets import iter_hotpot, iter_hotpot_documents
from magra_benchmark.hotpot_mirror import (
    MIRROR_DATASETS, MIRROR_REVISION, acquire_hotpot_mirror,
    iter_mirror_rows, normalize_hotpot_row,
)
from magra_rag.resources import ResourceBudget


def row(identifier="q1"):
    return {
        "id": identifier, "question": "Who wrote the caf\u00e9 record?", "answer": "Ada",
        "type": "bridge", "level": "hard",
        "context": {"title": ["Caf\u00e9", "Ada"], "sentences": [["Ada wrote the record.", "Second sentence."], ["Ada is a writer."]]},
        "supporting_facts": {"title": ["Caf\u00e9", "Ada"], "sent_id": [0, 0]},
    }


class NormalizeMirrorTests(unittest.TestCase):
    def test_original_schema_content_and_annotation_indices_are_preserved(self):
        value = row()
        original = copy.deepcopy(value)
        result = normalize_hotpot_row(value)
        self.assertEqual(value, original)
        self.assertEqual(result["_id"], "q1")
        self.assertEqual(result["context"], [["Caf\u00e9", ["Ada wrote the record.", "Second sentence."]], ["Ada", ["Ada is a writer."]]])
        self.assertEqual(result["supporting_facts"], [["Caf\u00e9", 0], ["Ada", 0]])
        self.assertEqual(result["answer"], "Ada")

    def test_missing_answers_are_never_fabricated(self):
        value = row()
        del value["answer"]
        result = normalize_hotpot_row(value)
        self.assertNotIn("answer", result)

    def test_mismatched_struct_arrays_invalid_sentences_and_bool_indices_fail(self):
        variants = []
        value = row()
        value["context"]["title"].append("Missing sentences")
        variants.append(value)
        value = row()
        value["context"]["sentences"][0] = "Not a sentence list"
        variants.append(value)
        value = row()
        value["supporting_facts"]["sent_id"][0] = True
        variants.append(value)
        value = row()
        value["supporting_facts"]["sent_id"][0] = -1
        variants.append(value)
        value = row()
        value["_id"] = "Different id"
        variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_hotpot_row(value)

    def test_record_sequence_schema_is_also_supported_without_guessing(self):
        value = row()
        value["context"] = [{"title": "Caf\u00e9", "sentences": ["Ada wrote the record."]}]
        value["supporting_facts"] = [{"title": "Caf\u00e9", "sent_id": 0}]
        result = normalize_hotpot_row(value)
        self.assertEqual(result["supporting_facts"], [["Caf\u00e9", 0]])


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow is not installed")
class MirrorAcquisitionTests(unittest.TestCase):
    def fixture(self, root, *, duplicate=False, expected_rows=2):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        paths, files = [], []
        for number in range(2):
            path = root / f"fixture-{number}.parquet"
            parquet.write_table(arrow.Table.from_pylist([row("q1" if duplicate else f"q{number}")]), path)
            paths.append(path)
            files.append({"path": f"distractor/{path.name}", "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        spec = {"configuration": "distractor", "mirror_split": "validation", "official_subset": "dev",
                "filename": "hotpot_fixture.jsonl", "rows": expected_rows, "files": files}
        return paths, spec

    def test_complete_local_shards_convert_and_existing_iterators_separate_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, spec = self.fixture(root)
            with mock.patch.dict(MIRROR_DATASETS, {"fixture": spec}):
                output = acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, budget=ResourceBudget(root))
                again = acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, budget=ResourceBudget(root))
            self.assertEqual(output, again)
            manifest = json.loads((output.parent / "acquisition.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["revision"], MIRROR_REVISION)
            self.assertEqual(manifest["origin"], "publisher_huggingface_mirror")
            self.assertFalse(manifest["official_original_bytes"])
            self.assertFalse(manifest["full_wikipedia_corpus"])
            self.assertEqual(manifest["query_count"], 2)
            self.assertEqual(manifest["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            queries = list(iter_hotpot(output, subset="dev"))
            documents = list(iter_hotpot_documents(output))
            self.assertEqual(len(queries), 2)
            self.assertEqual(len(documents), 4)
            self.assertTrue(all(query["answer"] == "Ada" for query in queries))
            self.assertTrue(all("answer" not in document and "supporting_facts" not in document for document in documents))

    def test_download_is_mocked_pinned_and_retains_raw_artifacts(self):
        class Response(io.BytesIO):
            def __init__(self, body):
                super().__init__(body)
                self.headers = {"Content-Length": str(len(body))}
            def geturl(self):
                return "https://cas-bridge.xethub.hf.co/verified-artifact"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, spec = self.fixture(root)
            bodies = {path.name: path.read_bytes() for path in paths}
            calls = []
            def opener(request, timeout):
                calls.append(request.full_url)
                self.assertIn(MIRROR_REVISION, request.full_url)
                return Response(bodies[request.full_url.rsplit("/", 1)[-1]])
            with mock.patch.dict(MIRROR_DATASETS, {"fixture": spec}):
                output = acquire_hotpot_mirror("fixture", root / "sources", opener=opener, budget=ResourceBudget(root))
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list(output.parent.glob("*.parquet"))), 2)

    def test_bad_checksum_duplicate_rows_and_incomplete_split_publish_nothing(self):
        for condition in ("checksum", "duplicate", "incomplete"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths, spec = self.fixture(root, duplicate=condition == "duplicate", expected_rows=3 if condition == "incomplete" else 2)
                if condition == "checksum":
                    paths[0].write_bytes(b"Corrupted")
                with mock.patch.dict(MIRROR_DATASETS, {"fixture": spec}), self.assertRaises(ValueError):
                    acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, budget=ResourceBudget(root))
                self.assertFalse(list((root / "sources").glob("hotpot_hf_*")))
                self.assertFalse(list((root / "sources").glob(".hotpot-mirror-*")))

    def test_download_and_output_limits_apply_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, spec = self.fixture(root)
            opener = mock.Mock()
            with mock.patch.dict(MIRROR_DATASETS, {"fixture": spec}):
                with self.assertRaises(ValueError):
                    acquire_hotpot_mirror("fixture", root / "sources", max_download_bytes=1, opener=opener, budget=ResourceBudget(root))
                opener.assert_not_called()
                with self.assertRaises(ValueError):
                    acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, max_output_bytes=1, budget=ResourceBudget(root))
            self.assertFalse(list((root / "sources").glob("hotpot_hf_*")))

    def test_local_shard_order_and_existing_output_integrity_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, spec = self.fixture(root)
            with mock.patch.dict(MIRROR_DATASETS, {"fixture": spec}):
                with self.assertRaises(ValueError):
                    acquire_hotpot_mirror("fixture", root / "sources", local_files=list(reversed(paths)), budget=ResourceBudget(root))
                output = acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, budget=ResourceBudget(root))
                output.write_text("{}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, budget=ResourceBudget(root))

    def test_conversion_streams_small_batches_and_validates_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = self.fixture(Path(temporary))
            self.assertEqual(len(list(iter_mirror_rows(paths, batch_size=1))), 2)
            with self.assertRaises(ValueError):
                list(iter_mirror_rows(paths, batch_size=257))

    def test_existing_manifest_cannot_masquerade_as_original_hotpot_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, spec = self.fixture(root)
            with mock.patch.dict(MIRROR_DATASETS, {"fixture": spec}):
                output = acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, budget=ResourceBudget(root))
                manifest_path = output.parent / "acquisition.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["official_original_bytes"] = True
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    acquire_hotpot_mirror("fixture", root / "sources", local_files=paths, budget=ResourceBudget(root))


if __name__ == "__main__":
    unittest.main()
