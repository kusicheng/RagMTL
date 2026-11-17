import json
from pathlib import Path
import tempfile
import unittest

from magra_benchmark.splits import (
    group_queries, make_splits, normalize_question, read_split_bundle,
    validate_splits, verify_split_bundle, write_split_bundle,
)


def query(identifier, question, **extra):
    return {"id": identifier, "question": question, "source": "fixture", "domain": "test", **extra}


class SplitTests(unittest.TestCase):
    def test_normalized_exact_and_unicode(self):
        rows = [query("a", "What is ＳＱＬ?"), query("b", "WHAT is sql")]
        groups = group_queries(rows)
        self.assertEqual(groups["a"], groups["b"])
        self.assertEqual(normalize_question("  ＳＱＬ! "), "sql")

    def test_explicit_paraphrase_group(self):
        groups = group_queries([query("a", "How warm is it?", group_id="temperature"),
                                query("b", "Give current degrees Celsius", group_id="temperature")])
        self.assertEqual(groups["a"], groups["b"])

    def test_near_duplicate_groups_and_transitive_closure(self):
        rows = [query("a", "one two three four five six seven eight nine ten"),
                query("b", "one two three four five six seven eight nine ten eleven"),
                query("c", "two three four five six seven eight nine ten eleven"),
                query("d", "how do migrating birds find their way home")]
        groups = group_queries(rows)
        self.assertEqual(groups["a"], groups["b"])
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["a"], groups["d"])

    def test_jaccard_prefix_filter_matches_brute_force(self):
        import random
        randomizer = random.Random(19)
        rows = [query(str(index), " ".join(sorted(randomizer.sample([f"w{i}" for i in range(15)], randomizer.randint(4, 12))))) for index in range(150)]
        groups = group_queries(rows, similarity_threshold=0.7)
        for left in rows:
            for right in rows:
                a, b = set(left["question"].split()), set(right["question"].split())
                if len(a & b) / len(a | b) >= 0.7:
                    self.assertEqual(groups[left["id"]], groups[right["id"]])

    def test_official_eval_wins_duplicate_train_and_dev(self):
        rows = [query("a", "same text", official_split="train"),
                query("b", "same text", official_split="test"),
                query("c", "other text", official_split="dev")]
        splits = make_splits(rows)
        self.assertEqual({row["id"] for row in splits["eval"]}, {"a", "b"})
        self.assertEqual([row["id"] for row in splits["dev"]], ["c"])
        self.assertEqual(splits["train"], [])

    def test_heldout_source_excludes_entire_group_from_adaptation(self):
        rows = [query("a", "same question", source="arguana", official_split="train"),
                query("b", "same question", source="another", official_split="train")]
        splits = make_splits(rows)
        self.assertEqual(len(splits["eval"]), 2)
        self.assertEqual(splits["train"], [])

    def test_heldout_domain(self):
        splits = make_splits([query("a", "Some question", official_split="train")], heldout_domains=["test"])
        self.assertEqual(len(splits["eval"]), 1)

    def test_official_dev_partition_is_opt_in_and_source_specific(self):
        rows = [query("hotpot", "Where was the inventor born?", source="hotpotqa", official_split="dev"),
                query("other", "Which alloy resists corrosion?", source="another", official_split="dev")]
        unchanged = make_splits(rows)
        self.assertEqual(len(unchanged["dev"]), 2)
        reserved = make_splits(rows, partition_official_dev_sources=["hotpotqa"], official_dev_eval_fraction=1)
        self.assertEqual([row["id"] for row in reserved["eval"]], ["hotpot"])
        self.assertEqual([row["id"] for row in reserved["dev"]], ["other"])
        self.assertEqual(reserved["eval"][0]["official_split"], "dev")

    def test_hotpot_settings_and_train_paraphrases_remain_in_one_partition(self):
        rows = [query("context", "Where was the inventor born?", source="hotpotqa", official_split="dev",
                      setting="provided_context", group_id="hotpotqa:question-one", answer="PRIVATE"),
                query("full", "Where was the inventor born?", source="hotpotqa", official_split="dev",
                      setting="full_corpus", group_id="hotpotqa:question-one", answer="PRIVATE"),
                query("train-paraphrase", "Identify that inventor's birthplace.", official_split="train",
                      group_id="hotpotqa:question-one")]
        reserved = make_splits(rows, partition_official_dev_sources=["hotpotqa"], official_dev_eval_fraction=1)
        self.assertEqual(len(reserved["eval"]), 3)
        self.assertEqual(reserved["train"], [])
        self.assertEqual(reserved["dev"], [])
        self.assertEqual(len({row["group_id"] for row in reserved["eval"]}), 1)

    def test_official_dev_partition_preserves_test_and_heldout_precedence(self):
        rows = [query("dev", "Who discovered this unusual mineral?", source="hotpotqa", official_split="dev"),
                query("test", "Who discovered this unusual mineral?", source="hotpotqa", official_split="test"),
                query("heldout", "How does a legal appeal proceed?", source="heldout", official_split="dev")]
        reserved = make_splits(rows, heldout_sources=["heldout"],
                               partition_official_dev_sources=["hotpotqa", "heldout"], official_dev_eval_fraction=0)
        self.assertEqual({row["id"] for row in reserved["eval"]}, {"dev", "test", "heldout"})
        self.assertEqual(reserved["dev"], [])

    def test_official_dev_partition_is_deterministic_and_retains_labeled_eval(self):
        rows = [query(str(index), f"question number {index}", source="hotpotqa", official_split="dev",
                      answer=f"private answer {index}") for index in range(80)]
        options = {"partition_official_dev_sources": ["hotpotqa"]}
        first = make_splits(rows, **options)
        self.assertEqual(first, make_splits(reversed(rows), **options))
        self.assertTrue(first["dev"])
        self.assertTrue(first["eval"])
        self.assertEqual(first["train"], [])
        self.assertEqual(sum(len(partition) for partition in first.values()), 80)
        self.assertTrue(all(row["answer"] and row["official_split"] == "dev" for row in first["eval"]))

    def test_official_dev_partition_does_not_override_respect_official_false(self):
        rows = [query("one", "An unusual question", source="hotpotqa", official_split="dev")]
        options = {"respect_official": False, "train_fraction": 0, "dev_fraction": 0}
        first = make_splits(rows, **options)
        second = make_splits(rows, **options, partition_official_dev_sources=["hotpotqa"], official_dev_eval_fraction=0)
        self.assertEqual(first, second)

    def test_order_independence_and_seed_control(self):
        rows = [query(str(index), f"question number {index}") for index in range(50)]
        first = make_splits(rows)
        self.assertEqual(first, make_splits(reversed(rows)))
        self.assertNotEqual(first, make_splits(rows, seed="different"))
        self.assertEqual(sum(len(rows) for rows in first.values()), 50)

    def test_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            make_splits([query("a", "question one"), query("a", "question two")])

    def test_manual_split_mixing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "crosses split"):
            validate_splits({"train": [query("a", "Same question")], "dev": [], "eval": [query("b", "Same question")]})

    def test_invalid_parameters(self):
        for values in ({"train_fraction": 0.9, "dev_fraction": 0.2}, {"train_fraction": float("nan")},
                       {"similarity_threshold": 0}, {"official_dev_eval_fraction": float("nan")},
                       {"official_dev_eval_fraction": -0.1}, {"official_dev_eval_fraction": 1.1}):
            with self.assertRaises(ValueError):
                make_splits([], **values)

    def test_bundle_keeps_labels_private_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "frozen"
            record = query("a", "What is the answer?", answer="PRIVATE SECRET", relevance={"d": 1},
                           unknown_annotation="PRIVATE EXTRA", official_split="test")
            manifest = write_split_bundle(directory, make_splits([record]))
            public_text = (directory / "public/eval.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE", public_text)
            self.assertNotIn("relevance", public_text)
            public = list(read_split_bundle(directory, "eval"))[0]
            self.assertNotIn("answer", public)
            full = list(read_split_bundle(directory, "eval", include_labels=True))[0]
            self.assertEqual(full["answer"], "PRIVATE SECRET")
            self.assertEqual(full["unknown_annotation"], "PRIVATE EXTRA")
            verify_split_bundle(directory, expected_manifest_sha256=manifest["manifest_sha256"])
            with self.assertRaises(FileExistsError):
                write_split_bundle(directory, make_splits([record]))

    def test_bundle_modification_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "frozen"
            write_split_bundle(directory, make_splits([query("a", "Question", official_split="test")]))
            with (directory / "public/eval.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                list(read_split_bundle(directory, "eval"))

    def test_pin_detects_manifest_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "frozen"
            original = write_split_bundle(directory, make_splits([]))
            path = directory / "manifest.json"
            data = json.loads(path.read_text())
            data["provenance"]["unexpected"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                verify_split_bundle(directory, expected_manifest_sha256=original["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
