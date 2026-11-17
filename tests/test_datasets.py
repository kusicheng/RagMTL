import bz2
import hashlib
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import zipfile

from magra_benchmark.datasets import (
    BEIR_DATASETS, HOTPOT_DATASETS, download_beir, download_hotpot, hotpot_document_id, iter_beir_documents,
    iter_beir_queries, iter_hotpot, iter_hotpot_documents, iter_hotpot_wikipedia,
    safe_extract_tar, safe_extract_zip,
)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def beir(self):
        directory = self.root / "scifact"
        (directory / "qrels").mkdir(parents=True)
        corpus = [{"_id": "d1", "title": "Study", "text": "Evidence", "answer": "must not index"},
                  {"_id": "d2", "title": "Other", "text": "Distractor"}]
        queries = [{"_id": "q1", "text": "Does evidence exist?"}, {"_id": "q2", "text": "Unjudged question"}]
        for name, rows in (("corpus.jsonl", corpus), ("queries.jsonl", queries)):
            (directory / name).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        (directory / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\n", encoding="utf-8")
        return directory

    def hotpot(self, **updates):
        record = {"_id": "q1", "question": "Where is A?", "answer": "B",
                  "supporting_facts": [["A", 0]], "context": [["A", ["A is B.", " Another sentence."]], ["D", ["Distractor."]]]}
        record.update(updates)
        path = self.root / "hotpot.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        return path

    def test_complete_corpus_does_not_follow_qrels(self):
        docs = list(iter_beir_documents(self.beir()))
        self.assertEqual([row["id"] for row in docs], ["scifact:d1", "scifact:d2"])
        self.assertFalse(any("answer" in row or "answer" in row["metadata"] for row in docs))

    def test_query_grades_namespaces_and_official_split(self):
        queries = list(iter_beir_queries(self.beir()))
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["relevance"], {"scifact:d1": 2, "scifact:d2": 0})
        self.assertEqual(queries[0]["official_split"], "test")

    def test_qrels_missing_question_fails(self):
        directory = self.beir()
        (directory / "queries.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing query"):
            list(iter_beir_queries(directory))

    def test_invalid_qrels_grade_fails(self):
        directory = self.beir()
        (directory / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\tnan\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(iter_beir_queries(directory))

    def test_bad_subset_cannot_traverse(self):
        with self.assertRaises(ValueError):
            list(iter_beir_queries(self.root, "../private"))

    def test_hotpot_labels_and_context_have_separate_iterators(self):
        path = self.hotpot()
        labels = list(iter_hotpot(path))[0]
        docs = list(iter_hotpot_documents(path))
        self.assertEqual(labels["supporting_facts"], [{"doc_id": docs[0]["id"], "sentence_id": 0}])
        self.assertEqual(labels["context_doc_ids"], [doc["id"] for doc in docs])
        self.assertEqual(docs[0]["metadata"]["sentences"], ["A is B.", " Another sentence."])
        self.assertNotIn("context", labels)
        self.assertTrue(all("answer" not in doc for doc in docs))

    def test_context_ids_are_per_question_and_full_corpus_independent(self):
        path = self.hotpot()
        context = list(iter_hotpot(path))[0]
        full = list(iter_hotpot(path, "full_corpus"))[0]
        self.assertNotEqual(full["supporting_facts"], context["supporting_facts"])
        self.assertEqual(full["group_id"], context["group_id"])
        self.assertNotIn("context_doc_ids", full)
        self.assertNotEqual(hotpot_document_id("A", "q1"), hotpot_document_id("A", "q2"))
        with self.assertRaisesRegex(ValueError, "context union"):
            list(iter_hotpot_documents(path, "full_corpus"))

    def test_canonical_title_preserves_publisher_identity(self):
        self.assertNotEqual(hotpot_document_id("Popular Science"), hotpot_document_id("Popular science"))
        self.assertNotEqual(hotpot_document_id("A/B"), hotpot_document_id("A B"))
        self.assertEqual(hotpot_document_id("Caf\u00e9"), hotpot_document_id("Cafe\u0301"))

    def test_hotpot_distinct_case_titles_both_remain_searchable(self):
        path = self.root / "case-titles.json"
        row = {"_id": "case-q", "question": "Which publication?", "answer": "Popular Science",
               "context": [["Popular Science", ["A magazine."]], ["Popular science", ["A concept."]]],
               "supporting_facts": [["Popular Science", 0]]}
        path.write_text(json.dumps([row]), encoding="utf-8")
        documents = list(iter_hotpot_documents(path))
        query = next(iter_hotpot(path))
        self.assertEqual(len({d["id"] for d in documents}), 2)
        self.assertEqual(set(query["context_doc_ids"]), {d["id"] for d in documents})
        self.assertIn(documents[0]["id"], query["relevance"])

    def test_test_set_without_answer_remains_unlabeled(self):
        path = self.hotpot()
        row = json.loads(path.read_text())[0]
        del row["answer"]
        del row["supporting_facts"]
        path.write_text(json.dumps([row]), encoding="utf-8")
        result = list(iter_hotpot(path, subset="test"))[0]
        self.assertNotIn("answer", result)
        self.assertEqual(result["supporting_facts"], [])

    def test_streaming_array_handles_boundaries(self):
        path = self.hotpot(question="word " * 14000)
        records = json.loads(path.read_text()) * 2
        path.write_text(json.dumps(records), encoding="utf-8")
        self.assertEqual(len(list(iter_hotpot(path))), 2)

    def test_malformed_json_is_rejected(self):
        path = self.hotpot()
        for value in ('[{"_id":"a"}', '[] garbage', '[{},]'):
            path.write_text(value, encoding="utf-8")
            with self.assertRaises((ValueError, KeyError)):
                list(iter_hotpot(path))

    def test_official_intro_corpus_streamed_from_bz2(self):
        path = self.root / "wiki_00.bz2"
        row = {"id": 1, "title": "A", "text": ["A is <a href='B'>B</a>.", " Next."], "url": "https://en.wikipedia.org/wiki/A"}
        with bz2.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        doc = list(iter_hotpot_wikipedia(path))[0]
        self.assertEqual(doc["id"], hotpot_document_id("A"))
        self.assertEqual(doc["text"], "A is B. Next.")
        self.assertEqual(doc["metadata"]["sentences"], ["A is B.", " Next."])

    def test_full_article_dump_rejected_due_sentence_alignment(self):
        path = self.root / "wiki.jsonl"
        path.write_text(json.dumps({"title": "A", "text": [["A"], ["Full article"]]}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "introductory"):
            list(iter_hotpot_wikipedia(path))

    def archive(self, members):
        path = self.root / "data.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipped:
            for name, content in members:
                member = zipfile.ZipInfo("placeholder")
                member.filename = name
                zipped.writestr(member, content)
        return path

    def test_safe_zip_extracts_regular_files(self):
        archive = self.archive([("scifact/corpus.jsonl", "data")])
        target = self.root / "safe"
        safe_extract_zip(archive, target)
        self.assertEqual((target / "scifact/corpus.jsonl").read_text(), "data")

    def test_zip_traversal_windows_and_bombs_rejected_before_writes(self):
        for unsafe in ("../escape", "/absolute", "C:/escape", "a\\escape", "a:secret", "CON.txt", "folder./file"):
            archive = self.archive([("okay", "data"), (unsafe, "danger")])
            target = self.root / "unsafe"
            with self.assertRaises(ValueError, msg=unsafe):
                safe_extract_zip(archive, target)
            self.assertFalse(target.exists())
        archive = self.archive([("large", "x" * 1000)])
        with self.assertRaises(ValueError):
            safe_extract_zip(archive, self.root / "budget", max_uncompressed_bytes=999)

    def test_zip_symlink_rejected(self):
        path = self.root / "data.zip"
        member = zipfile.ZipInfo("symlink")
        member.create_system = 3
        member.external_attr = (0o120777 << 16)
        with zipfile.ZipFile(path, "w") as zipped:
            zipped.writestr(member, "../outside")
        with self.assertRaises(ValueError):
            safe_extract_zip(path, self.root / "links")

    def test_download_validates_allowlist_before_network(self):
        with self.assertRaises(ValueError):
            download_beir("https://untrusted.example/data.zip", self.root)

    def test_download_checksum_layout_completeness_and_manifest(self):
        archive = self.archive([
            ("scifact/corpus.jsonl", json.dumps({"_id": "d1", "title": "A", "text": "Evidence"}) + "\n"),
            ("scifact/queries.jsonl", ""),
            ("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\n"),
        ])
        data = archive.read_bytes()

        class Response(io.BytesIO):
            headers = {"Content-Length": str(len(data))}

        class Opener:
            def open(self, request, timeout):
                return Response(data)

        spec = {**BEIR_DATASETS["scifact"], "documents": 1,
                "md5": hashlib.md5(data, usedforsecurity=False).hexdigest()}
        with patch("magra_benchmark.datasets.urllib.request.build_opener", return_value=Opener()), patch.dict(BEIR_DATASETS, {"scifact": spec}):
            target = download_beir("scifact", self.root / "downloads")
            manifest = json.loads((target / "acquisition.json").read_text())
            self.assertTrue(manifest["complete_corpus"])
            self.assertEqual(manifest["corpus_documents"], 1)
            self.assertIn("corpus.jsonl", manifest["files_sha256"])
            self.assertEqual(manifest["archive_sha256"], hashlib.sha256(data).hexdigest())
            with self.assertRaises(FileExistsError):
                download_beir("scifact", self.root / "downloads")

    def test_download_rejects_checksum_mismatch_and_leaves_no_corpus(self):
        class Response(io.BytesIO):
            headers = {"Content-Length": "3"}

        class Opener:
            def open(self, request, timeout):
                return Response(b"bad")

        destination = self.root / "downloads"
        with patch("magra_benchmark.datasets.urllib.request.build_opener", return_value=Opener()):
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                download_beir("scifact", destination)
        self.assertEqual(list(destination.iterdir()), [])

    def test_local_beir_archive_preserves_input_and_verifies_checksum(self):
        archive = self.archive([
            ("scifact/corpus.jsonl", json.dumps({"_id": "d1", "title": "A", "text": "Evidence"}) + "\n"),
            ("scifact/queries.jsonl", ""),
        ])
        data = archive.read_bytes()
        spec = {**BEIR_DATASETS["scifact"], "documents": 1,
                "md5": hashlib.md5(data, usedforsecurity=False).hexdigest()}
        with patch.dict(BEIR_DATASETS, {"scifact": spec}):
            target = download_beir("scifact", self.root / "local", archive_path=archive)
        self.assertTrue(archive.exists())
        manifest = json.loads((target / "acquisition.json").read_text())
        self.assertEqual(manifest["acquisition_method"], "verified_local_archive")

    def tar_archive(self, members):
        archive = self.root / "wiki.tar.bz2"
        with tarfile.open(archive, "w:bz2") as tar:
            for name, data, kind in members:
                member = tarfile.TarInfo(name)
                member.type = kind
                if kind == tarfile.REGTYPE:
                    member.size = len(data)
                    tar.addfile(member, io.BytesIO(data))
                else:
                    member.linkname = "../outside"
                    tar.addfile(member)
        return archive

    def test_tar_safe_streamed_extraction_and_dot_prefix(self):
        archive = self.tar_archive([("./AA/wiki_00.jsonl", b"record", tarfile.REGTYPE)])
        target = self.root / "tar-safe"
        safe_extract_tar(archive, target)
        self.assertEqual((target / "AA/wiki_00.jsonl").read_bytes(), b"record")

    def test_tar_traversal_links_devices_and_budget_rejected(self):
        for index, (name, kind) in enumerate([
            ("../escape", tarfile.REGTYPE), ("C:/escape", tarfile.REGTYPE),
            ("link", tarfile.SYMTYPE), ("link", tarfile.LNKTYPE), ("device", tarfile.CHRTYPE),
        ]):
            archive = self.tar_archive([(name, b"danger", kind)])
            with self.assertRaises(ValueError):
                safe_extract_tar(archive, self.root / f"unsafe-tar-{index}")
        archive = self.tar_archive([("large", b"0123456789", tarfile.REGTYPE)])
        with self.assertRaises(ValueError):
            safe_extract_tar(archive, self.root / "tar-budget", max_uncompressed_bytes=9)

    def test_tar_duplicate_destination_rejected(self):
        archive = self.tar_archive([("AA/file", b"first", tarfile.REGTYPE), ("aa/FILE", b"second", tarfile.REGTYPE)])
        with self.assertRaisesRegex(ValueError, "Duplicate tar"):
            safe_extract_tar(archive, self.root / "tar-duplicates")

    def test_hotpot_question_download_local_input_and_pin(self):
        source = self.hotpot()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        spec = {**HOTPOT_DATASETS["dev_distractor"], "queries": 1}
        with patch.dict(HOTPOT_DATASETS, {"dev_distractor": spec}):
            target = download_hotpot("dev_distractor", self.root / "hotpot-download", archive_path=source, expected_sha256=digest)
        self.assertTrue(source.exists())
        self.assertEqual(len(list(iter_hotpot(target))), 1)
        manifest = json.loads(target.with_suffix(".json.acquisition.json").read_text())
        self.assertFalse(manifest["complete_corpus"])
        self.assertTrue(manifest["sha256_pinned"])
        self.assertFalse(manifest["publisher_checksum_available"])

    def test_hotpot_question_download_rejects_incomplete_and_wrong_pin(self):
        source = self.hotpot()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            download_hotpot("dev_distractor", self.root / "incomplete", archive_path=source)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            download_hotpot("dev_distractor", self.root / "wrong-pin", archive_path=source, expected_sha256="0" * 64)

    def test_hotpot_wiki_download_verified_archive_audits_complete_corpus(self):
        row = {"id": 1, "title": "A", "text": ["A is B."]}
        contents = bz2.compress((json.dumps(row) + "\n").encode())
        archive = self.tar_archive([("abstracts/AA/wiki_00.bz2", contents, tarfile.REGTYPE)])
        data = archive.read_bytes()
        spec = {**HOTPOT_DATASETS["wiki"], "bytes": len(data),
                "md5": hashlib.md5(data, usedforsecurity=False).hexdigest()}
        with patch.dict(HOTPOT_DATASETS, {"wiki": spec}):
            target = download_hotpot("wiki", self.root / "wiki-download", archive_path=archive)
        self.assertTrue(archive.exists())
        manifest = json.loads((target / "acquisition.json").read_text())
        self.assertTrue(manifest["complete_corpus"])
        self.assertEqual(manifest["corpus_documents"], 1)
        self.assertEqual(manifest["archive_md5"], spec["md5"])
        self.assertEqual(len(list(iter_hotpot_wikipedia(target))), 1)

    def test_hotpot_wiki_is_explicit_and_bounded_before_network(self):
        with self.assertRaisesRegex(ValueError, "compressed byte budget"):
            download_hotpot("wiki", self.root, max_bytes=100)
        with self.assertRaises(ValueError):
            download_hotpot("https://other.example/data", self.root)


if __name__ == "__main__":
    unittest.main()
