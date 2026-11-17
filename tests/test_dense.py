                                                                                 

from contextlib import closing
import math
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from magra_rag.dense import (CrossEncoderReranker, DenseRetriever, QWEN_DENSE_MODEL,
                             QWEN_RERANKER_MODEL, retrieval_model_identity)
from magra_rag.resources import BudgetExceeded
from magra_rag.storage import DocumentStore


class Vector(list):
    ndim = 1

    def tobytes(self):
        return struct.pack("<" + "f" * len(self), *self)


class Matrix:
    def __init__(self, rows):
        self.rows = rows

    def __matmul__(self, query):
        return [sum(a * b for a, b in zip(row, query)) for row in self.rows]


def fake_numpy():
    def frombuffer(blob, *, dtype):
        if dtype != "<f4":
            raise AssertionError("Vectors must have an explicit portable float32 encoding")
        return Vector(struct.unpack("<" + "f" * (len(blob) // 4), blob))

    def asarray(values, *, dtype):
        if dtype != "<f4":
            raise AssertionError("Vectors must be stored as little-endian float32")
        return Vector(values)

    return SimpleNamespace(asarray=asarray, frombuffer=frombuffer, stack=Matrix,
                           isfinite=lambda values: SimpleNamespace(all=lambda: all(math.isfinite(x) for x in values)))


class Encoder:
    def __init__(self):
        self.calls = []
        self.before_encode = None

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        if self.before_encode is not None:
            self.before_encode()
        return [Vector([1.0, 0.0] if "Strong" in text or text.startswith("Represent this") else [0.0, 1.0])
                for text in texts]


class DenseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = DocumentStore(self.root / "corpus.sqlite")
        self.addCleanup(self.store.close)
        self.store.ingest([
            {"id": "beir:a", "title": "Strong", "text": "A scientific claim.", "metadata": {"source": "scifact"}},
            {"id": "beir:b", "title": "Weak", "text": "Another scientific claim.", "metadata": {"source": "scifact"}},
            {"id": "hotpotqa:context:c", "title": "Context", "text": "A question-specific paragraph.",
             "metadata": {"source": "hotpotqa"}},
            {"id": "hotpotqa:wiki:d", "title": "Wiki", "text": "A full corpus article.",
             "metadata": {"source": "hotpotqa"}},
        ])
        self.encoder = Encoder()
        self.sentence_factory = Mock(return_value=self.encoder)
        self.cross_model = SimpleNamespace(predict=Mock(return_value=[0.1, 0.9]))
        self.cross_factory = Mock(return_value=self.cross_model)
        self.torch = SimpleNamespace(set_num_threads=Mock())
        dependencies = {"numpy": fake_numpy(), "torch": self.torch,
                        "sentence_transformers": SimpleNamespace(SentenceTransformer=self.sentence_factory,
                                                                  CrossEncoder=self.cross_factory)}
        for patcher in (patch.dict(sys.modules, dependencies), patch.dict("os.environ"),
                        patch("magra_rag.dense.os.cpu_count", return_value=8)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def dense(self, **kwargs):
        return DenseRetriever(self.store, revision="fixture-pinned-commit", **kwargs)

    def test_source_build_preserves_full_corpus_and_scoped_search_ignores_unembedded_wiki(self):
        original = self.store.corpus_hash()
        dense = self.dense()
        result = dense.build(sources=["scifact"])
        self.assertEqual(result["embedded_documents"], 2)
        self.assertEqual(result["sources"], ["scifact"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.store.corpus_hash(), original)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 4)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 2)
        self.assertEqual([row["id"] for row in dense.search("question", source="scifact")], ["beir:a", "beir:b"])
        with self.assertRaisesRegex(ValueError, "matching documents"):
            dense.search("question")

    def test_all_filters_define_same_universe_for_completeness_and_vector_scan(self):
        dense = self.dense()
        dense.build(sources="scifact")
        combinations = [
            {"allowed_ids": {"beir:a"}},
            {"id_prefix": "beir:"},
            {"exclude_ids": {"hotpotqa:context:c", "hotpotqa:wiki:d"}},
            {"source": "scifact", "allowed_ids": {"beir:a", "hotpotqa:wiki:d"},
             "exclude_ids": {"beir:b"}, "id_prefix": "beir:"},
        ]
        for filters in combinations:
            with self.subTest(filters=filters):
                results = dense.search("question", **filters)
                self.assertTrue(results)
                self.assertTrue(all(row["id"].startswith("beir:") for row in results))
        self.assertEqual([row["id"] for row in dense.search("question", allowed_ids=iter(["beir:a"]))], ["beir:a"])
        before = len(self.encoder.calls)
        self.assertEqual(dense.search("question", source="hotpotqa", allowed_ids=set()), [])
        self.assertEqual(len(self.encoder.calls), before)
        with self.assertRaises(ValueError):
            dense.search("question", allowed_ids={"beir:a", "hotpotqa:wiki:d"})

    def test_stale_hash_or_wrong_model_blocks_only_matching_candidates(self):
        dense = self.dense()
        dense.build(sources="scifact")
        for field, value in (("content_hash", "stale-hash"), ("model", "different-model@revision")):
            with self.subTest(field=field):
                with self.store.connection:
                    self.store.connection.execute(f"UPDATE vectors SET {field}=? WHERE doc_id=?", (value, "beir:b"))
                with self.assertRaisesRegex(ValueError, "1 matching documents"):
                    dense.search("question", source="scifact")
                self.assertEqual([row["id"] for row in dense.search("question", source="scifact", exclude_ids={"beir:b"})], ["beir:a"])
                repaired = dense.build(sources=["scifact"])
                self.assertEqual(repaired["embedded_documents"], 1)

    def test_keyset_pages_release_select_lock_before_encode_and_other_writer_commit(self):
        dense = self.dense(batch_size=1)
        writes = []

        def concurrent_writer():
            self.assertFalse(self.store.connection.in_transaction)
            with closing(sqlite3.connect(self.store.path, timeout=0.05)) as other, other:
                key = "other-writer-" + str(len(writes))
                other.execute("INSERT INTO metadata VALUES(?,?)", (key, "committed"))
            writes.append(key)

        self.encoder.before_encode = concurrent_writer
        built = dense.build(sources=["scifact"])
        self.assertEqual(built["embedded_documents"], 2)
        self.assertEqual(len(writes), 2)
        self.assertTrue(all(len(texts) == 1 for texts, _ in self.encoder.calls))
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0], 2)
                                                                              
        self.assertEqual(dense.build(sources=["scifact"])["embedded_documents"], 0)
        self.assertEqual(len(writes), 2)

    def test_full_build_resumes_other_sources_without_reencoding_completed_source(self):
        dense = self.dense(batch_size=1)
        dense.build(sources=["scifact"])
        self.encoder.calls.clear()
        result = dense.build()
        self.assertIsNone(result["sources"])
        self.assertEqual(result["embedded_documents"], 2)
        self.assertTrue(all("scientific" not in text for texts, _ in self.encoder.calls for text in texts))
        self.assertEqual(len(dense.search("question", k=4)), 4)

    def test_empty_or_unknown_source_scope_never_silently_builds_entire_corpus(self):
        dense = self.dense()
        for sources in ([], ["typo-source"], [""]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                dense.build(sources=sources)
        self.assertEqual(self.encoder.calls, [])
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0)

    def test_model_loading_is_cpu_thread_bounded_and_cache_is_inside_accounted_root(self):
        dense = self.dense()
        reranker = CrossEncoderReranker(self.store, revision="fixture-rerank-commit")
        self.assertEqual(dense.cpu_threads, 4)
        self.assertEqual(reranker.cpu_threads, 4)
        self.assertEqual(self.torch.set_num_threads.call_count, 2)
        self.torch.set_num_threads.assert_called_with(4)
        for factory in (self.sentence_factory, self.cross_factory):
            kwargs = factory.call_args.kwargs
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["cache_folder"], str(self.store.budget.root / "models"))
            self.assertFalse(kwargs["trust_remote_code"])

    def test_budget_preflight_happens_before_model_loading_for_both_adapters(self):
        with patch.object(self.store.budget, "check", side_effect=BudgetExceeded("fixture resource cap")):
            with self.assertRaises(BudgetExceeded):
                self.dense()
            with self.assertRaises(BudgetExceeded):
                CrossEncoderReranker(self.store)
        self.sentence_factory.assert_not_called()
        self.cross_factory.assert_not_called()

    def test_bad_encoder_output_cannot_silently_zip_truncate_or_write_nonfinite_vectors(self):
        dense = self.dense(batch_size=2)
        for output in ([[1.0, 0.0]], [[1.0, 0.0], [float("nan"), 0.0]]):
            with self.subTest(output=output):
                self.encoder.encode = Mock(return_value=output)
                with self.assertRaises(ValueError):
                    dense.build(sources=["scifact"])
                self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0)

    def test_missing_optional_dependency_fails_without_fallback(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "optional dense dependencies"):
                self.dense()
            with self.assertRaisesRegex(RuntimeError, "optional dense dependencies"):
                CrossEncoderReranker(self.store)

    def test_reranker_preserves_candidates_and_uses_bounded_batch(self):
        reranker = CrossEncoderReranker(self.store)
        candidates = [self.store.get("beir:a"), self.store.get("beir:b")]
        ranked = reranker.rerank("question", candidates)
        self.assertEqual([row["id"] for row in ranked], ["beir:b", "beir:a"])
        self.assertNotIn("rerank_score", candidates[0])
        self.assertEqual(self.cross_model.predict.call_args.kwargs["batch_size"], 8)

    def test_native_qwen_dispatch_uses_its_own_query_protocol_and_sql_identity(self):
        revision = "a" * 40
        model_id = QWEN_DENSE_MODEL + "@" + revision + "#native-fixture"
        self.encoder.cpu_threads, self.encoder.model_id = 4, model_id
        factory = Mock(return_value=self.encoder)
        module = SimpleNamespace(QwenEncoder=factory, qwen_model_identity=lambda *a: model_id)
        with patch.dict(sys.modules, {"magra_rag.qwen": module}):
            dense = DenseRetriever(self.store, model=QWEN_DENSE_MODEL, revision=revision)
            self.assertEqual(dense.batch_size, 1)
            dense.build(sources="scifact")
            self.assertEqual(self.store.connection.execute("SELECT DISTINCT model FROM vectors").fetchone()[0], model_id)
            dense.search("Strong question", source="scifact")
        texts, kwargs = self.encoder.calls[-1]
        self.assertEqual(texts, ["Strong question"])
        self.assertTrue(kwargs["is_query"])
        self.sentence_factory.assert_not_called()
        factory.assert_called_once_with(self.store, QWEN_DENSE_MODEL, revision, batch_size=1)

    def test_qwen_build_cannot_overwrite_bge_and_bge_cannot_overwrite_qwen(self):
        dense = self.dense()
        dense.build(sources="scifact")
        original = self.store.connection.execute("SELECT model,embedding FROM vectors ORDER BY doc_id").fetchall()
        identity = QWEN_DENSE_MODEL + "@" + "a" * 40 + "#native-fixture"
        self.encoder.cpu_threads, self.encoder.model_id = 4, identity
        module = SimpleNamespace(QwenEncoder=Mock(return_value=self.encoder),
                                 qwen_model_identity=lambda *a: identity)
        with patch.dict(sys.modules, {"magra_rag.qwen": module}):
            qwen = DenseRetriever(self.store, model=QWEN_DENSE_MODEL, revision="a" * 40)
            with self.assertRaisesRegex(ValueError, "separate SQL store"):
                qwen.build(sources="scifact")
        self.assertEqual(original, self.store.connection.execute("SELECT model,embedding FROM vectors ORDER BY doc_id").fetchall())
        with self.store.connection:
            self.store.connection.execute("UPDATE vectors SET model=?", (identity,))
        with self.assertRaisesRegex(ValueError, "separate SQL store"):
            dense.build(sources="scifact")

    def test_native_qwen_reranker_dispatch_has_no_crossencoder_fallback(self):
        identity = QWEN_RERANKER_MODEL + "@" + "a" * 40 + "#native-fixture"
        native = Mock(model_id=identity, cpu_threads=4)
        native.rerank.return_value = [{"id": "native", "rerank_score": 2.0}]
        factory = Mock(return_value=native)
        module = SimpleNamespace(QwenReranker=factory, qwen_model_identity=lambda *a: identity)
        with patch.dict(sys.modules, {"magra_rag.qwen": module}):
            reranker = CrossEncoderReranker(self.store, model=QWEN_RERANKER_MODEL, revision="a" * 40)
            self.assertEqual(reranker.rerank("question", [{"id": "native"}]), native.rerank.return_value)
        self.cross_factory.assert_not_called()
        native.rerank.assert_called_once_with("question", [{"id": "native"}])

    def test_wrong_qwen_roles_fail_before_loading(self):
        with self.assertRaisesRegex(ValueError, "reranker cannot"):
            DenseRetriever(self.store, model=QWEN_RERANKER_MODEL, revision="a" * 40)
        with self.assertRaisesRegex(ValueError, "embedding model cannot"):
            CrossEncoderReranker(self.store, model=QWEN_DENSE_MODEL, revision="a" * 40)
        with self.assertRaisesRegex(ValueError, "Only the native"):
            retrieval_model_identity("Qwen/unimplemented-model", "a" * 40)

    def test_index_fingerprint_does_not_encode_and_binds_scoped_sql_vectors(self):
        from magra_rag.index_integrity import effective_query_scope
        dense = self.dense(model="fixture-encoder")
        dense.build(sources="scifact")
        calls = len(self.encoder.calls)
        scope = effective_query_scope({"id": "q", "corpus_source": "scifact"})
        before = dense.index_fingerprint(scopes=[scope])
        self.assertEqual(before["document_count"], 2)
        self.assertEqual(before["dimensions"], 2)
        self.assertEqual(len(self.encoder.calls), calls)
        with self.store.connection:
            self.store.connection.execute("UPDATE vectors SET embedding=? WHERE doc_id=?",
                                          (struct.pack('<ff', 0.5, 0.5), 'beir:a'))
        self.assertNotEqual(dense.index_fingerprint(scopes=[scope])["fingerprint_hash"], before["fingerprint_hash"])
        self.assertEqual(len(self.encoder.calls), calls)


if __name__ == "__main__":
    unittest.main()
