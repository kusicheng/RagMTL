                                                                                  
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from magra_rag.ann import (ExistingDenseQueryEncoder, FAISS_VERSION, FAISS_WHEEL_HASHES,
                      FaissHNSWDenseRetriever, HNSWConfig, PREFIX, ResourceGuard,
                      _hash, _sha, build_hnsw_flat)
from magra_rag.index_integrity import dense_index_fingerprint, effective_query_scope
from magra_rag.storage import DocumentStore

ROOT = Path(__file__).resolve().parents[1]
MODEL = "fixture/encoder@" + "0" * 40


class Flat(list):
    def reshape(self, rows, dimensions):
        if rows * dimensions != len(self):
            raise ValueError("Malformed synthetic shape")
        return Matrix(self[i * dimensions:(i + 1) * dimensions] for i in range(rows))


class Matrix(list):
    def tobytes(self, *, order):
        if order != "C":
            raise AssertionError("Canonical contiguous row order is required")
        return b"".join(struct.pack("<f", value) for row in self for value in row)

    def __matmul__(self, query):
        return [struct.unpack("<f", struct.pack("<f", sum(a * b for a, b in zip(row, query))))[0]
                for row in self]


class FakeNumpy:
    __version__ = "synthetic-double"
    @staticmethod
    def frombuffer(blob, *, dtype):
        if dtype != "<f4":
            raise AssertionError("Portable canonical float32 bytes are required")
        return Flat(value for (value,) in struct.iter_unpack("<f", blob))


class FakeFaiss:
    __version__ = FAISS_VERSION
    METRIC_INNER_PRODUCT = 0

    def __init__(self):
        self.threads, self.reads, self.searches = 1, 0, 0
        self.batch_sizes, self.on_add, self.search_override, self.fail_write = [], None, None, False
        self.IndexFlat = type("IndexFlat", (SimpleNamespace,), {})
        owner = self

        class IndexHNSWFlat:
            def __init__(self, dimensions, m, metric):
                self.d, self.m, self.metric_type, self.is_trained = dimensions, m, metric, True
                self.vectors, self.ntotal = [], 0
                self.storage = owner.IndexFlat(d=dimensions, metric_type=metric, ntotal=0)
                self.hnsw = SimpleNamespace(efConstruction=40, efSearch=16,
                                            nb_neighbors=lambda level: 2 * m if level == 0 else m,
                                            is_similarity=True, search_method=0,
                                            check_relative_distance=True, search_bounded_queue=True)

            def add(self, matrix):
                owner.batch_sizes.append(len(matrix))
                self.vectors.extend(list(row) for row in matrix)
                self.ntotal = self.storage.ntotal = len(self.vectors)
                if owner.on_add:
                    owner.on_add()

            def search(self, matrix, k):
                owner.searches += 1
                if owner.search_override:
                    return owner.search_override
                order = sorted(((math.fsum(a * b for a, b in zip(matrix[0], vector)), ordinal)
                                for ordinal, vector in enumerate(self.vectors)), key=lambda item: (-item[0], item[1]))[:k]
                return [[item[0] for item in order]], [[item[1] for item in order]]

        self.IndexHNSWFlat = IndexHNSWFlat

    def omp_set_num_threads(self, count):
        self.threads = count

    def omp_get_max_threads(self):
        return self.threads

    @staticmethod
    def RandomGenerator(seed):
        return SimpleNamespace(seed=seed)

    @staticmethod
    def downcast_index(index):
        return index

    def write_index(self, index, filename):
        if self.fail_write:
            raise OSError("synthetic write failure")
        Path(filename).write_text(json.dumps({"dimensions": index.d, "m": index.m, "metric": index.metric_type,
                                             "ef_construction": index.hnsw.efConstruction,
                                             "ef_search": index.hnsw.efSearch, "vectors": index.vectors}), encoding="utf-8")

    def read_index(self, filename):
        self.reads += 1
        record = json.loads(Path(filename).read_text(encoding="utf-8"))
        index = self.IndexHNSWFlat(record["dimensions"], record["m"], record["metric"])
        index.hnsw.efConstruction, index.hnsw.efSearch = record["ef_construction"], record["ef_search"]
        index.add(record["vectors"])
        return index


class Guard:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("synthetic resource cap")
        return {"synthetic_guard": True}


class TinyEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[1.0, 0.0] for _ in texts]


class Exact:
    def __init__(self, store):
        self.store, self.model_id, self.is_qwen = store, MODEL, False
        self.encoder, self.calls = TinyEncoder(), []

    def search(self, question, *, k, allowed_ids, exclude_ids, source, id_prefix):
        self.calls.append({"allowed_ids": allowed_ids, "exclude_ids": exclude_ids,
                           "source": source, "id_prefix": id_prefix})
        result = []
        query = ExistingDenseQueryEncoder(self)(question)
        with closing(self.store.connection.execute(
                "SELECT d.id,d.source,v.embedding FROM documents d JOIN vectors v ON v.doc_id=d.id")) as cursor:
            rows = cursor.fetchall()
        for doc_id, doc_source, blob in rows:
            if source is not None and doc_source != source or id_prefix is not None and not doc_id.startswith(id_prefix):
                continue
            if allowed_ids is not None and doc_id not in allowed_ids or doc_id in exclude_ids:
                continue
            score = math.fsum(a * b for a, (b,) in zip(query, struct.iter_unpack("<f", blob)))
            result.append(dict(self.store.get(doc_id), score=score, retriever="exact_fixture"))
        return sorted(result, key=lambda item: (-item["score"], item["id"]))[:k]


class Fixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="magra-ann-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assertFalse(self.root.is_relative_to(ROOT / "data"))
        self.store = DocumentStore(self.root / "corpus.sqlite")
        self.addCleanup(self.store.close)
        self.store.ingest([{"id": doc_id, "title": doc_id, "text": "public synthetic fixture",
                            "metadata": {"source": source}} for doc_id, source in
                           [("a:1", "a"), ("a:2", "a"), ("a:3", "a"), ("b:4", "b")]])
        for doc_id, values in [("a:1", (1, 0)), ("a:2", (.8, .6)), ("a:3", (0, 1)), ("b:4", (-1, 0))]:
            content = self.store.connection.execute("SELECT content_hash FROM documents WHERE id=?", (doc_id,)).fetchone()[0]
            with self.store.connection:
                self.store.connection.execute("INSERT INTO vectors VALUES(?,?,?,?,?)",
                                              (doc_id, MODEL, content, 2, struct.pack("<ff", *values)))
        self.corpus_before = self.store.corpus_hash()
        self.exact, self.faiss, self.np, self.guard = Exact(self.store), FakeFaiss(), FakeNumpy(), Guard()
        self.exact.np = self.np
        native = self.root / "fixture-native.bin"
        native.write_bytes(b"synthetic native-runtime identity; not a native ABI test")
        self.runtime = {"version": FAISS_VERSION, "wheel_sha256": sorted(FAISS_WHEEL_HASHES)[0],
                        "binary_files": [{"name": "fixture/native.bin", "path": str(native), "sha256": _sha(native)}]}
        self.config = HNSWConfig(2, source="a", id_prefix="a:", m=2, ef_construction=4,
                                 ef_search=8, batch_rows=2, batch_bytes=16, max_index_bytes=8_388_608)
        self.output = self.root / "ann"

    def build(self, **kwargs):
        return build_hnsw_flat(self.store, kwargs.pop("output", self.output), model_id=MODEL,
                               query_protocol_id=ExistingDenseQueryEncoder(self.exact).protocol_id,
                               config=kwargs.pop("config", self.config), faiss_module=self.faiss,
                               numpy_module=self.np, runtime_pin=self.runtime,
                               guard=kwargs.pop("guard", self.guard), **kwargs)

    def load(self, manifest=None, **kwargs):
        manifest = manifest or self.manifest
        return FaissHNSWDenseRetriever(self.exact, self.output,
                                       expected_manifest_hash=manifest["artifact_hash"],
                                       faiss_module=self.faiss, numpy_module=self.np,
                                       runtime_pin=self.runtime, guard=self.guard, **kwargs)

    def publish_modified_manifest(self, manifest):
        manifest["artifact_hash"] = _hash({key: value for key, value in manifest.items() if key != "artifact_hash"})
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


class AdapterTests(Fixture, unittest.TestCase):
    def test_bounded_build_save_reload_and_exact_dot_oracle(self):
        self.manifest = self.build()
        self.assertEqual(self.faiss.batch_sizes, [2, 1])
        self.assertEqual(self.manifest["raw_sql"]["document_count"], 3)
        self.assertEqual(self.store.corpus_hash(), self.corpus_before)
        adapter = self.load()
        for _ in range(2):
            result = adapter.search("synthetic question", k=3, source="a", id_prefix="a:")
            oracle = self.exact.search("synthetic question", k=3, source="a", id_prefix="a:", allowed_ids=None, exclude_ids=[])
            self.assertEqual([(d["id"], d["score"]) for d in result], [(d["id"], d["score"]) for d in oracle])
        self.assertEqual(self.faiss.reads, 1)
        self.assertEqual(self.store.corpus_hash(), self.corpus_before)
        self.assertTrue(any(call.get("disk_reserve", 0) > 0 and call.get("ram_reserve", 0) > 0 for call in self.guard.calls))

    def test_canonical_store_can_be_query_only_through_build_and_query(self):
        self.store.connection.execute("PRAGMA query_only=ON")
        self.manifest = self.build()
        self.assertTrue(self.load().search("fixture", source="a", id_prefix="a:"))

    def test_fixed_scope_mismatch_refuses_search_and_fingerprint(self):
        self.manifest = self.build()
        adapter = self.load()
        for source, prefix in [(None, "a:"), ("b", "a:"), ("a", None), ("a", "a:1")]:
            with self.subTest(source=source, prefix=prefix), self.assertRaisesRegex(ValueError, "exactly match"):
                adapter.search("fixture", source=source, id_prefix=prefix)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            adapter.index_fingerprint(scopes=[{"query_id": "q", "source": "b"}])

    def test_allowlist_and_exclusion_are_exact_fallback_including_arguana_self(self):
        self.manifest = self.build()
        adapter = self.load()
        result = adapter.search("fixture", source="a", id_prefix="a:", allowed_ids=["a:3", "b:4"])
        self.assertEqual([d["id"] for d in result], ["a:3"])
        scope = effective_query_scope({"id": "a:1", "source": "arguana", "corpus_source": "a"})
        result = adapter.search("fixture", source=scope["source"], id_prefix="a:", exclude_ids=scope["exclude_ids"])
        self.assertEqual([d["id"] for d in result], ["a:2", "a:3"])
        self.assertEqual(self.faiss.searches, 0)
        self.assertTrue(all(d["retriever"] == "dense_exact_filter_fallback" for d in result))
        before = len(self.exact.encoder.calls)
        self.assertEqual(adapter.search("fixture", source="a", id_prefix="a:", allowed_ids=[]), [])
        self.assertEqual(len(self.exact.encoder.calls), before)

    def test_manifest_binary_and_mapping_checksums_precede_native_load(self):
        for filename in ["manifest.json", "index.faiss", "ordinals.sqlite"]:
            with self.subTest(filename=filename):
                output = self.root / filename.replace(".", "-")
                manifest = self.build(output=output)
                path = output / filename
                with path.open("ab") as stream:
                    stream.write(b"corruption")
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    FaissHNSWDenseRetriever(self.exact, output, expected_manifest_hash=manifest["artifact_hash"],
                                            faiss_module=self.faiss, numpy_module=self.np,
                                            runtime_pin=self.runtime, guard=self.guard)
                self.assertEqual(self.faiss.reads, 0)

    def test_manifest_external_pin_rejects_resigned_change(self):
        self.manifest = self.build()
        modified = json.loads(json.dumps(self.manifest));modified["config"]["ef_search"] += 1
        self.publish_modified_manifest(modified)
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            self.load(self.manifest)

    def test_external_and_path_escape_artifacts_are_refused(self):
        self.manifest = self.build()
        modified = json.loads(json.dumps(self.manifest));modified["files"]["index"]["path"] = "../index.faiss"
        self.publish_modified_manifest(modified)
        with self.assertRaisesRegex(ValueError, "fixed local filenames"):
            self.load(modified)
        with self.assertRaises(ValueError):
            build_hnsw_flat(self.store, self.root.parent / "outside-ann", model_id=MODEL,
                            query_protocol_id=ExistingDenseQueryEncoder(self.exact).protocol_id,
                            config=self.config, faiss_module=self.faiss, numpy_module=self.np,
                            runtime_pin=self.runtime, guard=self.guard)

    def test_partial_build_is_ineligible_and_output_cannot_be_reused(self):
        self.faiss.fail_write = True
        with self.assertRaises(OSError):
            self.build()
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(json.loads((self.output / "failure.json").read_text())["status"], "partial_ineligible")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.store.corpus_hash(), self.corpus_before)

    def test_guard_failure_is_before_index_allocation_or_publication(self):
        with self.assertRaisesRegex(RuntimeError, "resource cap"):
            self.build(guard=Guard(fail=True))
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(self.faiss.batch_sizes, [])

    def test_mutation_during_build_refuses_complete_manifest(self):
        def change():
            self.faiss.on_add = None
            with self.store.connection:
                self.store.connection.execute("UPDATE vectors SET embedding=? WHERE doc_id='a:3'", (struct.pack("<ff", -1, 0),))
        self.faiss.on_add = change
        with self.assertRaisesRegex(ValueError, "changed during|streamed SQL"):
            self.build()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_transient_vector_change_restored_before_final_fingerprint_is_ineligible(self):
        initial = dense_index_fingerprint(self.store, MODEL, scopes=[self.config.scope()], expected_dimensions=2)
        calls = []
        def change_and_restore():
            calls.append(1)
            values = (-1, 0) if len(calls) == 1 else (0, 1)
            with self.store.connection:
                self.store.connection.execute("UPDATE vectors SET embedding=? WHERE doc_id='a:3'",
                                              (struct.pack("<ff", *values),))
        self.faiss.on_add = change_and_restore
        with self.assertRaisesRegex(ValueError, "Actual streamed SQL"):
            self.build()
        self.assertEqual(len(calls), 2)
        self.assertEqual(dense_index_fingerprint(self.store, MODEL, scopes=[self.config.scope()], expected_dimensions=2), initial)
        self.assertFalse((self.output / "manifest.json").exists())

    def test_accumulated_native_growth_headroom_is_rechecked_before_each_add(self):
        state = {"headroom": 1000}
        def dynamic_guard(**kwargs):
            if kwargs.get("ram_reserve", 0) > state["headroom"]:
                raise RuntimeError("dynamic native-growth headroom exhausted")
            if self.faiss.batch_sizes == [2]:
                state["headroom"] = 500
            return {"synthetic_headroom": state["headroom"]}
        with self.assertRaisesRegex(RuntimeError, "native-growth headroom"):
            self.build(guard=dynamic_guard)
        self.assertEqual(self.faiss.batch_sizes, [2])
        self.assertFalse((self.output / "manifest.json").exists())

    def test_missing_stale_nonfinite_and_nonunit_vectors_refuse_build(self):
        cases = [("DELETE FROM vectors WHERE doc_id='a:1'", ()),
                 ("UPDATE vectors SET content_hash='stale' WHERE doc_id='a:1'", ()),
                 ("UPDATE vectors SET embedding=? WHERE doc_id='a:1'", (struct.pack("<ff", math.nan, 0),)),
                 ("UPDATE vectors SET embedding=? WHERE doc_id='a:1'", (struct.pack("<ff", 2, 0),))]
        original = tuple(self.store.connection.execute("SELECT * FROM vectors WHERE doc_id='a:1'").fetchone())
        for number, (sql, params) in enumerate(cases):
            with self.subTest(number=number):
                with self.store.connection:
                    self.store.connection.execute(sql, params)
                with self.assertRaises(ValueError):
                    self.build(output=self.root / f"bad-{number}")
                with self.store.connection:
                    self.store.connection.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?)", original)

    def test_relevant_mutation_after_load_fails_unrelated_source_changes_do_not(self):
        self.manifest = self.build();adapter = self.load()
        scopes = [{"query_id": "q", "source": "a", "id_prefix": "a:", "exclude_ids": []}]
        before = adapter.index_fingerprint(scopes=scopes)
        with self.store.connection:
            self.store.connection.execute("UPDATE vectors SET embedding=? WHERE doc_id='b:4'", (struct.pack("<ff", 0, 1),))
        self.assertEqual(adapter.index_fingerprint(scopes=scopes), before)
        with self.store.connection:
            self.store.connection.execute("UPDATE vectors SET embedding=? WHERE doc_id='a:1'", (struct.pack("<ff", 0, 1),))
        with self.assertRaisesRegex(ValueError, "stale"):
            adapter.search("fixture", source="a", id_prefix="a:")

    def test_same_connection_transaction_and_external_commit_fail_closed(self):
        self.manifest = self.build();adapter = self.load()
        self.store.connection.execute("BEGIN")
        with self.assertRaisesRegex(ValueError, "Commit or roll back"):
            adapter.search("fixture", source="a", id_prefix="a:")
        self.store.connection.rollback()
        with closing(sqlite3.connect(self.store.path)) as other:
            with other:
                other.execute("UPDATE vectors SET embedding=? WHERE doc_id='a:2'", (struct.pack("<ff", 0, 1),))
        with self.assertRaisesRegex(ValueError, "stale"):
            adapter.search("fixture", source="a", id_prefix="a:")

    def test_file_change_after_load_refuses_queries(self):
        self.manifest = self.build();adapter = self.load()
        with (self.output / "index.faiss").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after load"):
            adapter.search("fixture", source="a", id_prefix="a:")

    def test_in_memory_vector_mutation_is_detected_at_fingerprint_boundary(self):
        self.manifest = self.build();adapter = self.load()
        scopes = [{"query_id": "q", "source": "a", "id_prefix": "a:"}]
        before = adapter.index_fingerprint(scopes=scopes)
        adapter.index.vectors[0], adapter.index.vectors[2] = adapter.index.vectors[2], adapter.index.vectors[0]
        with self.assertRaisesRegex(ValueError, "in-memory ANN"):
            adapter.index_fingerprint(scopes=scopes)
        self.assertTrue(before["runtime_index_sha256"])
        self.assertEqual(list((self.root / "tmp").glob("ann-fingerprint-*.faiss")), [])
        self.assertEqual(self.store.corpus_hash(), self.corpus_before)

    def test_runtime_search_parameters_join_fingerprint_and_mutation_is_rejected(self):
        self.manifest = self.build();first = self.load();second = self.load(ef_search=16)
        scopes = [{"query_id": "q", "source": "a", "id_prefix": "a:"}]
        self.assertNotEqual(first.index_fingerprint(scopes=scopes)["fingerprint_hash"],
                            second.index_fingerprint(scopes=scopes)["fingerprint_hash"])
        second.index.hnsw.efSearch = 99
        with self.assertRaisesRegex(ValueError, "search parameters"):
            second.search("fixture", source="a", id_prefix="a:")

    def test_hidden_search_option_changes_are_rejected_and_seed_is_pinned(self):
        self.manifest = self.build();adapter = self.load()
        self.assertEqual(self.manifest["config"]["seed"], 12345)
        self.assertEqual(self.manifest["score_protocol"], {"arithmetic": "numpy_float32_matmul",
                                                          "numpy_version": "synthetic-double"})
        adapter.index.hnsw.search_bounded_queue = False
        with self.assertRaisesRegex(ValueError, "search parameters"):
            adapter.search("fixture", source="a", id_prefix="a:")

    def test_raw_payload_byte_limit_refuses_before_allocating_index(self):
        with self.assertRaisesRegex(ValueError, "FP32 payload"):
            self.build(config=replace(self.config, max_index_bytes=8))
        self.assertEqual(self.faiss.batch_sizes, [])
        self.assertFalse((self.output / "manifest.json").exists())

    def test_quantized_or_unknown_storage_is_not_treated_as_flat(self):
        self.manifest = self.build();adapter = self.load()
        adapter.index.storage = SimpleNamespace(d=2, metric_type=0, ntotal=3)
        with self.assertRaisesRegex(ValueError, "unquantized"):
            adapter.search("fixture", source="a", id_prefix="a:")

    def test_duplicate_negative_and_malformed_ann_results_fail(self):
        self.manifest = self.build();adapter = self.load()
        for value in [([[1, .8]], [[0, 0]]), ([[1, .8]], [[0, -1]]), ([[math.nan]], [[0]]), ([], [])]:
            with self.subTest(value=value):
                self.faiss.search_override = value
                with self.assertRaises(ValueError):
                    adapter.search("fixture", k=2, source="a", id_prefix="a:")

    def test_wrong_query_vector_and_native_runtime_pin_refuse(self):
        self.manifest = self.build();adapter = self.load()
        self.exact.encoder.encode = lambda *args, **kwargs: [[2.0, 0.0]]
        with self.assertRaisesRegex(ValueError, "unit canonical"):
            adapter.search("fixture", source="a", id_prefix="a:")
        self.runtime["binary_files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "native binary"):
            self.load()

    def test_existing_dense_cpu_query_formatting_preserves_bge_and_qwen(self):
        encoder = ExistingDenseQueryEncoder(self.exact)
        encoder("fixture")
        self.assertEqual(self.exact.encoder.calls[-1][0], [PREFIX + "fixture"])
        self.assertNotIn("is_query", self.exact.encoder.calls[-1][1])
        self.exact.is_qwen = True
        qwen = ExistingDenseQueryEncoder(self.exact);qwen("fixture")
        self.assertEqual(self.exact.encoder.calls[-1], (["fixture"], {"normalize_embeddings": True,
                                                                     "show_progress_bar": False, "is_query": True}))
        self.assertNotEqual(encoder.protocol_id, qwen.protocol_id)

    def test_global_guard_measures_host_and_refuses_hard_cap(self):
        guard = ResourceGuard(self.store.budget)
        with patch("magra_rag.ann.host_memory", return_value={"total_bytes": 40_000_000_000,
                                                        "available_bytes": 8_000_000_000,
                                                        "used_bytes": 32_000_000_000}):
            with self.assertRaisesRegex(RuntimeError, "host cap"):
                guard()

    def test_config_and_pinned_revision_validation(self):
        for change in [{"threads": 8}, {"dimensions": True}, {"batch_bytes": 4}, {"source": ""},
                       {"unit_tolerance": math.nan}, {"ef_construction": 1}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change)
        with self.assertRaisesRegex(ValueError, "full pinned"):
            build_hnsw_flat(self.store, self.output, model_id="fixture@main", query_protocol_id="0" * 64,
                            config=self.config, faiss_module=self.faiss, numpy_module=self.np,
                            runtime_pin=self.runtime, guard=self.guard)


class NativeTests(Fixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from magra_rag.ann_catalog import LoadedRuntime, read_runtime
        runtime_path=os.environ.get("MAGRA_ANN_RUNTIME_CONFIG")
        if not runtime_path:self.skipTest("Explicit local MAGRA_ANN_RUNTIME_CONFIG is required")
        configuration=read_runtime(runtime_path)
        loaded=LoadedRuntime(configuration,self.store.budget)
        self.addCleanup(loaded.close)
        self.faiss,self.np=loaded.faiss,loaded.np;self.exact.np=self.np
        self.runtime=configuration["pin"]

    def test_native_stream_save_reload_exact_synthetic_dot_and_actual_host_guard(self):
        self.guard = ResourceGuard(self.store.budget)
        self.manifest = self.build();adapter = self.load()
        scopes = [{"query_id": "q", "source": "a", "id_prefix": "a:"}]
        before = adapter.index_fingerprint(scopes=scopes)
        result = adapter.search("synthetic fixture", k=3, source="a", id_prefix="a:")
        oracle = self.exact.search("synthetic fixture", k=3, source="a", id_prefix="a:", allowed_ids=None, exclude_ids=[])
        self.assertEqual([(item["id"], item["score"]) for item in result], [(item["id"], item["score"]) for item in oracle])
        self.assertEqual(self.store.corpus_hash(), self.corpus_before)
        self.assertIn("host_ram", self.manifest["build_observation"]["initial"])
        self.assertEqual(adapter.index_fingerprint(scopes=scopes), before)
        changed_search = self.load(ef_search=16)
        changed_before = changed_search.index_fingerprint(scopes=scopes)
        self.assertNotEqual(changed_before["fingerprint_hash"], before["fingerprint_hash"])
        changed_search.search("synthetic fixture", k=3, source="a", id_prefix="a:")
        self.assertEqual(changed_search.index_fingerprint(scopes=scopes), changed_before)

    def test_native_wrong_metric_or_dimensions_refuse_after_checksum_validation(self):
        self.manifest = self.build()
        for dimensions, metric in [(2, self.faiss.METRIC_L2), (3, self.faiss.METRIC_INNER_PRODUCT)]:
            with self.subTest(dimensions=dimensions, metric=metric):
                bad = self.faiss.IndexHNSWFlat(dimensions, self.config.m, metric)
                bad.hnsw.efConstruction, bad.hnsw.efSearch = self.config.ef_construction, self.config.ef_search
                matrix = self.np.zeros((3, dimensions), dtype="float32");matrix[:, 0] = 1
                bad.add(matrix)
                self.faiss.write_index(bad, str(self.output / "index.faiss"))
                changed = json.loads(json.dumps(self.manifest))
                changed["files"]["index"].update(bytes=(self.output / "index.faiss").stat().st_size,
                                                 sha256=_sha(self.output / "index.faiss"))
                self.publish_modified_manifest(changed)
                with self.assertRaisesRegex(ValueError, "structure, metric"):
                    self.load(changed)

    def test_native_scoring_exactly_matches_existing_dense_numpy_fp32_arithmetic(self):
        from magra_rag.dense import DenseRetriever
        query = self.np.asarray([.8, .6], dtype="<f4")
        exact = DenseRetriever.__new__(DenseRetriever)
        exact.store, exact.model_id, exact.is_qwen, exact.np = self.store, MODEL, False, self.np
        exact.encoder = SimpleNamespace(encode=lambda *args, **kwargs: [query])
        self.exact = exact
        self.manifest = self.build();adapter = self.load()
        result = adapter.search("synthetic fixture", k=3, source="a", id_prefix="a:")
        oracle = exact.search("synthetic fixture", k=3, source="a", id_prefix="a:")
        self.assertEqual([(item["id"], item["score"]) for item in result],
                         [(item["id"], item["score"]) for item in oracle])
        self.assertEqual(self.manifest["score_protocol"], {"arithmetic": "numpy_float32_matmul",
                                                          "numpy_version": self.np.__version__})

    def test_native_corrupt_manifest_is_rejected_before_native_read(self):
        self.manifest = self.build()
        with (self.output / "manifest.json").open("ab") as stream:
            stream.write(b"not-json")
        with patch.object(self.faiss, "read_index", wraps=self.faiss.read_index) as read:
            with self.assertRaises(json.JSONDecodeError):
                self.load()
            read.assert_not_called()

    def test_native_reset_add_same_structure_changes_frozen_memory_identity(self):
        self.manifest = self.build();adapter = self.load()
        scopes = [{"query_id": "q", "source": "a", "id_prefix": "a:"}]
        adapter.index_fingerprint(scopes=scopes)
        adapter.index.reset()
        adapter.index.add(self.np.asarray([[0, 1], [.8, .6], [1, 0]], dtype="float32"))
        with self.assertRaisesRegex(ValueError, "in-memory ANN"):
            adapter.index_fingerprint(scopes=scopes)
        self.assertEqual(list((self.root / "tmp").glob("ann-fingerprint-*.faiss")), [])

    def test_native_loaded_extension_must_belong_to_verified_inventory(self):
        self.runtime["binary_files"] = [item for item in self.runtime["binary_files"]
                                         if not item["path"].endswith(".pyd")]
        with self.assertRaisesRegex(ValueError, "Actually loaded FAISS extension"):
            self.build()


def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromTestCase(AdapterTests)
    if os.environ.get("MAGRA_ANN_NATIVE_TESTS") == "1":
        suite.addTests(loader.loadTestsFromTestCase(NativeTests))
    return suite


if __name__ == "__main__":
    unittest.main()
