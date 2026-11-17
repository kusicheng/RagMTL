                                                                      

                                                                           
                                                                        
   
from __future__ import annotations

from contextlib import closing
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import sys
import tempfile
from types import ModuleType

from magra_rag.index_integrity import _frame, dense_index_fingerprint

FAISS_VERSION = "1.15.1"
FAISS_WHEEL_HASHES = {
    "8a577dd6d52f685326570105c3d18feb3776799d080534e329a191740d6362b6",
    "f2c31b7f2f6647eb76829a5cfe3c398fb9346df9f26b1d4db35269c91eb58c33",
}
PREFIX = "Represent this sentence for searching relevant passages: "
FILTER_POLICY = "exact_for_any_allowlist_or_nonempty_exclusion-v1"
SEARCH_OPTIONS = {"is_similarity": True, "search_method": 0,
                  "check_relative_distance": True, "search_bounded_queue": True}
MANIFEST_LIMIT = 65_536


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def _ids(values):
    if values is None:
        return None
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("Document filters require a collection of nonempty IDs")
    values = list(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Document filters require nonempty string IDs")
    return sorted(set(values))


def _inside(budget, path, *, file=False):
    original = Path(path).absolute()
    resolved = budget.contains(original)
    for item in (original, *original.parents):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError("ANN artifact paths cannot contain links or junctions")
        if item == budget.root:
            break
    if file and not resolved.is_file():
        raise ValueError("Required ANN artifact file is absent")
    return resolved


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def host_memory():
    pass                                                                           
    if sys.platform == "win32":
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total",
                 "virtual_available", "extended_available")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        measure = kernel.GlobalMemoryStatusEx
        measure.argtypes, measure.restype = [ctypes.POINTER(Memory)], wintypes.BOOL
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not measure(ctypes.byref(memory)):
            raise ctypes.WinError(ctypes.get_last_error())
        total, available = memory.total, memory.available
    elif sys.platform.startswith("linux"):
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                fields[key] = int(value.split()[0]) * 1024
        total, available = fields["MemTotal"], fields["MemAvailable"]
    else:
        raise RuntimeError("A measured host-memory guard is required on this platform")
    if not 0 <= available <= total or total <= 0:
        raise ValueError("Invalid native host-memory observation")
    return {"total_bytes": total, "available_bytes": available, "used_bytes": total - available}


class ResourceGuard:
    def __init__(self, budget, *, host_limit_bytes=32_000_000_000):
        self.budget = budget
        self.host_limit = _integer(host_limit_bytes, "host_limit_bytes", 1, 32_000_000_000)

    def __call__(self, *, disk_reserve=0, ram_reserve=0):
        _integer(ram_reserve, "ram_reserve", 0, 2_000_000_000_000)
        resources = self.budget.check(disk_reserve)
        memory = host_memory()
        if memory["used_bytes"] + ram_reserve >= self.host_limit:
            raise RuntimeError("Measured global host RAM plus reservation reaches the host cap")
        if memory["available_bytes"] < ram_reserve + 64_000_000:
            raise RuntimeError("Insufficient measured global host RAM headroom")
        return {"data_budget": resources, "host_ram": memory, "reserved_ram_bytes": ram_reserve}


@dataclass(frozen=True)
class HNSWConfig:
    dimensions: int
    source: str | None = None
    id_prefix: str | None = None
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    threads: int = 1
    seed: int = 12345
    batch_rows: int = 1024
    batch_bytes: int = 4_194_304
    max_index_bytes: int = 24_000_000_000
    unit_tolerance: float = 0.0001

    def __post_init__(self):
        for name, low, high in [("dimensions", 1, 262144), ("m", 2, 128),
                                ("ef_construction", 2, 65536), ("ef_search", 1, 65536),
                                ("threads", 1, 4), ("batch_rows", 1, 4096),
                                ("seed", 0, 2_147_483_647),
                                ("batch_bytes", 4, 16_777_216),
                                ("max_index_bytes", 1, 32_000_000_000)]:
            _integer(getattr(self, name), name, low, high)
        if self.batch_bytes < self.dimensions * 4 or self.ef_construction < self.m:
            raise ValueError("Batch byte bound or efConstruction is too small")
        if not isinstance(self.unit_tolerance, (int, float)) or isinstance(self.unit_tolerance, bool) \
                or not math.isfinite(self.unit_tolerance) or not 0 < self.unit_tolerance <= 0.001:
            raise ValueError("Unit tolerance must be finite and in (0, .001]")
        for name in ("source", "id_prefix"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value or "\0" in value):
                raise ValueError(f"{name} must be a nonempty literal string or None")

    def scope(self):
        return {"query_id": "ann-artifact-universe", "source": self.source,
                "id_prefix": self.id_prefix, "allowed_ids": None, "exclude_ids": []}


class ExistingDenseQueryEncoder:
    pass                                                                       
    def __init__(self, dense):
        self.dense = dense
        self.protocol_id = _hash({"format": "magra-existing-dense-query-v1",
                                  "model_id": dense.model_id, "is_query": bool(dense.is_qwen),
                                  "prefix": "" if dense.is_qwen else PREFIX,
                                  "normalize_embeddings": True})

    def __call__(self, question):
        kwargs = {"normalize_embeddings": True, "show_progress_bar": False}
        if self.dense.is_qwen:
            kwargs["is_query"] = True
        values = self.dense.encoder.encode([question if self.dense.is_qwen else PREFIX + question], **kwargs)
        if len(values) != 1:
            raise ValueError("Query encoder did not return exactly one vector")
        return values[0]


def _runtime(faiss, pin):
    required = {"version", "wheel_sha256", "binary_files"}
    if not isinstance(pin, dict) or set(pin) != required:
        raise ValueError("An exact FAISS wheel/native-binary runtime pin is required")
    if pin["version"] != FAISS_VERSION or getattr(faiss, "__version__", None) != FAISS_VERSION:
        raise ValueError("FAISS version differs from the staged primary pin")
    if pin["wheel_sha256"] not in FAISS_WHEEL_HASHES:
        raise ValueError("FAISS wheel hash differs from the pinned published wheels")
    binaries = pin["binary_files"]
    if not isinstance(binaries, list) or not binaries or len(binaries) > 64:
        raise ValueError("FAISS runtime requires a bounded native-binary inventory")
    identities, inventory_paths = [], {}
    for item in binaries:
        if not isinstance(item, dict) or set(item) != {"name", "path", "sha256"}:
            raise ValueError("Malformed native-binary inventory")
        name, native = item["name"], Path(item["path"])
        if (not isinstance(name, str) or not name or "\\" in name or name.startswith("/")
                or ".." in name.split("/") or not native.is_file() or native.is_symlink()
                or _sha(native) != item["sha256"]):
            raise ValueError("FAISS native binary checksum or path mismatch")
        identities.append({"name": name, "sha256": item["sha256"]})
        inventory_paths[native.resolve()] = name
    if len({item["name"] for item in identities}) != len(identities):
        raise ValueError("Duplicate runtime binary identities")
    loaded_extensions = []
    if isinstance(faiss, ModuleType):
        package = Path(faiss.__file__).resolve().parent
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if name.startswith("faiss._swigfaiss") and filename:
                native = Path(filename).resolve()
                if native.suffix in {".pyd", ".so"}:
                    if native not in inventory_paths or native.parent != package:
                        raise ValueError("Actually loaded FAISS extension is outside the verified runtime inventory")
                    loaded_extensions.append(inventory_paths[native])
        if not loaded_extensions:
            raise ValueError("Cannot bind the actually loaded FAISS native extension")
    return {"version": pin["version"], "wheel_sha256": pin["wheel_sha256"],
            "implementation": "native_module" if isinstance(faiss, ModuleType) else "injected_test_double",
            "loaded_extensions": sorted(loaded_extensions),
            "native_binaries": sorted(identities, key=lambda item: item["name"])}


def _where(config, *, after=None):
    clauses, parameters = [], []
    if config.source is not None:
        clauses.append("d.source=?")
        parameters.append(config.source)
    if config.id_prefix is not None:
        clauses.append("substr(d.id,1,?)=?")
        parameters.extend([len(config.id_prefix), config.id_prefix])
    if after is not None:
        clauses.append("d.id>?")
        parameters.append(after)
    return " AND ".join(clauses or ["1"]), parameters


def _fingerprint(store, model_id, config, scopes=None):
    return dense_index_fingerprint(store, model_id, scopes=[config.scope()] if scopes is None else scopes,
                                   expected_dimensions=config.dimensions)


def _vector(blob, dimensions, tolerance):
    if not isinstance(blob, bytes) or len(blob) != dimensions * 4:
        raise ValueError("Invalid canonical little-endian float32 vector bytes")
    squared = 0.0
    for (value,) in struct.iter_unpack("<f", blob):
        if not math.isfinite(value):
            raise ValueError("Nonfinite canonical vector")
        squared += value * value
    if abs(math.sqrt(squared) - 1.0) > tolerance:
        raise ValueError("HNSWFlat inner product requires unit canonical vectors")


def _structure(faiss, index, config, count, ef_search):
    if not isinstance(index, faiss.IndexHNSWFlat):
        raise ValueError("Only native IndexHNSWFlat is supported")
    storage = faiss.downcast_index(index.storage)
    if not isinstance(storage, faiss.IndexFlat):
        raise ValueError("HNSWFlat requires unquantized native IndexFlat storage")
    if (index.d != config.dimensions
            or index.metric_type != faiss.METRIC_INNER_PRODUCT or not index.is_trained
            or index.ntotal != count or storage.metric_type != faiss.METRIC_INNER_PRODUCT
            or storage.d != config.dimensions or storage.ntotal != count
            or index.hnsw.nb_neighbors(0) != 2 * config.m
            or index.hnsw.efConstruction != config.ef_construction
            or index.hnsw.efSearch != ef_search
            or any(getattr(index.hnsw, name) != value for name, value in SEARCH_OPTIONS.items())):
        raise ValueError("FAISS index structure, metric, count, or search parameters differ from the manifest")
    if faiss.omp_get_max_threads() != config.threads:
        raise ValueError("FAISS thread count changed from the pinned thread runtime")


def _mapping_hash(connection, expected_count):
    digest = hashlib.sha256(b"magra-ann-ordinals-v1\0")
    count = 0
    with closing(connection.execute("SELECT ordinal,doc_id,content_hash FROM ordinals ORDER BY ordinal")) as cursor:
        for ordinal, doc_id, content_hash in cursor:
            if ordinal != count or not isinstance(doc_id, str) or not doc_id or not isinstance(content_hash, str):
                raise ValueError("ANN ordinal mapping is not contiguous and complete")
            digest.update((_json([ordinal, doc_id, content_hash]) + "\n").encode())
            count += 1
    if count != expected_count:
        raise ValueError("ANN ordinal mapping count differs from the complete SQL universe")
    return digest.hexdigest()


def build_hnsw_flat(store, output, *, model_id, query_protocol_id, config,
                    faiss_module, numpy_module, runtime_pin, guard=None):
    pass                                                                      

                                                                              
                                                                            
       
    if (not isinstance(config, HNSWConfig) or not isinstance(model_id, str)
            or not re.fullmatch(r"[^@\s]+@[0-9a-f]{40}(?:#[^\s]+)?", model_id)):
        raise ValueError("A validated config and full pinned model revision are required")
    if not isinstance(query_protocol_id, str) or not re.fullmatch(r"[0-9a-f]{64}", query_protocol_id):
        raise ValueError("A hashed existing Dense query protocol is required")
    runtime = _runtime(faiss_module, runtime_pin)
    output = _inside(store.budget, output)
    output.mkdir(parents=True, exist_ok=False)
    guard = guard or ResourceGuard(store.budget)
    try:
        initial = _fingerprint(store, model_id, config)
        count = initial["document_count"]
        if not count:
            raise ValueError("Cannot publish an ANN index for an empty universe")
        where, parameters = _where(config)
        with closing(store.connection.execute(
                "SELECT COALESCE(SUM(length(CAST(d.id AS BLOB))),0) FROM documents d WHERE " + where,
                parameters)) as cursor:
            id_bytes = cursor.fetchone()[0]
        core = count * (config.dimensions * 4 + 8 * config.m + 64)
        if count * config.dimensions * 4 > config.max_index_bytes:
            raise ValueError("Canonical FP32 payload already exceeds the declared ANN index byte bound")
        reservation = core * 2 + 1_048_576
        resources = guard(disk_reserve=reservation * 2 + id_bytes * 8 + count * 1024 + 1_048_576,
                          ram_reserve=core * 2 + config.batch_bytes * 4)
        faiss_module.omp_set_num_threads(config.threads)
        index = faiss_module.IndexHNSWFlat(config.dimensions, config.m, faiss_module.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction, index.hnsw.efSearch = config.ef_construction, config.ef_search
        index.hnsw.rng = faiss_module.RandomGenerator(config.seed)
        for name, value in SEARCH_OPTIONS.items():
            setattr(index.hnsw, name, value)
        mapping_path, index_path = output / "ordinals.sqlite", output / "index.faiss"
        written, last_id = 0, None
        streamed_universe = hashlib.sha256(b"magra-dense-universe-v1\0")
        streamed_vectors = hashlib.sha256(b"magra-dense-vectors-v1\0")
        _frame(streamed_vectors, model_id)
        _frame(streamed_vectors, "<f4")
        rows_per_page = min(config.batch_rows, config.batch_bytes // (config.dimensions * 4))
        with closing(sqlite3.connect(mapping_path)) as mapping:
            mapping.execute("PRAGMA journal_mode=DELETE")
            mapping.execute("PRAGMA synchronous=FULL")
            mapping.execute("PRAGMA cache_size=-4096")
            mapping.execute("CREATE TABLE ordinals(ordinal INTEGER PRIMARY KEY,doc_id TEXT UNIQUE NOT NULL,content_hash TEXT NOT NULL)")
            while True:
                where, parameters = _where(config, after=last_id)
                with closing(store.connection.execute(
                        "SELECT d.id,d.content_hash,v.model,v.content_hash,v.dimensions,typeof(v.embedding),length(v.embedding) "
                        "FROM documents d INDEXED BY sqlite_autoindex_documents_1 "
                        "LEFT JOIN vectors v ON v.doc_id=d.id WHERE " + where + " ORDER BY d.id LIMIT ?",
                        [*parameters, rows_per_page])) as cursor:
                    rows = cursor.fetchall()
                if not rows:
                    break
                for doc_id, content, model, vector_hash, dimensions, kind, size in rows:
                    if (model != model_id or vector_hash != content or dimensions != config.dimensions
                            or kind != "blob" or size != config.dimensions * 4):
                        raise ValueError("Canonical SQL vector is missing, stale, or from another model")
                with closing(store.connection.execute(
                        "SELECT doc_id,model,content_hash,dimensions,embedding FROM vectors "
                        "WHERE doc_id IN (SELECT value FROM json_each(?))",
                        [_json([row[0] for row in rows])])) as cursor:
                    vector_rows = {row[0]: tuple(row[1:]) for row in cursor.fetchall()}
                blobs, ordinals = [], []
                for doc_id, content, *_ in rows:
                    item = vector_rows.pop(doc_id, None)
                    if item is None or item[:3] != (model_id, content, config.dimensions):
                        raise ValueError("Canonical SQL vectors changed between bounded page reads")
                    blob = item[3]
                    _vector(blob, config.dimensions, config.unit_tolerance)
                    blobs.append(blob)
                    ordinals.append((written + len(ordinals), doc_id, content))
                accumulated_core = (written + len(rows)) * (config.dimensions * 4 + config.m * 8 + 64)
                                                                              
                                                                                
                guard(ram_reserve=accumulated_core * 2 + config.batch_bytes * 4)
                matrix = numpy_module.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(rows), config.dimensions)
                actual_bytes = matrix.tobytes(order="C")
                if actual_bytes != b"".join(blobs):
                    raise ValueError("NumPy changed the canonical float32 bytes passed to FAISS")
                for number, (doc_id, content, *_) in enumerate(rows):
                    blob = actual_bytes[number * config.dimensions * 4:(number + 1) * config.dimensions * 4]
                    _frame(streamed_universe, doc_id)
                    _frame(streamed_universe, content)
                    for field in (doc_id, model_id, content):
                        _frame(streamed_vectors, field)
                    _frame(streamed_vectors, struct.pack(">Q", config.dimensions))
                    _frame(streamed_vectors, blob)
                index.add(matrix)
                with mapping:
                    mapping.executemany("INSERT INTO ordinals VALUES(?,?,?)", ordinals)
                written += len(rows)
                last_id = rows[-1][0]
                guard()
            mapping_hash = _mapping_hash(mapping, count)
        streamed = {"universe_hash": streamed_universe.hexdigest(),
                    "vectors_hash": streamed_vectors.hexdigest(), "document_count": written}
        if any(streamed[key] != initial[key] for key in streamed):
            raise ValueError("Actual streamed SQL vector bytes differ from the initial fingerprint")
        if written != count or _fingerprint(store, model_id, config) != initial:
            raise ValueError("Canonical SQL scope changed during ANN construction")
        _structure(faiss_module, index, config, count, config.ef_search)
        guard(disk_reserve=reservation)
        partial = output / "index.faiss.partial"
        faiss_module.write_index(index, str(partial))
        if not partial.is_file() or partial.stat().st_size > config.max_index_bytes:
            raise ValueError("Serialized ANN index exceeds the declared byte bound")
        os.replace(partial, index_path)
        files = {name: {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
                 for name, path in [("index", index_path), ("mapping", mapping_path)]}
        final_resources = guard(disk_reserve=MANIFEST_LIMIT)
        manifest = {"format": "magra-hnsw-flat-v1", "status": "complete", "model_id": model_id,
                    "query_protocol_id": query_protocol_id, "config": asdict(config),
                    "metric": "inner_product", "dtype": "<f4", "normalized": True,
                    "raw_sql": initial, "mapping_hash": mapping_hash, "files": files,
                    "streamed_sql_binding": streamed,
                    "faiss_runtime": runtime, "filter_policy": FILTER_POLICY,
                    "search_options": SEARCH_OPTIONS,
                    "score_protocol": {"arithmetic": "numpy_float32_matmul",
                                       "numpy_version": numpy_module.__version__},
                    "build_observation": {"initial": resources, "final": final_resources}}
        manifest["artifact_hash"] = _hash(manifest)
        _atomic_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        _atomic_json(output / "failure.json", {"status": "partial_ineligible", "error_type": type(error).__name__})
        raise


class FaissHNSWDenseRetriever:
    pass                                                                            
    def __init__(self, exact_backend, artifact, *, expected_manifest_hash,
                 faiss_module, numpy_module, runtime_pin, query_encoder=None,
                 ef_search=None, internal_candidates=0, guard=None):
        self.exact, self.store = exact_backend, exact_backend.store
        self.model_id, self.faiss, self.np = exact_backend.model_id, faiss_module, numpy_module
        if getattr(exact_backend, "np", None) is not numpy_module:
            raise ValueError("ANN rescoring must use the exact backend's injected NumPy module")
        self.guard = guard or ResourceGuard(self.store.budget)
        self.output = _inside(self.store.budget, artifact)
        manifest_path = _inside(self.store.budget, self.output / "manifest.json", file=True)
        if manifest_path.stat().st_size > MANIFEST_LIMIT:
            raise ValueError("ANN manifest exceeds its byte bound")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = _hash({key: value for key, value in manifest.items() if key != "artifact_hash"})
        if (manifest.get("format") != "magra-hnsw-flat-v1" or manifest.get("status") != "complete"
                or manifest.get("artifact_hash") != actual or actual != expected_manifest_hash):
            raise ValueError("Complete ANN manifest identity mismatch")
        self.config = HNSWConfig(**manifest["config"])
        self.encoder = query_encoder or ExistingDenseQueryEncoder(exact_backend)
        if (manifest["model_id"] != self.model_id or manifest["query_protocol_id"] != self.encoder.protocol_id
                or manifest["metric"] != "inner_product" or manifest["dtype"] != "<f4"
                or manifest["normalized"] is not True or manifest["filter_policy"] != FILTER_POLICY
                or manifest["search_options"] != SEARCH_OPTIONS
                or manifest["score_protocol"] != {"arithmetic": "numpy_float32_matmul",
                                                   "numpy_version": numpy_module.__version__}
                or manifest["faiss_runtime"] != _runtime(faiss_module, runtime_pin)):
            raise ValueError("ANN model, query protocol, metric, or runtime identity mismatch")
        self.manifest = manifest
        self.paths = {}
        for name, expected_name in [("index", "index.faiss"), ("mapping", "ordinals.sqlite")]:
            info = manifest["files"][name]
            if info["path"] != expected_name:
                raise ValueError("ANN artifact paths must use the fixed local filenames")
            path = _inside(self.store.budget, self.output / expected_name, file=True)
            _integer(info["bytes"], "artifact_file_bytes", 1, self.store.budget.disk_limit)
            if name == "index" and path.stat().st_size > self.config.max_index_bytes:
                raise ValueError("ANN index exceeds its declared byte bound")
            if path.stat().st_size != info["bytes"] or _sha(path) != info["sha256"]:
                raise ValueError("ANN artifact file checksum mismatch")
            self.paths[name] = path
        if self.paths["index"].stat().st_size > self.config.max_index_bytes:
            raise ValueError("ANN index exceeds its declared byte bound")
        if _fingerprint(self.store, self.model_id, self.config) != manifest["raw_sql"]:
            raise ValueError("ANN artifact is stale for the current canonical SQL scope")
        if any(manifest["streamed_sql_binding"][key] != manifest["raw_sql"][key]
               for key in ("universe_hash", "vectors_hash", "document_count")):
            raise ValueError("ANN build did not bind its actual streamed SQL bytes")
        count = manifest["raw_sql"]["document_count"]
        with closing(sqlite3.connect(self.paths["mapping"].as_uri() + "?mode=ro", uri=True, timeout=30)) as mapping:
            if _mapping_hash(mapping, count) != manifest["mapping_hash"]:
                raise ValueError("ANN ordinal mapping logical hash mismatch")
        self.guard(ram_reserve=self.paths["index"].stat().st_size * 2 + self.config.batch_bytes)
        faiss_module.omp_set_num_threads(self.config.threads)
        self.index = faiss_module.read_index(str(self.paths["index"]))
        _structure(faiss_module, self.index, self.config, count, self.config.ef_search)
        original_serialization = self._serialized_index_hash()
        if original_serialization != manifest["files"]["index"]["sha256"]:
            raise ValueError("Loaded native ANN serialization differs from the pinned artifact")
        self.internal_candidates = _integer(internal_candidates, "internal_candidates", 0, 4096)
        self.ef_search = self.config.ef_search if ef_search is None else _integer(ef_search, "ef_search", 1, 65536)
        self.index.hnsw.efSearch = self.ef_search
        self.runtime_index_sha256 = (original_serialization if self.ef_search == self.config.ef_search
                                     else self._serialized_index_hash())
        self.file_tokens = {name: self._file_token(path) for name, path in self.paths.items()}
        self.file_tokens["manifest"] = self._file_token(manifest_path)
        self.sql_token = self._sql_token()
        self.guard()

    def _serialized_index_hash(self):
        pass                                                                       
        self.guard(disk_reserve=self.config.max_index_bytes, ram_reserve=self.config.batch_bytes * 4)
        temporary_root = _inside(self.store.budget, self.store.budget.root / "tmp")
        temporary_root.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=temporary_root, prefix="ann-fingerprint-",
                                         suffix=".faiss", delete=False) as stream:
            path = Path(stream.name)
        try:
            self.faiss.write_index(self.index, str(path))
            if path.stat().st_size > self.config.max_index_bytes:
                raise ValueError("In-memory ANN serialization exceeds its declared byte bound")
            self.guard()
            return _sha(path)
        finally:
            _inside(self.store.budget, path).unlink(missing_ok=True)

    @staticmethod
    def _file_token(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    def _sql_token(self):
        with closing(self.store.connection.execute("PRAGMA data_version")) as cursor:
            version = cursor.fetchone()[0]
        return version, self.store.connection.total_changes, self._file_token(self.store.path)

    def _fresh(self):
        if self.store.connection.in_transaction:
            raise ValueError("Commit or roll back canonical SQL before using a published ANN artifact")
        self.guard()
        for name, token in self.file_tokens.items():
            path = self.output / "manifest.json" if name == "manifest" else self.paths[name]
            if self._file_token(_inside(self.store.budget, path, file=True)) != token:
                raise ValueError("Immutable ANN artifact changed after load; reload explicitly")
        token = self._sql_token()
        if token != self.sql_token:
            if _fingerprint(self.store, self.model_id, self.config) != self.manifest["raw_sql"]:
                raise ValueError("ANN artifact is stale for the current canonical SQL scope")
            self.sql_token = token
        _structure(self.faiss, self.index, self.config, self.manifest["raw_sql"]["document_count"], self.ef_search)

    def _scope(self, source, id_prefix):
        if (source, id_prefix) != (self.config.source, self.config.id_prefix):
            raise ValueError("Query source/prefix must exactly match the fixed ANN universe")

    def index_fingerprint(self, *, scopes):
        self._fresh()
        runtime_sha = self._serialized_index_hash()
        if runtime_sha != self.runtime_index_sha256:
            raise ValueError("Loaded in-memory ANN graph or vector storage changed after validation")
        scopes = list(scopes)
        for scope in scopes:
            self._scope(scope.get("source"), scope.get("id_prefix"))
        result = {"format": "magra-hnsw-query-index-v2", "model_id": self.model_id,
                  "artifact_hash": self.manifest["artifact_hash"],
                  "runtime_index_sha256": runtime_sha,
                  "raw_sql": _fingerprint(self.store, self.model_id, self.config, scopes),
                  "search_parameters": {"ef_search": self.ef_search, "threads": self.config.threads,
                                        "internal_candidates": self.internal_candidates,
                                        "filter_policy": FILTER_POLICY, "options": SEARCH_OPTIONS,
                                        "score_protocol": self.manifest["score_protocol"]}}
        self._fresh()
        result["fingerprint_hash"] = _hash(result)
        return result

    def search(self, question, *, k=20, allowed_ids=None, exclude_ids=None, source=None, id_prefix=None):
        self._scope(source, id_prefix)
        _integer(k, "k", 0, 4096)
        allowed, excluded = _ids(allowed_ids), _ids(exclude_ids) or []
        self._fresh()
        if k == 0 or allowed == []:
            return []
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty query question is required")
        if allowed is not None or excluded:
            result = self.exact.search(question, k=k, allowed_ids=allowed, exclude_ids=excluded,
                                       source=source, id_prefix=id_prefix)
            self._fresh()
            return [dict(document, retriever="dense_exact_filter_fallback") for document in result]
        values = tuple(float(value) for value in self.encoder(question))
        if len(values) != self.config.dimensions:
            raise ValueError("Query vector dimensions differ from the indexed protocol")
        query = struct.pack("<" + "f" * len(values), *values)
        _vector(query, self.config.dimensions, self.config.unit_tolerance)
        count = min(max(k, self.internal_candidates), self.manifest["raw_sql"]["document_count"])
        scores, ordinals = self.index.search(self.np.frombuffer(query, dtype="<f4").reshape(1, len(values)), count)
        if len(scores) != 1 or len(ordinals) != 1 or len(ordinals[0]) != count or len(scores[0]) != count:
            raise ValueError("Malformed ANN result shape")
        ids = []
        with closing(sqlite3.connect(self.paths["mapping"].as_uri() + "?mode=ro", uri=True, timeout=30)) as mapping:
            for ordinal, score in zip(ordinals[0], scores[0]):
                if isinstance(ordinal, bool) or int(ordinal) != ordinal or not math.isfinite(float(score)):
                    raise ValueError("Malformed ANN result identity or score")
                with closing(mapping.execute("SELECT doc_id,content_hash FROM ordinals WHERE ordinal=?", (int(ordinal),))) as cursor:
                    row = cursor.fetchone()
                if row is None or row[0] in ids:
                    raise ValueError("ANN result has a missing or duplicate ordinal")
                ids.append(row[0])
                                                                                
                                                                               
        ranked = []
        page_rows = min(self.config.batch_rows, self.config.batch_bytes // (len(values) * 4))
        query_vector = self.np.frombuffer(query, dtype="<f4")
        for offset in range(0, len(ids), page_rows):
            page_ids, blobs = ids[offset:offset + page_rows], []
            for doc_id in page_ids:
                with closing(self.store.connection.execute(
                        "SELECT d.content_hash,v.content_hash,v.model,v.dimensions,v.embedding FROM documents d "
                        "JOIN vectors v ON v.doc_id=d.id WHERE d.id=?", (doc_id,))) as cursor:
                    row = cursor.fetchone()
                if row is None or row[0] != row[1] or row[2] != self.model_id or row[3] != len(values):
                    raise ValueError("ANN candidate no longer matches canonical SQL")
                _vector(row[4], len(values), self.config.unit_tolerance)
                blobs.append(row[4])
            matrix = self.np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(blobs), len(values))
            scores = matrix @ query_vector
            for doc_id, score in zip(page_ids, scores):
                if not math.isfinite(float(score)):
                    raise ValueError("Nonfinite NumPy FP32 candidate score")
                document = self.store.get(doc_id)
                if document is None:
                    raise ValueError("ANN candidate document disappeared")
                ranked.append(dict(document, score=float(score), retriever="dense_ann_hnsw_flat"))
            self.guard()
        self._fresh()
        return sorted(ranked, key=lambda item: (-item["score"], item["id"]))[:k]
