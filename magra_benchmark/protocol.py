                                                                      

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys

from .splits import read_split_bundle, verify_split_bundle
from .datasets import sha256_file


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open("rb") as stream:
        while line := stream.readline(16_000_001):
            if len(line) > 16_000_000:
                raise ValueError("JSONL record exceeds 16 MB; split oversized owned documents before import")
            if line.strip():
                yield json.loads(line.decode("utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
                                                                                      
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def code_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    files = sorted([*root.glob("magra_rag/*.py"), *root.glob("magra_benchmark/*.py")])
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def sample_queries(rows, limit_per_stratum=None, *, seed="magra-v1"):
    if limit_per_stratum is not None and limit_per_stratum < 1:
        raise ValueError("Query limit must be positive")
    groups = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in ("source", "domain", "setting", "track", "scenario"))
        groups.setdefault(key, []).append(row)
    selected = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda r: json_hash([seed, r["id"]]))
        selected.extend(ordered[:limit_per_stratum] if limit_per_stratum else ordered)
    return selected


def dense_expected_dimensions(model_id: str) -> int | None:
    pass                                                                         
    model = model_id.split("@", 1)[0]
    return {"BAAI/bge-small-en-v1.5": 384,
            "Qwen/Qwen3-Embedding-0.6B": 1024}.get(model)


def _require_pinned_retrieval_model(model_id, name: str, *, role=None, metadata=None) -> None:
    if model_id is None:
        return
    advanced = isinstance(model_id, str) and "#qwen-native-cuda-v1:" in model_id
    pattern = (r"Qwen/Qwen3-Reranker-0\.6B@[0-9a-f]{40}#qwen-native-cuda-v1:[0-9a-f]{16}"
               if advanced and role == "reranker_model" else
               r"[^@\s#]+@[0-9a-f]{40}(?:#qwen-native-v1:[0-9a-f]{16,64})?")
    if (not isinstance(model_id, str) or not re.fullmatch(pattern, model_id)
            or ("#" in model_id and not model_id.startswith("Qwen/"))):
        raise ValueError(f"Pin dense and reranker models to lowercase 40-hex revisions before locked evaluation: {name}")
    if advanced:
        from magra_rag.dense import reranker_runtime_metadata
        base, suffix = model_id.split("@", 1)
        expected = reranker_runtime_metadata(base, suffix.split("#", 1)[0], device="cuda:0")
        if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"CUDA reranker requires matching execution metadata, package versions and hash: {name}")


def _development_dense_queries(report, name: str, public_queries: dict) -> list[dict]:
    pass                                                                          
    ids = report.get("run_metadata", {}).get("query_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(qid, str) or not qid for qid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(f"Dense development report requires unique nonempty query_ids: {name}")
    rows = report.get("per_query")
    if not isinstance(rows, list):
        raise ValueError(f"Dense development report is missing per-query membership: {name}")
    report_ids = [row.get("query_id") for row in rows]
    if (any(not isinstance(qid, str) or not qid for qid in report_ids)
            or len(set(report_ids)) != len(report_ids) or set(report_ids) != set(ids)
            or report.get("query_count") != len(ids)):
        raise ValueError(f"Dense development report query_ids differ from scored queries: {name}")
    unknown = set(ids) - set(public_queries)
    if unknown:
        raise ValueError(f"Dense development report includes unknown public DEV query IDs: {name}")
    return [public_queries[qid] for qid in sorted(ids)]


def runtime_identity(metadata):
    pass                                                                           
    result = {field:metadata.get(field) for field in
              ("generation","model","dense_model","dense_index_hash","reranker_model","max_tokens","reasoning_effort")}
    if metadata.get("dense_backend_hash") is not None:result["dense_backend_hash"] = metadata["dense_backend_hash"]
    advanced = isinstance(metadata.get("reranker_model"), str) and "#qwen-native-cuda-v1:" in metadata["reranker_model"]
    execution_hash = metadata.get("reranker_execution_hash")
    if advanced:
        if not isinstance(execution_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", execution_hash):
            raise ValueError("CUDA reranker runtime requires its execution hash")
        result["reranker_execution_hash"] = execution_hash
    elif execution_hash is not None or metadata.get("reranker_execution") is not None:
        raise ValueError("Reranker execution metadata requires the explicit CUDA reranker policy")
    return result


def freeze_protocol(path, *, bundle, corpus_hash, configs, dev_reports, baseline_names,
                    candidate_name, quality_metric="ndcg@10", limit_per_stratum=None,
                    calibration_hash=None, constraints=None, store=None):
    from .evaluation import select_baselines
    manifest = verify_split_bundle(bundle)
    if set(configs) != set(dev_reports):
        raise ValueError("Each frozen configuration requires a development report")
    if candidate_name not in configs or any(n not in configs for n in baseline_names):
        raise ValueError("Candidate and all baselines require frozen configurations")
    if store is not None and store.corpus_hash() != corpus_hash:
        raise ValueError("SQL store corpus differs from the development corpus")
    public_dev, fingerprint_cache, dev_indexes, eval_indexes = None, {}, {}, {}
    evaluation_queries = sample_queries(read_split_bundle(bundle, "eval"), limit_per_stratum)
    runtimes = {}

    def fingerprint(model_id, queries):
        from magra_rag.index_integrity import dense_index_fingerprint, effective_query_scope
        scopes = [effective_query_scope(query) for query in queries]
        key = (model_id, json_hash(scopes))
        if key not in fingerprint_cache:
            fingerprint_cache[key] = dense_index_fingerprint(
                store, model_id, scopes=scopes, expected_dimensions=dense_expected_dimensions(model_id))
        return fingerprint_cache[key]

    for name, report in dev_reports.items():
        if report.get("splits") != ["dev"]:
            raise ValueError(f"Development report must use dev queries only: {name}")
        metadata = report.get("run_metadata", {})
        if metadata.get("config") != configs[name] or metadata.get("corpus_hash") != corpus_hash:
            raise ValueError(f"Development run configuration or corpus differs: {name}")
        if metadata.get("bundle_manifest_sha256") != sha256_file(Path(bundle) / "manifest.json"):
            raise ValueError(f"Development run used a different split bundle: {name}")
        if metadata.get("status") != "complete":
            raise ValueError(f"Development run is incomplete: {name}")
        if metadata.get("environment", {}).get("code_hash") != code_hash():
            raise ValueError(f"Development run implementation differs: {name}")
        for field in ("dense_model", "reranker_model"):
            _require_pinned_retrieval_model(metadata.get(field), name, role=field, metadata=metadata)
        runtime = runtime_identity(metadata)
        runtime["dense_index_hash"] = None
        uses_dense = configs[name].get("use_dense") or configs[name].get("strategy") in {"dense", "hybrid", "rerank"}
        if uses_dense or metadata.get("dense_model") is not None:
            if store is None:
                raise ValueError(f"Dense development runs require the SQL store to freeze index identity: {name}")
            model_id = metadata.get("dense_model")
            if model_id is None:
                raise ValueError(f"Dense development report is missing its embedding model identity: {name}")
            if public_dev is None:
                public_dev = {}
                for query in read_split_bundle(bundle, "dev"):
                    qid = query.get("id")
                    if not isinstance(qid, str) or not qid or qid in public_dev:
                        raise ValueError("Public DEV records require unique nonempty query IDs")
                    public_dev[qid] = query
            dev_queries = _development_dense_queries(report, name, public_dev)
            if metadata.get("dense_backend_hash") is not None:
                from magra_rag.ann_catalog import describe_catalog
                backend = metadata.get("dense_backend", {})
                if not isinstance(backend,dict) or set(backend)!={"catalog_path","identity"}:
                    raise ValueError("ANN development report requires explicit published backend provenance")
                current = describe_catalog(backend["catalog_path"],store.budget,model_id=model_id,queries=dev_queries)
                if current["identity"] != backend["identity"] or current["identity"]["backend_hash"] != runtime["dense_backend_hash"]:
                    raise ValueError("ANN development artifact/runtime/search configuration differs")
                describe_catalog(backend["catalog_path"],store.budget,model_id=model_id,queries=evaluation_queries)
            dev_indexes[name] = fingerprint(model_id, dev_queries)
            if metadata.get("dense_index_hash") != dev_indexes[name]["fingerprint_hash"]:
                raise ValueError(f"Development dense index differs from the current scoped SQL vectors: {name}")
            eval_indexes[name] = fingerprint(model_id, evaluation_queries)
            runtime["dense_index_hash"] = eval_indexes[name]["fingerprint_hash"]
        elif metadata.get("dense_backend_hash") is not None:
            raise ValueError("Non-dense report cannot identify an ANN backend")
        elif metadata.get("dense_index_hash") is not None:
            raise ValueError(f"Non-dense development report cannot identify a dense index: {name}")
        runtimes[name] = runtime
    selection = select_baselines(dev_reports, baseline_names, quality_metric=quality_metric,
                                 constraints=constraints)
    if store is not None and store.corpus_hash() != corpus_hash:
        raise ValueError("SQL corpus changed while freezing the protocol")
    frozen = {"format": "magra-preregistration-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "bundle_manifest_sha256": sha256_file(Path(bundle) / "manifest.json"), "corpus_hash": corpus_hash,
              "configs": configs, "development_report_hashes": {n: json_hash(r) for n, r in dev_reports.items()},
              "runtimes": runtimes,
              "dense_indexes": {"development": dev_indexes, "evaluation": eval_indexes},
              "baseline_names": list(baseline_names), "candidate_name": candidate_name,
              "quality_metric": quality_metric, "baseline_selection": selection,
              "eval_query_ids": sorted(q["id"] for q in evaluation_queries),
              "eval_queries_hash": json_hash(sorted(evaluation_queries, key=lambda q: q["id"])),
              "limit_per_stratum": limit_per_stratum, "seed": "magra-v1",
              "calibration_hash": calibration_hash, "constraints": constraints or {},
              "code_hash": code_hash(),
              "acceptance": {"quality": "paired 95% quality CI above zero at comparable cost",
                             "cost": "paired 95% cost CI below zero with prespecified quality noninferiority margin",
                             "quality_noninferiority_margin": 0.0,
                             "generality": "five domains including an adaptation-heldout domain; natural and web/failure strata separately"},
              "status": "preregistered; no outcome claimed"}
    frozen["protocol_hash"] = json_hash(frozen)
    write_json(path, frozen)
    return frozen


def verify_protocol(path, *, bundle, corpus_hash, config, queries, calibration_hash=None, runtime=None):
    frozen = load_json(path)
    expected = frozen.pop("protocol_hash")
    if json_hash(frozen) != expected:
        raise ValueError("Preregistration integrity check failed")
    frozen["protocol_hash"] = expected
    manifest = verify_split_bundle(bundle, expected_manifest_sha256=frozen["bundle_manifest_sha256"])
    if corpus_hash != frozen["corpus_hash"]:
        raise ValueError("Corpus changed after preregistration")
    if code_hash() != frozen["code_hash"]:
        raise ValueError("Implementation changed after preregistration; refreeze using dev only")
    if config not in frozen["configs"].values():
        raise ValueError("Configuration was not preregistered")
    if runtime is not None:
        runtime = {"reasoning_effort": None, "dense_index_hash": None, **runtime}
    if runtime is not None and not any(config == value and runtime == frozen["runtimes"][name]
                                       for name, value in frozen["configs"].items()):
        raise ValueError("Model revisions or generation settings differ from preregistration")
    if runtime is None and any(value.get("generation") or value.get("dense_model") or value.get("reranker_model")
                               for value in frozen["runtimes"].values()):
        raise ValueError("Full runtime identity is required for model-based locked evaluation")
    if sorted(q["id"] for q in queries) != frozen["eval_query_ids"]:
        raise ValueError("Evaluation query sample differs from preregistration")
    if json_hash(sorted(queries, key=lambda q: q["id"])) != frozen["eval_queries_hash"]:
        raise ValueError("Evaluation query contents differ from preregistration")
    if calibration_hash != frozen["calibration_hash"]:
        raise ValueError("Calibration artifact differs from preregistration")
    return frozen


def environment() -> dict:
    return {"python": sys.version, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "code_hash": code_hash()}
