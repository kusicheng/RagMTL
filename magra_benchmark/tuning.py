                                                                             

                                                                                
                                                                               
                                                                            
                                                                          
   

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import math
import re
from typing import Callable, Iterable, Mapping

from magra_rag.engine import MagraEngine
from magra_rag.retrieval import RetrievalConfig, Retriever

from .evaluation import evaluate_run, select_baselines
from .metrics import retrieval_metrics
from .protocol import code_hash, environment, json_hash
from .splits import PUBLIC_FIELDS


def default_tuning_configs(
    *, dense=None, reranker=None, top_k: int = 10, candidate_k: int = 50,
    context_bytes: int = 12_000,
) -> dict[str, RetrievalConfig]:
    pass                                                                     

                                                                      
                                                                             
                                                                            
                                                                               
                                                                               
                                                                         
       
    common = dict(top_k=top_k, candidate_k=candidate_k, context_bytes=context_bytes,
                  initial_k=min(10, candidate_k))
    result = {}
    weights = (0.5, 1.0, 2.0, 4.0)
    for weight in weights:
        result[f"bm25_title_{weight:g}"] = RetrievalConfig(strategy="bm25", title_weight=weight, **common)
    if dense is not None:
        result["dense"] = RetrievalConfig(strategy="dense", **common)
        for weight in weights:
            result[f"hybrid_title_{weight:g}"] = RetrievalConfig(strategy="hybrid", title_weight=weight, **common)
    if reranker is not None:
        for weight in weights:
            result[f"bm25_rerank_title_{weight:g}"] = RetrievalConfig(
                strategy="bm25", title_weight=weight, use_reranker=True, **common)
            if dense is not None:
                result[f"hybrid_rerank_title_{weight:g}"] = RetrievalConfig(
                    strategy="rerank", title_weight=weight, **common)
    backend_variants = [("", False, False)]
    if dense is not None:
        backend_variants.append(("_dense", True, False))
    if reranker is not None:
        backend_variants.append(("_rerank", False, True))
    if dense is not None and reranker is not None:
        backend_variants.append(("_dense_rerank", True, True))
    for suffix, use_dense, use_reranker in backend_variants:
        for strategy in ("magra", "magra_no_bridge", "magra_no_selection"):
            result[strategy + suffix] = RetrievalConfig(
                strategy=strategy, use_dense=use_dense, use_reranker=use_reranker, **common)
    return result


def _index(rows: Iterable[Mapping], *, label: str) -> dict[str, dict]:
    if isinstance(rows, Mapping):
                                                                          
        prepared = []
        for qid, row in rows.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} entries must be mappings")
            if row.get("id", row.get("query_id", qid)) != qid:
                raise ValueError(f"{label} mapping key disagrees with its record ID")
            prepared.append({"id": qid, **dict(row)})
        rows = prepared
    indexed = {}
    for row in rows:
        row = dict(row)
        qid = row.get("id", row.get("query_id"))
        if not isinstance(qid, str) or not qid or qid in indexed:
            raise ValueError(f"{label} requires unique nonempty string IDs")
        row["id"] = qid
        row.pop("query_id", None)
        indexed[qid] = row
    return indexed


def _prepare_queries(queries, private_labels, *, heldout_sources, heldout_domains, k):
    indexed = _index(queries, label="Development queries")
    if private_labels is not None:
        labels = _index(private_labels, label="Private development labels")
        if set(labels) != set(indexed):
            raise ValueError("Public queries and private labels must have exactly the same IDs")
        for qid, private in labels.items():
            for field, value in private.items():
                if field in PUBLIC_FIELDS and field != "id" and indexed[qid].get(field) != value:
                    raise ValueError("Private labels cannot change public query fields")
                if field in indexed[qid] and indexed[qid][field] != value:
                    raise ValueError("Conflicting public/private query annotations")
            indexed[qid].update(private)
    if not indexed:
        raise ValueError("Tuning requires development queries")
    forbidden_sources, forbidden_domains = set(heldout_sources), set(heldout_domains)
    for row in indexed.values():
        if row.get("split") != "dev":
            raise ValueError("Retrieval tuning is restricted to split='dev'")
        if row.get("official_split") in {"test", "eval", "evaluation", "locked_eval"}:
            raise ValueError("Official evaluation/test queries cannot be used for tuning")
        if (row.get("source") in forbidden_sources or row.get("domain") in forbidden_domains
                or row.get("held_out") or row.get("heldout") or row.get("adaptation_heldout")
                or row.get("heldout_source") or row.get("heldout_domain")):
            raise ValueError("A source/domain held out from adaptation cannot enter tuning")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Development queries need nonempty questions")
        if not isinstance(row.get("relevance"), Mapping):
            raise ValueError("Retrieval tuning requires private relevance judgments")
        retrieval_metrics([], row["relevance"], k)                                   
        if row.get("setting") != "provided_context" and (
                "allowed_doc_ids" in row or "context_doc_ids" in row):
            raise ValueError("Full-corpus tuning cannot use a restricted document candidate list")
    return list(indexed.values())


def _sample_queries(rows: list[dict], limit_per_stratum: int | None, seed: str) -> list[dict]:
    if (limit_per_stratum is not None and (isinstance(limit_per_stratum, bool)
            or not isinstance(limit_per_stratum, int) or limit_per_stratum < 1)):
        raise ValueError("limit_per_stratum must be a positive integer")
    grouped = defaultdict(list)
    for row in rows:
        axes = {name: row.get(name, False if name == "synthetic" else "") for name in
                ("source", "domain", "setting", "track", "scenario", "synthetic", "web_mode")}
        grouped[json_hash(axes)].append(row)
    selected = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda row: (json_hash([seed, row["id"]]), row["id"]))
        selected.extend(ordered[:limit_per_stratum] if limit_per_stratum else ordered)
    return selected


def tune_retrieval(
    store, queries: Iterable[Mapping], configs: Mapping[str, RetrievalConfig] | None = None, *,
    bundle_manifest_sha256: str, private_labels: Iterable[Mapping] | Mapping | None = None,
    dense=None, reranker=None, baseline_names: Iterable[str] | None = None,
    limit_per_stratum: int | None = None, seed: str = "magra-tune-v1",
    heldout_sources: Iterable[str] = ("arguana",), heldout_domains: Iterable[str] = (),
    quality_metric: str | None = None, k: int = 10,
    constraints: Mapping[str, float] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    pass                                                                          

                                                                            
                                                                               
                                                                           
                                                                               

                                                                                
                                                                              
                                                                              
                                                                             
                                                                            
       
    if not isinstance(bundle_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", bundle_manifest_sha256):
        raise ValueError("bundle_manifest_sha256 must be a lowercase SHA-256 digest")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    metric = quality_metric or f"ndcg@{k}"
    allowed_metrics = {f"{name}@{k}" for name in ("ndcg", "recall", "mrr", "map", "precision")}
    allowed_metrics.update({f"selected_{level}_{name}" for level in ("document", "sentence")
                            for name in ("precision", "recall", "f1", "em")})
    if metric not in allowed_metrics:
        raise ValueError("Tuning requires a retrieval or selected-evidence quality metric")
    source_holdouts = (heldout_sources,) if isinstance(heldout_sources, str) else tuple(heldout_sources)
    domain_holdouts = (heldout_domains,) if isinstance(heldout_domains, str) else tuple(heldout_domains)
    prepared = _prepare_queries(queries, private_labels, heldout_sources=source_holdouts,
                                heldout_domains=domain_holdouts, k=k)
    selected = _sample_queries(prepared, limit_per_stratum, seed)
    default_grid = configs is None
    grid = dict(default_tuning_configs(dense=dense, reranker=reranker, top_k=k,
                                      candidate_k=max(50, k)) if configs is None else configs)
    if not grid or any(not isinstance(name, str) or not name or name == "__heldout_fallbacks__" for name in grid):
        raise ValueError("Tuning needs named configurations")
    if any(not isinstance(config, RetrievalConfig) for config in grid.values()):
        raise ValueError("Each tuning configuration must be a RetrievalConfig")
    names = list(baseline_names) if baseline_names is not None else [
        name for name, config in grid.items() if not config.strategy.startswith("magra")]
    if len(set(names)) != len(names) or set(names) - set(grid):
        raise ValueError("Baseline names must uniquely reference declared configurations")
    limits = dict(constraints) if constraints is not None else {
        "process_peak_rss_bytes": 31_999_999_999,
        "peak_observed_physical_bytes": store.budget.disk_limit,
    }
    if any(isinstance(limit, bool) or not isinstance(limit, (int, float))
           or not math.isfinite(limit) or limit < 0 for limit in limits.values()):
        raise ValueError("Resource constraints must be finite and nonnegative")
                                                                                
                                                                             
    retrievers = {name: Retriever(store, config, dense=dense, reranker=reranker)
                  for name, config in grid.items()}
    corpus_hash = store.corpus_hash()
    implementation_hash = code_hash()
    runtime_environment = environment()
    public_queries = [{field: value for field, value in row.items() if field in PUBLIC_FIELDS}
                      for row in selected]
    fingerprint_method = getattr(dense, "index_fingerprint", None)
    requires_dense = any(config.use_dense or config.strategy in {"dense", "hybrid", "rerank"}
                         for config in grid.values())
    index_fingerprint = None
    if requires_dense and callable(fingerprint_method):
        from magra_rag.index_integrity import effective_query_scope
        index_scopes = [effective_query_scope(query) for query in public_queries]
        index_fingerprint = fingerprint_method(scopes=index_scopes)
    serialized_configs = {name: asdict(config) for name, config in grid.items()}
    reports, prediction_hashes = {}, {}
    for name, retriever in retrievers.items():
        started = datetime.now(timezone.utc).isoformat()
        engine = MagraEngine(retriever)
        outputs = []
        for index, query in enumerate(public_queries, 1):
            outputs.append(engine.run(query, retrieval_only=True))
            if on_progress is not None:
                on_progress(name, index, len(public_queries))
        resources = store.budget.check()
        prediction_hashes[name] = json_hash(outputs)
        metadata = {
            "format": "magra-run-v1", "split": "dev", "created_at": started,
            "corpus_hash": corpus_hash, "bundle_manifest_sha256": bundle_manifest_sha256,
            "query_ids": [row["id"] for row in selected], "config": serialized_configs[name],
            "code_hash": implementation_hash, "environment": runtime_environment,
            "generation": False, "model": None, "max_tokens": None,
            "calibration_hash": None,
            "dense_model": getattr(dense, "model_id", None) if
            (retriever.config.use_dense or retriever.config.strategy in {"dense", "hybrid", "rerank"}) else None,
            "dense_index_hash": index_fingerprint["fingerprint_hash"] if index_fingerprint and
            (retriever.config.use_dense or retriever.config.strategy in {"dense", "hybrid", "rerank"}) else None,
            "dense_index_fingerprint": index_fingerprint if
            (retriever.config.use_dense or retriever.config.strategy in {"dense", "hybrid", "rerank"}) else None,
            "reranker_model": getattr(reranker, "model_id", None) if
            (retriever.config.use_reranker or retriever.config.strategy == "rerank") else None,
            "prediction_records_hash": prediction_hashes[name],
            "prediction_count": len(outputs), "resources": resources, "status": "complete",
            "tuning": {"scope": "dev_only", "seed": seed, "limit_per_stratum": limit_per_stratum,
                       "corpus_sampled": False, "generation_invoked": False},
        }
        reports[name] = evaluate_run(selected, outputs, k=k, run_metadata=metadata)
    if store.corpus_hash() != corpus_hash:
        raise RuntimeError("SQL corpus changed during tuning; discard this development run")
    if index_fingerprint is not None and fingerprint_method(scopes=index_scopes) != index_fingerprint:
        raise RuntimeError("Dense SQL index changed during tuning; discard this development run")
    if code_hash() != implementation_hash:
        raise RuntimeError("Implementation changed during tuning; rerun the declared grid")
    best = select_baselines(reports, grid.keys(), quality_metric=metric, constraints=limits)
    baselines = select_baselines(reports, names, quality_metric=metric, constraints=limits) if names else {}
    manifest = {
        "format": "magra-dev-tuning-v1", "status": "complete", "split": "dev",
        "bundle_manifest_sha256": bundle_manifest_sha256, "corpus_hash": corpus_hash,
        "code_hash": implementation_hash, "environment": runtime_environment,
        "configs": serialized_configs, "baseline_names": names, "quality_metric": metric,
        "constraints": limits, "default_grid": default_grid, "seed": seed,
        "query_ids": sorted(row["id"] for row in selected), "input_query_count": len(prepared),
        "sampled_query_count": len(selected), "limit_per_stratum": limit_per_stratum,
        "public_queries_hash": json_hash(sorted(public_queries, key=lambda row: row["id"])),
        "heldout_sources": sorted(source_holdouts), "heldout_domains": sorted(domain_holdouts),
        "full_corpus_retained": True, "prediction_record_hashes": prediction_hashes,
        "report_hashes": {name: json_hash(report) for name, report in reports.items()},
        "best_per_stratum": best, "baseline_selection": baselines,
        "backend_inventory": {"dense_injected": dense is not None, "reranker_injected": reranker is not None,
                              "dense_model": getattr(dense, "model_id", None),
                              "reranker_model": getattr(reranker, "model_id", None)},
        "limitations": [
            "Selections and scores are development observations; no locked evaluation claim is made.",
            "Only explicitly injected backends enter the default grid; no model download or paid call occurs.",
            "Resource telemetry covers process-lifetime peak RAM and the accounted data root, not total host RAM or GPU VRAM.",
            "Resource peaks share one process across configurations; independent deployment measurements are still required.",
            "Retrieval-only tuning leaves monetary serving cost unmeasured.",
        ],
    }
    manifest["manifest_hash"] = json_hash(manifest)
    return {"development_reports": reports, "best_per_stratum": best,
            "baseline_selection": baselines, "configs": serialized_configs, "manifest": manifest}
