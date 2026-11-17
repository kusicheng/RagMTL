                                                                              

from __future__ import annotations

import argparse
from magra_continuity.source_runtime import SourceDocumentedParser
from contextlib import closing, ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import time

from magra_benchmark.datasets import (BEIR_DATASETS, download_beir, iter_beir_documents,
    iter_beir_queries, iter_hotpot, iter_hotpot_documents, iter_hotpot_wikipedia, sha256_file)
from magra_benchmark.protocol import (environment, freeze_protocol, json_hash, load_json,
    read_jsonl, sample_queries, verify_protocol, write_json, dense_expected_dimensions, runtime_identity)
from magra_benchmark.splits import make_splits, read_split_bundle, verify_split_bundle, write_split_bundle
from .engine import MagraEngine
from .dense import (DEFAULT_DENSE_MODEL, DEFAULT_RERANKER_MODEL, QWEN_DENSE_MODEL,
                    QWEN_RERANKER_MODEL, retrieval_model_identity, reranker_runtime_metadata)
from .resources import ResourceBudget, peak_rss_bytes
from .retrieval import RetrievalConfig, Retriever
from .storage import DocumentStore


def _budget(args):
    return ResourceBudget(args.data_root, disk_bytes=args.disk_bytes)


def _store(args, readonly=False):
    return DocumentStore(args.db or Path(args.data_root) / "corpus.sqlite", budget=_budget(args),
                         readonly=readonly)


def _config(args):
    return RetrievalConfig(**(load_json(args.config) if args.config else {"strategy": args.strategy}))


def _reranker_runtime(args, enabled):
    device = getattr(args, "reranker_device", "cpu")
    if not enabled:
        if device != "cpu":
            raise ValueError("--reranker-device requires an active reranker")
        return {"reranker_model": None}
    return reranker_runtime_metadata(args.reranker_model, args.reranker_revision, device=device)


def _retriever(store, config, args):
    reranker_policy = _reranker_runtime(args, config.use_reranker or config.strategy == "rerank")
    dense, reranker, runtime = None, None, None
    approximate = getattr(args, "dense_backend", "exact") == "hnsw"
    setup = {}
    try:
        if approximate:
            from .ann_catalog import LoadedRuntime
            started = time.perf_counter()
            runtime = LoadedRuntime(args._ann_description["catalog"]["runtime"], store.budget, serving=True)
            setup["runtime_startup_ms"] = (time.perf_counter()-started)*1000
        started = time.perf_counter()
        if config.use_dense or config.strategy in {"dense", "hybrid", "rerank"}:
            from .dense import DenseRetriever
            if runtime is not None:runtime.guard(ram_reserve=3_500_000_000)
            dense = DenseRetriever(store, model=args.dense_model, revision=args.dense_revision)
        if config.use_reranker or config.strategy == "rerank":
            from .dense import CrossEncoderReranker
            if runtime is not None:runtime.guard(ram_reserve=3_500_000_000)
            options = {"device": args.reranker_device} if getattr(args, "reranker_device", "cpu") != "cpu" else {}
            reranker = CrossEncoderReranker(store, model=args.reranker_model, revision=args.reranker_revision, **options)
            if options and reranker.runtime_metadata != reranker_policy:
                raise ValueError("Instantiated CUDA reranker execution differs from pre-load identity")
        setup["model_load_latency_ms"] = (time.perf_counter()-started)*1000
        if approximate:
            from .ann_catalog import RoutedHNSWDenseRetriever
            runtime.refresh({"dense":getattr(dense,"cpu_threads",None), "reranker":getattr(reranker,"cpu_threads",None)})
            started = time.perf_counter()
            dense = RoutedHNSWDenseRetriever(dense, args._ann_description, runtime)
            setup["index_load_and_validation_ms"] = (time.perf_counter()-started)*1000
            dense.setup_observation = setup
        return Retriever(store, config, dense=dense, reranker=reranker)
    except BaseException:
        try:
            if getattr(args, "reranker_device", "cpu") != "cpu" and reranker is not None:reranker.close()
        finally:
            if runtime is not None:runtime.close()
        raise


def _dense_fingerprint(store, model_id, queries):
    if model_id is None:
        return None
    from .index_integrity import dense_index_fingerprint, effective_query_scope
    dimensions = dense_expected_dimensions(model_id)
    return dense_index_fingerprint(store, model_id, scopes=[effective_query_scope(q) for q in queries],
                                   expected_dimensions=dimensions)


def _generation_settings(args):
    from .configuration import load_openrouter_settings
    return load_openrouter_settings(env_file=args.env_file, model_id=args.model,
                                    profile=getattr(args, "profile", "auto"))


def _generation_options(args, settings):
    from .configuration import paid_controls
    paid = settings.profile == "paid"
    prices = (getattr(args, "prompt_price", None), getattr(args, "completion_price", None))
    shared, limit = getattr(args, "shared_ledger", None), getattr(args, "max_usd", None)
    if paid:
        controls = paid_controls(max_total_usd=limit, shared_ledger_path=shared,
                                 prompt_price=prices[0], completion_price=prices[1])
        limit, shared = controls["max_total_usd"], controls["shared_ledger_path"]
        prices = controls["prompt_price"], controls["completion_price"]
    elif shared is not None:
        raise ValueError("--shared-ledger requires the paid profile")
    tokens = getattr(args, "max_output_tokens", None)
    tokens = (4096 if paid or args.command in {"judge", "judge-inputs"} else 512) if tokens is None else tokens
    coverage_path = getattr(args, "billing_coverage", None)
    coverage_sha = getattr(args, "billing_coverage_sha256", None)
    coverage = None
    if coverage_path is not None or coverage_sha is not None:
        if not paid or coverage_path is None or coverage_sha is None:
            raise ValueError("Paid billing coverage requires both --billing-coverage and --billing-coverage-sha256")
        from .billing_coverage import load_billing_coverage
        coverage = load_billing_coverage(Path(__file__).resolve().parents[1], coverage_path,
                                        expected_sha256=coverage_sha)
    return {"profile": settings.profile, "model_id": settings.model_id, "api_key": settings.api_key,
            "env_file": None, "max_total_usd": limit, "shared_ledger_path": shared,
            "billing_coverage": coverage,
            "max_tokens": tokens, "reasoning_effort": getattr(args, "reasoning_effort", None),
            "prompt_usd_per_million": 0 if prices[0] is None else prices[0],
            "completion_usd_per_million": 0 if prices[1] is None else prices[1]}


def _generation_client(args, budget, settings=None):
    from .generation import OpenRouterM3
    settings = settings or _generation_settings(args)
    return OpenRouterM3(data_root=args.data_root, budget=budget, **_generation_options(args, settings))


def _paid_options(parser):
    parser.add_argument("--profile", choices=("auto", "paid", "legacy"), default="auto",
                        help="Auto prefers ai-model-key-increased-paid, then ai-model-paid-key; explicit other/free --model uses legacy")
    parser.add_argument("--shared-ledger", help="Paid: exact existing shared ledger only; never reset")
    parser.add_argument("--billing-coverage", help="Explicit local proof for a retained unknown reservation")
    parser.add_argument("--billing-coverage-sha256", help="Trusted SHA256 of the explicitly selected coverage proof")


def cmd_doctor(args):
    settings = _generation_settings(args)
    options = _generation_options(args, settings)
    budget = _budget(args)
    with closing(sqlite3.connect(":memory:")) as c:
        c.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
    result = {"python": sys.version, "sqlite": sqlite3.sqlite_version, "fts5": True,
              "openrouter_key_present": settings.key_configured,
              "model": settings.model_id, "profile": settings.profile, "cpu_supported": True,
              "prompt_price_per_million": options["prompt_usd_per_million"],
              "completion_price_per_million": options["completion_usd_per_million"],
              "shared_ledger": str(options["shared_ledger_path"]) if options["shared_ledger_path"] else None,
              "cumulative_max_usd": None if settings.profile == "paid" else options["max_total_usd"],
              "monthly_max_usd": options["max_total_usd"] if settings.profile == "paid" else None,
              "resources": budget.check()}
    if args.check_openrouter:
        client = _generation_client(args, budget, settings)
        result["catalog"] = client.preflight()
        result["account_limits"] = client.account_limits()
    return result


def cmd_download(args):
    if sum((bool(args.hotpot), bool(args.ragtruth), bool(args.datasets))) != 1:
        raise ValueError("Choose one Hotpot variant, RAGTruth, or one or more BEIR datasets")
    budget = _budget(args)
    destination = budget.contains(Path(args.data_root) / "sources")
                                                                                   
    if not args.ragtruth:
        budget.check(args.max_download_bytes + args.max_extracted_bytes)
    result = []
    if args.ragtruth:
        if args.archive:
            raise ValueError("RAGTruth acquisition uses pinned publisher files; --archive is unavailable")
        from magra_benchmark.ragtruth import RAGTRUTH_REVISION, acquire_ragtruth
        path = acquire_ragtruth(destination / ("ragtruth-" + RAGTRUTH_REVISION[:12]),
                               budget, max_bytes=args.max_download_bytes)
        result.append(str(path))
    elif args.hotpot:
        from magra_benchmark.datasets import download_hotpot
        path = download_hotpot(args.hotpot, destination, max_bytes=args.max_download_bytes,
                               max_uncompressed_bytes=args.max_extracted_bytes,
                               archive_path=budget.contains(args.archive) if args.archive else None)
        result.append(str(path))
    else:
        for name in args.datasets:
            kwargs = {"max_bytes": args.max_download_bytes,
                      "max_uncompressed_bytes": args.max_extracted_bytes}
            if args.archive:
                if len(args.datasets) != 1:
                    raise ValueError("--archive requires exactly one dataset")
                kwargs["archive_path"] = budget.contains(args.archive)
            path = download_beir(name, destination, **kwargs)
            result.append(str(path))
            budget.check()
    return {"acquired": result, "resources": budget.check()}


def cmd_ingest(args):
    path = Path(args.path)
    if args.format == "beir":
        documents = iter_beir_documents(path)
    elif args.format == "hotpot_context":
        documents = iter_hotpot_documents(path)
    elif args.format == "hotpot_wiki":
        documents = iter_hotpot_wikipedia(path)
    else:
        documents = read_jsonl(path)
    with _store(args) as store:
        imported = store.ingest(documents, batch_size=args.batch_size)
        from magra_benchmark.integrity import record_import
        imported["import_receipt"] = record_import(
            store, format=args.format, source_path=path,
            imported_count=imported["inserted"] + imported["already_present"])
        imported["corpus_hash"] = store.corpus_hash()
        return imported


def cmd_prepare(args):
    budget = _budget(args)
    budget.contains(args.out)
    queries, acquisition = {}, []
    priority = {"train": 0, "dev": 1, "test": 2, "eval": 2}
    for directory in args.beir:
        directory = Path(directory)
        manifest_path = directory / "acquisition.json"
        if manifest_path.exists():
            acquisition.append(load_json(manifest_path))
        for subset in ("train", "dev", "test"):
            if not (directory / "qrels" / (subset + ".tsv")).exists():
                continue
            for query in iter_beir_queries(directory, subset):
                old = queries.get(query["id"])
                if old and old["question"] != query["question"]:
                    raise ValueError("Conflicting repeated official query ID")
                if not old or priority[subset] > priority[old["official_split"]]:
                    queries[query["id"]] = query
    for path, official in [(p, "train") for p in args.hotpot_train] + [(p, "dev") for p in args.hotpot_dev]:
        hotpot_manifest = Path(path).parent / "acquisition.json"
        original_manifest = Path(path).with_suffix(Path(path).suffix + ".acquisition.json")
        if hotpot_manifest.exists():
            acquisition.append(load_json(hotpot_manifest))
        elif original_manifest.exists():
            acquisition.append(load_json(original_manifest))
        for setting in args.hotpot_settings:
            for query in iter_hotpot(path, setting, subset=official):
                if query["id"] in queries:
                    raise ValueError("Repeated Hotpot query; provide one question file per split")
                queries[query["id"]] = query
    for path in args.queries:
        for query in read_jsonl(path):
            if query["id"] in queries:
                raise ValueError("Repeated owned query ID")
            queries[query["id"]] = query
    if not queries:
        raise ValueError("No queries selected")
    splits = make_splits(queries.values(), heldout_sources=args.heldout_sources,
                         partition_official_dev_sources=("hotpotqa",) if args.partition_hotpot_dev else (),
                         similarity_threshold=args.duplicate_threshold)
    provenance = {"heldout_sources": args.heldout_sources, "acquisitions": acquisition,
                  "partition_official_dev_sources": ["hotpotqa"] if args.partition_hotpot_dev else [],
                  "full_corpus_preserved": True,
                  "note": "Bundle contains queries and private annotations; SQL corpus completeness must be audited separately."}
    result = write_split_bundle(args.out, splits, provenance=provenance,
                                similarity_threshold=args.duplicate_threshold)
    budget.check()
    return result


def cmd_dense_index(args):
    from .dense import DenseRetriever
    with _store(args) as store:
        return DenseRetriever(store, model=args.dense_model, revision=args.dense_revision,
                              batch_size=args.batch_size).build(sources=args.sources)


def cmd_ann_build(args):
    pass                                                                         
    from types import SimpleNamespace
    from .ann import HNSWConfig, ExistingDenseQueryEncoder, build_hnsw_flat
    from .ann_catalog import LoadedRuntime, read_runtime
    budget = _budget(args)
    model_id = retrieval_model_identity(args.dense_model, args.dense_revision)
    dimensions = dense_expected_dimensions(model_id)
    if dimensions is None:raise ValueError("Known pinned embedding dimensions required")
    configuration = read_runtime(budget.contains(args.ann_runtime))
    config = HNSWConfig(dimensions=dimensions, source=args.source, id_prefix=args.id_prefix,
        threads=1, m=args.m, ef_construction=args.ef_construction, ef_search=args.ef_search,
        max_index_bytes=args.max_index_bytes)
    protocol = ExistingDenseQueryEncoder(SimpleNamespace(model_id=model_id, is_qwen=args.dense_model==QWEN_DENSE_MODEL)).protocol_id
    with _store(args, readonly=True) as store, closing(LoadedRuntime(configuration,budget)) as runtime:
        manifest = build_hnsw_flat(store, budget.contains(args.out), model_id=model_id, query_protocol_id=protocol,
            config=config, faiss_module=runtime.faiss, numpy_module=runtime.np, runtime_pin=configuration["pin"], guard=runtime.guard)
        runtime.validate_files()
        return {"artifact":str(budget.contains(args.out)), "artifact_hash":manifest["artifact_hash"],
                "runtime_profile":runtime.profile(), "canonical_SQL_readonly":True}


def cmd_run(args):
    config = _config(args)
    reranker_policy = _reranker_runtime(args, config.use_reranker or config.strategy == "rerank")
    approximate = getattr(args,"dense_backend","exact") == "hnsw"
    if approximate and (not args.ann_catalog or not (config.use_dense or config.strategy in {"dense","hybrid","rerank"})):
        raise ValueError("HNSW requires an explicit catalog and dense retrieval configuration")
    if not approximate and getattr(args,"ann_catalog",None):raise ValueError("ANN catalog requires --dense-backend hnsw")
    budget = _budget(args)
    out = budget.contains(args.out)
    if out.exists() or Path(str(out) + ".manifest.json").exists():
        raise FileExistsError("Run outputs already exist; choose a new path")
    calibrator = None
    if args.calibration:
        from magra_benchmark.calibration import load_calibrator
        calibrator = load_calibrator(args.calibration)
    queries = sample_queries(read_split_bundle(args.bundle, args.split), args.limit_per_stratum)
    settings = _generation_settings(args) if args.generate else None
    options = _generation_options(args, settings) if settings else None
    runtime = {"generation": args.generate, "model": settings.model_id if settings else None,
               "max_tokens": options["max_tokens"] if options else None,
               "reasoning_effort": (options["reasoning_effort"] or ("low" if settings.profile == "paid" else None)) if options else None,
               "dense_model": retrieval_model_identity(args.dense_model, args.dense_revision)
               if config.use_dense or config.strategy in {"dense", "hybrid", "rerank"} else None,
               "reranker_model": reranker_policy["reranker_model"]}
    cuda_reranker = "reranker_execution_hash" in reranker_policy
    if cuda_reranker:runtime["reranker_execution_hash"] = reranker_policy["reranker_execution_hash"]
    with _store(args, readonly=approximate) as store, ExitStack() as cleanup:
        corpus_hash = store.corpus_hash()
        from magra_benchmark.integrity import audit_corpus
        corpus_audit = audit_corpus(store, queries)
        index_fingerprint = _dense_fingerprint(store, runtime["dense_model"], queries)
        runtime["dense_index_hash"] = index_fingerprint["fingerprint_hash"] if index_fingerprint else None
        ann_description = None
        if approximate:
            from .ann_catalog import describe_catalog
            ann_description = describe_catalog(args.ann_catalog,store.budget,model_id=runtime["dense_model"],queries=queries)
            args._ann_description = ann_description
            runtime["dense_backend_hash"] = ann_description["identity"]["backend_hash"]
        if args.split == "eval":
            if not args.freeze:
                raise ValueError("Locked eval requires --freeze preregistration built from dev reports")
            verify_protocol(args.freeze, bundle=args.bundle, corpus_hash=corpus_hash,
                            config=asdict(config), queries=queries,
                            calibration_hash=calibrator.artifact_hash if calibrator else None, runtime=runtime)
        load_started = time.perf_counter()
        retriever = _retriever(store, config, args)
        if cuda_reranker:
            cleanup.callback(retriever.reranker.close)
            if retriever.reranker.runtime_metadata != reranker_policy:
                raise ValueError("Actual CUDA reranker differs from locked admission identity")
        model_load_latency_ms = (time.perf_counter() - load_started) * 1000
        setup_latency_ms = model_load_latency_ms
        ann_fingerprint = None
        if approximate:
            cleanup.callback(retriever.dense.close)
            from .index_integrity import effective_query_scope
            ann_fingerprint = retriever.dense.index_fingerprint(scopes=[effective_query_scope(q) for q in queries])
            model_load_latency_ms = retriever.dense.setup_observation["model_load_latency_ms"]
        generator = None
        if args.generate:
            generator = _generation_client(args, budget, settings)
        engine = MagraEngine(retriever, generator, calibrator=calibrator)
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format": "magra-run-v1", "split": args.split,
                    "created_at": datetime.now(timezone.utc).isoformat(), "corpus_hash": corpus_hash,
                    "bundle_manifest_sha256": sha256_file(Path(args.bundle) / "manifest.json"),
                    "query_ids": [q["id"] for q in queries], "config": asdict(config),
                    "corpus_audit": corpus_audit,
                    "generation": bool(generator), "model": settings.model_id if generator else None,
                    "max_tokens": generator.max_tokens if generator else None,
                    "reasoning_effort": generator.reasoning_effort if generator else None,
                    "calibration_hash": calibrator.artifact_hash if calibrator else None,
                    "dense_model": retriever.dense.model_id if retriever.dense else None,
                    "dense_index_hash": runtime["dense_index_hash"],
                    "dense_index_fingerprint": index_fingerprint,
                    "reranker_model": retriever.reranker.model_id if retriever.reranker else None,
                    "model_load_latency_ms": model_load_latency_ms,
                    "environment": environment(), "status": "running"}
        if cuda_reranker:
            metadata.update(reranker_policy, reranker_resources=retriever.reranker.resource_observations())
        if approximate:
            metadata.update(dense_backend_hash=runtime["dense_backend_hash"],
                dense_backend={"catalog_path":ann_description["catalog_path"],"identity":ann_description["identity"]},
                ann_index_fingerprint=ann_fingerprint, dense_runtime_profile=retriever.dense.runtime.profile(),
                retrieval_backend_setup_latency_ms=setup_latency_ms, backend_setup=retriever.dense.setup_observation,
                timing_policy="Engine requests exclude model/index setup and full artifact-boundary audits; per-request native thread and immutable SQL/artifact checks remain included.")
        manifest_path = Path(str(out) + ".manifest.json")
        write_json(manifest_path, metadata)
        count = 0
                                                                                                       
        with out.open("x", encoding="utf-8") as stream:
            for query in queries:
                result = engine.run(query, retrieval_only=not args.generate)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                count += 1
        final_fingerprint = _dense_fingerprint(store, runtime["dense_model"], queries)
        if final_fingerprint != index_fingerprint:
            raise ValueError("Dense index changed during the run; partial output remains ineligible")
        if approximate:
            audit_started = time.perf_counter()
            retriever.dense.runtime.validate_files()
            final_ann = retriever.dense.index_fingerprint(scopes=[effective_query_scope(q) for q in queries])
            declared = describe_catalog(args.ann_catalog,store.budget,model_id=runtime["dense_model"],queries=queries)
            if final_ann != ann_fingerprint or declared["identity"] != ann_description["identity"]:
                raise ValueError("ANN runtime/artifact/search identity changed; partial output remains ineligible")
            metadata["final_backend_audit_ms_excluded_from_requests"] = (time.perf_counter()-audit_started)*1000
        if cuda_reranker:
            retriever.reranker.close()
            metadata["reranker_resources"] = retriever.reranker.resource_observations()
            metadata["reranker_resources_scope"] = "guard historical samples/peaks, observed after explicit cleanup"
        metadata.update(status="complete", predictions_sha256=sha256_file(out),
                        resources=budget.check(), prediction_count=count)
        if generator:
            metadata["spend"] = generator.ledger.summary()
                                                                                         
        manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"predictions": str(out), "manifest": str(manifest_path), "query_count": count,
                "resources": metadata["resources"]}


def cmd_evaluate(args):
    from magra_benchmark.evaluation import evaluate_run
    metadata = load_json(str(args.predictions) + ".manifest.json")
    if metadata.get("status") != "complete" or sha256_file(args.predictions) != metadata["predictions_sha256"]:
        raise ValueError("Run incomplete or prediction file changed")
    if metadata["bundle_manifest_sha256"] != sha256_file(Path(args.bundle) / "manifest.json"):
        raise ValueError("Run was produced against a different split bundle")
    selected = set(metadata["query_ids"])
    labels = [q for q in read_split_bundle(args.bundle, metadata["split"], include_labels=True)
              if q["id"] in selected]
    report = evaluate_run(labels, read_jsonl(args.predictions), k=args.k, run_metadata=metadata)
    _budget(args).contains(args.out)
    write_json(args.out, report)
    return {"report": str(args.out), "query_count": report["query_count"],
            "missing_predictions": len(report["missing_predictions"]), "groups": report["groups"]}


def _named_paths(items):
    result = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or name in result:
            raise ValueError("Expected unique NAME=PATH arguments")
        result[name] = path
    return result


def cmd_freeze(args):
    reports = {name: load_json(path) for name, path in _named_paths(args.dev_reports).items()}
    configs = {name: asdict(RetrievalConfig(**load_json(path)))
               for name, path in _named_paths(args.configs).items()}
    calibrator_hash = None
    if args.calibration:
        from magra_benchmark.calibration import load_calibrator
        calibrator_hash = load_calibrator(args.calibration).artifact_hash
    _budget(args).contains(args.out)
    with _store(args, readonly=True) as store:
        return freeze_protocol(args.out, bundle=args.bundle, corpus_hash=store.corpus_hash(),
                               configs=configs, dev_reports=reports, baseline_names=args.baselines,
                               candidate_name=args.candidate, quality_metric=args.quality_metric,
                               limit_per_stratum=args.limit_per_stratum, calibration_hash=calibrator_hash,
                               store=store,
                               constraints={"process_peak_rss_bytes": 31_999_999_999,
                                            "peak_observed_physical_bytes": args.disk_bytes})


def cmd_calibrate(args):
    from magra_benchmark.calibration import fit_calibrator
    labels = {q["id"]: q for q in read_split_bundle(args.bundle, "dev", include_labels=True)}
    from magra_benchmark.metrics import answer_metrics
    predictions = list(read_jsonl(args.predictions))
    records = []
    for p in predictions:
        q = labels.get(p["query_id"])
        if q is None:
            raise ValueError("Calibration predictions must be dev only")
        answers = q.get("answers", [q["answer"]] if "answer" in q else None)
        if answers is None or p.get("abstained", False):
            continue
        records.append({"query_id": q["id"], "split": "dev", "confidence": p["confidence"],
                        "group_id": q.get("group_id", q["id"]),
                        "correct": bool(answer_metrics(p["answer"], answers)["answer_em"])})
    _budget(args).contains(args.out)
    result = fit_calibrator(records, provenance={"dataset_hash": json_hash(labels),
                            "split_hash": sha256_file(Path(args.bundle) / "manifest.json"),
                            "predictions_hash": sha256_file(args.predictions)},
                            output_path=args.out, target_accuracy=args.target_accuracy,
                            min_accepted=args.min_accepted)
    return result.to_dict()


def cmd_tune(args):
    from magra_benchmark.tuning import tune_retrieval
    reranker_policy = _reranker_runtime(args, args.reranker)
    cuda_reranker = "reranker_execution_hash" in reranker_policy
    out = _budget(args).contains(args.out)
    if out.exists():
        raise FileExistsError("Tuning output already exists")
    dense, reranker = None, None
    with _store(args) as store, ExitStack() as cleanup:
        queries = list(read_split_bundle(args.bundle, "dev", include_labels=True))
        from magra_benchmark.integrity import audit_corpus
        audit_corpus(store, queries)
        if args.dense:
            from .dense import DenseRetriever
            dense = DenseRetriever(store, model=args.dense_model, revision=args.dense_revision)
        if args.reranker:
            from .dense import CrossEncoderReranker
            options = {"device": args.reranker_device} if cuda_reranker else {}
            reranker = CrossEncoderReranker(store, model=args.reranker_model, revision=args.reranker_revision, **options)
            if cuda_reranker:
                cleanup.callback(reranker.close)
                if reranker.runtime_metadata != reranker_policy:
                    raise ValueError("CUDA tuner reranker differs from pre-load identity")
        result = tune_retrieval(store, queries,
                                bundle_manifest_sha256=sha256_file(Path(args.bundle) / "manifest.json"),
                                dense=dense, reranker=reranker, limit_per_stratum=args.limit_per_stratum,
                                constraints={"process_peak_rss_bytes": 31_999_999_999,
                                             "peak_observed_physical_bytes": args.disk_bytes})
        if cuda_reranker:
            reranker.close()
            observation = reranker.resource_observations()
            for report in result["development_reports"].values():
                metadata = report["run_metadata"]
                if metadata.get("reranker_model") is not None:
                    if metadata["reranker_model"] != reranker_policy["reranker_model"]:
                        raise ValueError("Tuner report CUDA reranker identity differs")
                    metadata.update(reranker_policy, reranker_resources=observation,
                                    reranker_resources_scope="shared tuner guard across all configs; historical samples after cleanup")
            manifest = result["manifest"]
            manifest["backend_inventory"].update(reranker_policy)
            manifest["report_hashes"] = {name:json_hash(report) for name, report in result["development_reports"].items()}
            manifest.pop("manifest_hash", None)
            manifest["manifest_hash"] = json_hash(manifest)
        out.mkdir(parents=True)
        for name, report in result["development_reports"].items():
            write_json(out / (name + ".report.json"), report)
            write_json(out / (name + ".config.json"), result["configs"][name])
        write_json(out / "tuning.json", result)
        return {"directory": str(out), "candidates": list(result["configs"]),
                "baseline_selection": result["baseline_selection"]}


def cmd_compare(args):
    from magra_benchmark.evaluation import compare_runs
    frozen = load_json(args.freeze)
    prediction_paths = _named_paths(args.runs)
    dev_reports = {name: load_json(path) for name, path in _named_paths(args.dev_reports).items()}
    if set(dev_reports) != set(frozen["development_report_hashes"]):
        raise ValueError("Development reports differ from frozen baseline selection")
    if any(json_hash(r) != frozen["development_report_hashes"][n] for n, r in dev_reports.items()):
        raise ValueError("Development report contents differ from preregistration")
    with _store(args, readonly=True) as store:
        corpus_hash = store.corpus_hash()
    queries = sample_queries(read_split_bundle(args.bundle, "eval", include_labels=True),
                             frozen["limit_per_stratum"])
    public = sample_queries(read_split_bundle(args.bundle, "eval"), frozen["limit_per_stratum"])
    metadata, runs = {}, {}
    for name, path in prediction_paths.items():
        meta = load_json(path + ".manifest.json")
        if meta.get("status") != "complete" or sha256_file(path) != meta.get("predictions_sha256"):
            raise ValueError("Incomplete or modified comparison run")
        runtime = runtime_identity(meta)
        verify_protocol(args.freeze, bundle=args.bundle, corpus_hash=corpus_hash,
                        config=meta["config"], queries=public,
                        calibration_hash=meta.get("calibration_hash"), runtime=runtime)
        if meta["config"] != frozen["configs"][name] or meta.get("split") != "eval":
            raise ValueError("Comparison name, configuration or split mismatch")
        metadata[name], runs[name] = meta, list(read_jsonl(path))
    result = compare_runs(queries, runs, baseline_names=frozen["baseline_names"],
                          candidate_name=frozen["candidate_name"], development_reports=dev_reports,
                          quality_metric=frozen["quality_metric"], run_metadata=metadata,
                          constraints=frozen["constraints"])
    _budget(args).contains(args.out)
    write_json(args.out, result)
    return {"comparison": str(args.out), "generality_claim_supported": False}


def cmd_failures(args):
    from magra_benchmark.robustness import build_failure_suite
    budget = _budget(args)
    out = budget.contains(args.out)
    if out.exists():
        raise FileExistsError("Failure suite output already exists")
    queries, documents, provenance = [], [], []
    for seed in range(args.seeds):
        for case in build_failure_suite(seed=seed, as_of=args.as_of):
            label = case["labels"]
            documents.extend(case["documents"])
            q = {key: value for key, value in case["query"].items() if key != "text"}
            q.update(source="magra_synthetic", scenario=case["case"], synthetic=True,
                     allowed_doc_ids=[d["id"] for d in case["documents"]],
                     answerable=label["answerable"], expected_behavior=label["expected_behavior"],
                     relevance={doc_id: 1 for doc_id in label["relevant_ids"]},
                     supporting_evidence=[{"doc_id": doc_id, "sentence_id": 0}
                                          for doc_id in label["supporting_ids"]])
            if label["expected_answer"] is not None:
                q["answers"] = label.get("accepted_answers", [label["expected_answer"]])
            queries.append(q)
            provenance.append({"case_id": case["case_id"], "provenance": case["provenance"],
                               "labels": label, "excluded": case["overlay"]["exclude_ids"]})
    out.mkdir(parents=True)
    with DocumentStore(out / "corpus.sqlite", budget=budget) as store:
        imported = store.ingest(documents)
    write_json(out / "private_provenance.json", provenance)
    manifest = write_split_bundle(out / "benchmark", make_splits(queries, heldout_sources=()),
                                  provenance={"synthetic": True, "generator": "magra.synthetic_failures.v1",
                                              "seeds": args.seeds, "as_of": args.as_of,
                                              "candidate_isolation": "allowlist applied before SQL retrieval"})
    return {"directory": str(out), "documents": imported["total_documents"], "splits": manifest["counts"],
            "synthetic": True, "natural_coverage": False}


def cmd_web(args):
    from .web import BoundedWebClient, SnapshotStore, LiveWebTrack
    budget = _budget(args)
    root = budget.contains(args.out)
    client = BoundedWebClient(args.allowed_domains, max_requests=args.max_requests,
                               max_bytes=args.max_bytes, max_response_bytes=min(2_000_000, args.max_bytes),
                               timeout=args.timeout)
    budget.check(args.max_bytes * 3 + 65536)
    if args.command == "web-snapshot":
        doc = SnapshotStore(root, max_bytes=args.max_bytes * 3 + 65536).capture(args.url, client)
        return {"document": doc, "resources": budget.check()}
    track = LiveWebTrack(client)
    rows = [track.acquire(url) for url in args.urls]
    write_json(root, {"track": "live_web", "acquisitions": rows, "summary": track.summary()})
    return {"path": str(root), "summary": track.summary(), "resources": budget.check()}


def _audit_judgments(paths):
    pass                                                                         
    for path in paths:
        manifest_path = Path(str(path) + ".manifest.json")
        rows = list(read_jsonl(path))
        if manifest_path.exists():
            metadata = load_json(manifest_path)
            if metadata.get("status") != "complete" or metadata.get("judgments_sha256") != sha256_file(path):
                raise ValueError("Judge run incomplete or judgment file changed")
            ids = [row.get("query_id") for row in rows]
            if (len(rows) != metadata.get("judgment_count") or len(set(ids)) != len(ids)
                    or set(ids) != set(metadata.get("query_ids", []))):
                raise ValueError("Judgment rows disagree with completed run manifest")
        yield from rows


def cmd_judge_fit_dev(args):
    from magra_benchmark.evaluation import fit_judge_dev
    out = _budget(args).contains(args.out)
    result = fit_judge_dev(read_jsonl(args.human_labels), _audit_judgments(args.judgments),
                           min_per_class=args.min_per_class, min_coverage=args.min_coverage)
    write_json(out, result)
    return {"path": str(out), "status": result["status"],
            "locked_judge_hash": result["locked_judge_hash"], "use_for_acceptance": False}


def cmd_judge_audit(args):
    from magra_benchmark.evaluation import evaluate_judge_calibration
    result = evaluate_judge_calibration(read_jsonl(args.human_labels), _audit_judgments(args.judgments),
                                         min_per_class=args.min_per_class,
                                         min_coverage=args.min_coverage,
                                         min_balanced_accuracy=args.min_balanced_accuracy,
                                         frozen_dev_fit=args.frozen_dev_fit)
    _budget(args).contains(args.out)
    write_json(args.out, result)
    return {"path": str(args.out), "use_for_acceptance": result.get("use_for_acceptance", False)}


def _judge_client(args, budget):
    return _generation_client(args, budget)


def cmd_judge(args):
    from magra_benchmark.judge_workflow import judge_run
    budget = _budget(args)
    return judge_run(bundle=args.bundle, predictions=args.predictions, out=args.out,
                     client=_judge_client(args, budget), budget=budget,
                     min_request_interval_seconds=args.min_request_interval)


def cmd_judge_inputs(args):
    from magra_benchmark.judge_workflow import judge_inputs
    budget = _budget(args)
    return judge_inputs(inputs=args.inputs, expected_sha256=args.inputs_sha256, out=args.out,
                        split=args.split, input_manifest=args.input_manifest,
                        client=_judge_client(args, budget), budget=budget,
                        min_request_interval_seconds=args.min_request_interval)


def cmd_prepare_judge_audit(args):
    from magra_benchmark.ragtruth import prepare_ragtruth_audit
    if args.all_tasks:
        if not args.exclude_audit:
            raise ValueError("--all-tasks requires --exclude-audit to keep the new audit independent")
        from magra_benchmark.ragtruth import prepare_ragtruth_broad_audit
        return prepare_ragtruth_broad_audit(
            args.source, args.out, _budget(args), exclude_audit=args.exclude_audit,
            dev_per_class=args.dev_per_class, eval_per_class=args.eval_per_class,
            seed=args.seed or "magra-ragtruth-broad-v1")
    if args.exclude_audit:
        raise ValueError("--exclude-audit requires --all-tasks")
    return prepare_ragtruth_audit(args.source, args.out, _budget(args),
                                 dev_per_class=args.dev_per_class,
                                 eval_per_class=args.eval_per_class, seed=args.seed or "magra-ragtruth-v1")


def cmd_audit(args):
    from magra_benchmark.integrity import audit_corpus
    with _store(args, readonly=True) as store:
        return audit_corpus(store, read_split_bundle(args.bundle, args.split), strict=not args.allow_incomplete)


def build_parser():
    parser = SourceDocumentedParser(prog="magra", source_path="magra_rag/cli.py")
    parser.add_argument("--data-root", default="data", help="Account all corpora, models, logs and artifacts here")
    parser.add_argument("--db", help="SQL corpus path inside data root (default data/corpus.sqlite)")
    parser.add_argument("--disk-bytes", type=int, default=2_000_000_000_000)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    _paid_options(p)
    p.add_argument("--env-file", default=".env", help="Local credentials/configuration; never printed")
    p.add_argument("--model", help="Exact OpenRouter model ID; overrides OPENROUTER_MODEL")
    p.add_argument("--max-usd", type=float, help="Paid UTC monthly cap defaults to 100 USD and cannot exceed 100")
    p.add_argument("--prompt-price", type=float)
    p.add_argument("--completion-price", type=float)
    p.add_argument("--check-openrouter", action="store_true", help="Read-only live catalog and account quota checks")
    p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("download")
    p.add_argument("datasets", nargs="*", default=[], help="BEIR names: scifact nfcorpus fiqa arguana")
    p.add_argument("--archive", help="Verified local BEIR ZIP, useful with platform-native TLS download")
    p.add_argument("--hotpot", help="Official Hotpot acquisition key (see README)")
    p.add_argument("--ragtruth", action="store_true", help="Pinned public human-annotation dataset for the separate judge audit")
    p.add_argument("--max-download-bytes", type=int, default=200_000_000)
    p.add_argument("--max-extracted-bytes", type=int, default=2_000_000_000)
    p.set_defaults(func=cmd_download)
    p = sub.add_parser("ingest")
    p.add_argument("--format", choices=["beir", "hotpot_context", "hotpot_wiki", "jsonl"], required=True)
    p.add_argument("path")
    p.add_argument("--batch-size", type=int, default=2048, help="Bounded SQL import transaction size")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("prepare")
    p.add_argument("--beir", nargs="*", default=[])
    p.add_argument("--queries", nargs="*", default=[], help="Owned natural or failure-task labeled JSONL")
    p.add_argument("--hotpot-train", nargs="*", default=[])
    p.add_argument("--hotpot-dev", nargs="*", default=[])
    p.add_argument("--hotpot-settings", nargs="+", choices=["provided_context", "full_corpus"],
                   default=["provided_context", "full_corpus"])
    p.add_argument("--heldout-sources", nargs="*", default=["arguana"])
    p.add_argument("--duplicate-threshold", type=float, default=0.85)
    p.add_argument("--partition-hotpot-dev", action="store_true",
                   help="Split official labeled Hotpot dev groups between internal dev and locked eval")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("dense-index")
    p.add_argument("--dense-model", choices=[DEFAULT_DENSE_MODEL, QWEN_DENSE_MODEL], default=DEFAULT_DENSE_MODEL)
    p.add_argument("--dense-revision", help="Pin an HF commit for reproducibility")
    p.add_argument("--batch-size", type=int, help="Embedding page size; BGE defaults16, native Qwen defaults1")
    p.add_argument("--sources", nargs="+", help="Optional complete source names to encode")
    p.set_defaults(func=cmd_dense_index)
    p = sub.add_parser("ann-build", help="Build a fresh derived HNSW artifact; canonical SQL remains read-only")
    p.add_argument("--dense-model", choices=[DEFAULT_DENSE_MODEL,QWEN_DENSE_MODEL], default=DEFAULT_DENSE_MODEL)
    p.add_argument("--dense-revision", required=True, help="Exact embedding commit used by existing SQL vectors")
    p.add_argument("--ann-runtime", required=True, help="Local JSON with explicit site/native/wrapper pins")
    p.add_argument("--source", help="Exact source; omitted means one full-corpus artifact")
    p.add_argument("--id-prefix")
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=512)
    p.add_argument("--max-index-bytes", type=int, default=24_000_000_000)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_ann_build)
    p = sub.add_parser("run")
    p.add_argument("--dense-backend", choices=["exact","hnsw"], default="exact")
    p.add_argument("--ann-catalog", help="Explicit published artifacts/scopes, runtime pins, efSearch and internal candidates")
    _paid_options(p)
    p.add_argument("--bundle", required=True)
    p.add_argument("--split", choices=["train", "dev", "eval"], default="dev")
    p.add_argument("--strategy", default="magra")
    p.add_argument("--config")
    p.add_argument("--limit-per-stratum", type=int)
    p.add_argument("--dense-model", choices=[DEFAULT_DENSE_MODEL, QWEN_DENSE_MODEL], default=DEFAULT_DENSE_MODEL)
    p.add_argument("--reranker-model", choices=[DEFAULT_RERANKER_MODEL, QWEN_RERANKER_MODEL], default=DEFAULT_RERANKER_MODEL)
    p.add_argument("--dense-revision")
    p.add_argument("--reranker-revision")
    p.add_argument("--reranker-device", choices=["cpu", "cuda:0"], default="cpu",
                   help="Explicit native Qwen CUDA FP16 reranking; CPU remains default")
    p.add_argument("--calibration")
    p.add_argument("--freeze")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--env-file", default=".env", help="Local credentials/configuration; never printed")
    p.add_argument("--model", help="Exact OpenRouter model ID; overrides OPENROUTER_MODEL")
    p.add_argument("--max-output-tokens", type=int, help="Paid default4096; legacy default512")
    p.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                   help="Explicit model reasoning policy; frozen with generation settings")
    p.add_argument("--max-usd", type=float, help="Legacy cumulative cap; paid UTC monthly default 100, maximum 100")
    p.add_argument("--prompt-price", type=float, help="Paid exact0.15; legacy default0 USD per million")
    p.add_argument("--completion-price", type=float, help="Paid exact0.60; legacy default0 USD per million")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("evaluate")
    p.add_argument("--bundle", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("freeze")
    p.add_argument("--bundle", required=True)
    p.add_argument("--dev-reports", nargs="+", required=True, help="NAME=PATH pairs")
    p.add_argument("--configs", nargs="+", required=True, help="NAME=PATH pairs")
    p.add_argument("--baselines", nargs="+", required=True)
    p.add_argument("--candidate", default="magra")
    p.add_argument("--quality-metric", default="ndcg@10")
    p.add_argument("--limit-per-stratum", type=int)
    p.add_argument("--calibration")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_freeze)
    p = sub.add_parser("calibrate")
    p.add_argument("--bundle", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--target-accuracy", type=float, default=0.9)
    p.add_argument("--min-accepted", type=int, default=20)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("tune")
    p.add_argument("--bundle", required=True)
    p.add_argument("--dense", action="store_true")
    p.add_argument("--reranker", action="store_true")
    p.add_argument("--dense-model", choices=[DEFAULT_DENSE_MODEL, QWEN_DENSE_MODEL], default=DEFAULT_DENSE_MODEL)
    p.add_argument("--reranker-model", choices=[DEFAULT_RERANKER_MODEL, QWEN_RERANKER_MODEL], default=DEFAULT_RERANKER_MODEL)
    p.add_argument("--dense-revision")
    p.add_argument("--reranker-revision")
    p.add_argument("--reranker-device", choices=["cpu", "cuda:0"], default="cpu",
                   help="Explicit native Qwen CUDA FP16 reranking; CPU remains default")
    p.add_argument("--limit-per-stratum", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_tune)
    p = sub.add_parser("compare")
    p.add_argument("--bundle", required=True)
    p.add_argument("--freeze", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="NAME=PATH prediction pairs")
    p.add_argument("--dev-reports", nargs="+", required=True, help="NAME=PATH report pairs")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("failures")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--as-of", default="2026-10-20")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_failures)
    for command in ("web-snapshot", "web-live"):
        p = sub.add_parser(command)
        if command == "web-snapshot":
            p.add_argument("url")
        else:
            p.add_argument("urls", nargs="+")
        p.add_argument("--allowed-domains", nargs="+", required=True)
        p.add_argument("--max-requests", type=int, default=8)
        p.add_argument("--max-bytes", type=int, default=10_000_000)
        p.add_argument("--timeout", type=float, default=10)
        p.add_argument("--out", required=True)
        p.set_defaults(func=cmd_web)
    p = sub.add_parser("judge-audit")
    p.add_argument("--human-labels", required=True)
    p.add_argument("--judgments", nargs="+", required=True)
    p.add_argument("--min-per-class", type=int, default=5)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--min-balanced-accuracy", type=float, default=0.8)
    p.add_argument("--frozen-dev-fit", help="Hash-bound judge-fit-dev artifact created before evaluation")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_judge_audit)
    p = sub.add_parser("judge-fit-dev")
    p.add_argument("--human-labels", required=True, help="Development labels only; evaluation rows are rejected")
    p.add_argument("--judgments", nargs="+", required=True)
    p.add_argument("--min-per-class", type=int, default=5)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_judge_fit_dev)
    for command in ("judge", "judge-inputs"):
        p = sub.add_parser(command)
        if command == "judge":
            p.add_argument("--bundle", required=True)
            p.add_argument("--predictions", required=True)
            p.set_defaults(func=cmd_judge)
        else:
            p.add_argument("--inputs", required=True, help="Public query/prediction/evidence JSONL; no labels")
            p.add_argument("--inputs-sha256", required=True, help="Pin the public input bytes before inference")
            p.add_argument("--input-manifest", help="Optional public adapter manifest binding this input file")
            p.add_argument("--split", choices=["dev", "eval"])
            p.set_defaults(func=cmd_judge_inputs)
        _paid_options(p)
        p.add_argument("--env-file", default=".env")
        p.add_argument("--model", help="Exact OpenRouter judge model ID")
        p.add_argument("--max-output-tokens", type=int)
        p.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                       help="Explicit reasoning policy; use low to reserve room for structured output")
        p.add_argument("--min-request-interval", type=float, default=5,
                       help="Seconds between judge request starts (default 5; no automatic retries)")
        p.add_argument("--max-usd", type=float, help="Optional project USD cap")
        p.add_argument("--prompt-price", type=float)
        p.add_argument("--completion-price", type=float)
        p.add_argument("--out", required=True)
    p = sub.add_parser("prepare-judge-audit")
    p.add_argument("--source", required=True, help="Verified local publisher RAGTruth acquisition directory")
    p.add_argument("--all-tasks", action="store_true", help="Include QA, two news sources, and Yelp; counts are per stratum")
    p.add_argument("--exclude-audit", help="Existing QA audit whose complete context groups must be excluded")
    p.add_argument("--dev-per-class", type=int, default=40, help="Items per human class (and per stratum with --all-tasks)")
    p.add_argument("--eval-per-class", type=int, default=40, help="Items per human class (and per stratum with --all-tasks)")
    p.add_argument("--seed", help="Deterministic selection seed; defaults to the adapter's versioned seed")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_prepare_judge_audit)
    p = sub.add_parser("audit")
    p.add_argument("--bundle", required=True)
    p.add_argument("--split", choices=["train", "dev", "eval"], default="dev")
    p.add_argument("--allow-incomplete", action="store_true")
    p.set_defaults(func=cmd_audit)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(f"magra: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
