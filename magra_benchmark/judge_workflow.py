                                                                               

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

from magra_rag.generation import MODEL, IncompleteResponseError, OpenRouterM3, PaidHTTPError, RateLimitError
from magra_rag.resources import ResourceBudget

from .datasets import sha256_file
from .judge import (
    MAX_INPUT_BYTES, SYSTEM, JudgeError, M3GroundednessJudge, _inputs, judge_runtime,
)
from .protocol import environment, json_hash
from .splits import PUBLIC_FIELDS, SPLIT_NAMES, read_split_bundle, verify_split_bundle


                                                                             
                                                                              
MAX_PREDICTION_FILE_BYTES = 256_000_000
MAX_MANIFEST_BYTES = 16_000_000
MAX_JUDGMENT_ROW_BYTES = 128_000
MAX_AUDIT_INPUT_BYTES = 64_000_000
MAX_AUDIT_INPUT_ROWS = 1000


class JudgeWorkflowError(ValueError):
    pass                                                            


def _now():
    return datetime.now(timezone.utc).isoformat()


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JudgeWorkflowError("Duplicate fields in judge input artifact")
        result[key] = value
    return result


def _constant(_):
    raise JudgeWorkflowError("Nonfinite values in judge input artifact")


def _decode(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _validated_inputs(bundle, predictions, client):
    manifest_path = Path(str(predictions) + ".manifest.json")
    if predictions.is_symlink() or manifest_path.is_symlink():
        raise JudgeWorkflowError("Judge prediction artifacts cannot be symbolic links")
    if predictions.stat().st_size > MAX_PREDICTION_FILE_BYTES:
        raise JudgeWorkflowError("Prediction artifact exceeds the 256 MB judge batch bound")
    with manifest_path.open("rb") as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise JudgeWorkflowError("Prediction manifest exceeds judge bound")
    metadata = _decode(raw)
    if (not isinstance(metadata, dict) or metadata.get("format") != "magra-run-v1"
            or metadata.get("status") != "complete" or metadata.get("generation") is not True):
        raise JudgeWorkflowError("Judge requires a complete generated prediction run")
    prediction_hash = sha256_file(predictions)
    manifest_hash = sha256_file(manifest_path)
    if not _digest(metadata.get("predictions_sha256")) or prediction_hash != metadata["predictions_sha256"]:
        raise JudgeWorkflowError("Prediction file hash differs from complete run manifest")
    bundle_hash = metadata.get("bundle_manifest_sha256")
    if not _digest(bundle_hash) or sha256_file(bundle / "manifest.json") != bundle_hash:
        raise JudgeWorkflowError("Prediction run belongs to a different or changed bundle")
    split = metadata.get("split")
    if split not in SPLIT_NAMES:
        raise JudgeWorkflowError("Prediction run requires an explicit train/dev/eval split")
    identifiers = metadata.get("query_ids")
    if (not isinstance(identifiers, list) or not identifiers
            or any(not isinstance(qid, str) or not qid.strip()
                   or len(qid.encode("utf-8")) > 2048 for qid in identifiers)
            or len(identifiers) != len(set(identifiers))):
        raise JudgeWorkflowError("Prediction manifest query IDs must be unique and nonempty")
    if (type(metadata.get("prediction_count")) is not int
            or metadata["prediction_count"] != len(identifiers)):
        raise JudgeWorkflowError("Prediction manifest count differs from query IDs")
    selected = set(identifiers)
    queries = {}
                                                                            
                                                                          
    for query in read_split_bundle(bundle, split, include_labels=False,
                                   expected_manifest_sha256=bundle_hash):
        if not isinstance(query, dict) or set(query) - PUBLIC_FIELDS:
            raise JudgeWorkflowError("Bundle contains a nonpublic query record")
        qid = query.get("id")
        if not isinstance(qid, str) or not qid:
            raise JudgeWorkflowError("Bundle query requires a nonempty ID")
        if qid in queries:
            raise JudgeWorkflowError("Bundle query IDs are duplicated")
                                                                            
        queries[qid] = query if qid in selected else None
    if not selected <= queries.keys():
        raise JudgeWorkflowError("Prediction manifest query IDs are absent from its public split")
    prepared, seen, consumed = [], set(), 0
    with predictions.open("rb") as stream:
        while raw := stream.readline(MAX_INPUT_BYTES + 1):
            consumed += len(raw)
            if consumed > MAX_PREDICTION_FILE_BYTES:
                raise JudgeWorkflowError("Prediction artifact exceeds the 256 MB judge batch bound")
            if len(raw) > MAX_INPUT_BYTES:
                raise JudgeWorkflowError("Prediction row exceeds judge byte bound")
            if not raw.strip():
                continue
            row = _decode(raw)
            if not isinstance(row, dict):
                raise JudgeWorkflowError("Prediction rows must be objects")
            qid = row.get("query_id")
            if not isinstance(qid, str) or qid not in selected or qid in seen:
                raise JudgeWorkflowError("Prediction rows contain duplicate or extra query IDs")
            seen.add(qid)
            if not isinstance(row.get("evidence"), list):
                raise JudgeWorkflowError("Generated predictions must retain their selected evidence")
            query = queries[qid]
            if query.get("split", split) != split:
                raise JudgeWorkflowError("Public query split disagrees with prediction manifest")
            query.setdefault("split", split)
            group = query.get("group_id", qid)
            if not isinstance(group, str) or not group.strip() or len(group.encode("utf-8")) > 2048:
                raise JudgeWorkflowError("Public query duplicate group must be a nonempty string")
            _, prediction, evidence, _, original = _inputs(query, row, row["evidence"], MAX_INPUT_BYTES)
            if client.key and client.key in original:
                raise JudgeWorkflowError("Credential material is forbidden in judge input")
            prepared.append((query, prediction, evidence))
    if seen != selected:
        raise JudgeWorkflowError("Prediction rows are missing manifest query IDs")
    if sha256_file(predictions) != prediction_hash or sha256_file(manifest_path) != manifest_hash:
        raise JudgeWorkflowError("Prediction artifacts changed during validation")
    if sha256_file(bundle / "manifest.json") != bundle_hash:
        raise JudgeWorkflowError("Bundle changed during validation")
    return metadata, prepared, prediction_hash, manifest_hash, bundle_hash


def _replace_manifest(path, metadata, budget):
    pass                                                                      
    raw = _encoded(metadata)
    with tempfile.NamedTemporaryFile(prefix=".judge-manifest-", suffix=".tmp", dir=path.parent,
                                     delete=False) as stream:
        temporary = budget.contains(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failure(error):
    result = {"type": type(error).__name__}
    if isinstance(error, PaidHTTPError):
        result.update(status_code=error.status_code, goal_stop_required=error.status_code == 402)
    if isinstance(error, IncompleteResponseError):
        result.update(error.as_dict())
    if isinstance(error, RateLimitError):
        result["status_code"] = 429
        retry = error.retry_after_seconds
        result["retry_after_seconds"] = retry if (type(retry) in (int, float)
                                                  and math.isfinite(retry) and retry >= 0) else None
        result["rate_limit"] = {key: value for key, value in error.rate_limit.items()
                                if key in {"x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
                                and type(value) in (int, float) and math.isfinite(value) and value >= 0}
    return result


def _interval(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 60):
        raise JudgeWorkflowError("Judge request interval must be finite and between 0 and 60 seconds")
    return float(value)


def judge_run(*, bundle, predictions, out, client: OpenRouterM3, budget: ResourceBudget,
              min_request_interval_seconds=0) -> dict:
    pass                                                                         

                                                                               
                                                                                 
                                                                                
                                                                               
                                                                          
       
    interval = _interval(min_request_interval_seconds)
    out = budget.contains(out)
    manifest_path = budget.contains(str(out) + ".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError("Judge outputs already exist; choose a new path")
                                                                               
    getattr(client.ledger, "budget", budget).contains(client.ledger.path)
    bundle, predictions = Path(bundle), Path(predictions)
    try:
        source, prepared, prediction_hash, source_manifest_hash, bundle_hash = _validated_inputs(
            bundle, predictions, client)
    except (JudgeWorkflowError, JudgeError):
        raise
    except Exception:
        raise JudgeWorkflowError("Cannot validate judge input artifacts") from None
    namespace = "prediction:" + prediction_hash + ":"
    source_ids = [query["id"] for query, _, _ in prepared]
    metadata = {"source_kind": "generated_run", "split": source["split"], "query_count": len(prepared),
                "judgment_count": 0, "query_ids": [namespace + qid for qid in source_ids],
                "source_query_ids": source_ids, "observation_namespace": namespace,
                "predictions_sha256": prediction_hash, "prediction_manifest_sha256": source_manifest_hash,
                "bundle_manifest_sha256": bundle_hash,
                "public_queries_sha256": json_hash([query for query, _, _ in prepared])}

    def verify_sources():
        if (sha256_file(predictions) != prediction_hash
                or sha256_file(Path(str(predictions) + ".manifest.json")) != source_manifest_hash):
            raise JudgeWorkflowError("Source artifacts changed while judgments were running")
        verify_split_bundle(bundle, expected_manifest_sha256=bundle_hash)

    return _execute(prepared=prepared, out=out, manifest_path=manifest_path, client=client,
                    budget=budget, metadata=metadata, identity_prefix=namespace,
                    verify_sources=verify_sources, interval=interval)


def judge_inputs(*, inputs, out, client: OpenRouterM3, budget: ResourceBudget,
                 expected_sha256: str, split=None, input_manifest=None,
                 min_request_interval_seconds=0) -> dict:
    pass                                                                         

                                                                              
                                                                                
                                                                       
       
    interval = _interval(min_request_interval_seconds)
    out = budget.contains(out)
    manifest_path = budget.contains(str(out) + ".manifest.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError("Judge outputs already exist; choose a new path")
    getattr(client.ledger, "budget", budget).contains(client.ledger.path)
    if not _digest(expected_sha256):
        raise JudgeWorkflowError("Public judge inputs require a pinned SHA256")
    if split is not None and split not in SPLIT_NAMES:
        raise JudgeWorkflowError("Judge input split must be train/dev/eval")
    inputs = Path(inputs)
    try:
        if inputs.is_symlink() or inputs.stat().st_size > MAX_AUDIT_INPUT_BYTES:
            raise JudgeWorkflowError("Public judge input exceeds the 64 MB batch bound or is a link")
        if sha256_file(inputs) != expected_sha256:
            raise JudgeWorkflowError("Public judge input SHA256 mismatch")
        public_manifest_hash = None
        if input_manifest is not None:
            input_manifest = Path(input_manifest)
            if input_manifest.is_symlink():
                raise JudgeWorkflowError("Public input manifest cannot be a symbolic link")
            with input_manifest.open("rb") as stream:
                raw = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(raw) > MAX_MANIFEST_BYTES:
                raise JudgeWorkflowError("Public input manifest exceeds bound")
            public_manifest = _decode(raw)
            if (not isinstance(public_manifest, dict)
                    or public_manifest.get("format") != "magra-ragtruth-public-inputs-v1"
                    or public_manifest.get("files", {}).get(inputs.name, {}).get("sha256") != expected_sha256):
                raise JudgeWorkflowError("Public input manifest does not bind these inputs")
            public_manifest_hash = sha256_file(input_manifest)
        prepared, seen, groups, consumed = [], set(), {}, 0
        with inputs.open("rb") as stream:
            while raw := stream.readline(MAX_INPUT_BYTES + 1):
                consumed += len(raw)
                if consumed > MAX_AUDIT_INPUT_BYTES:
                    raise JudgeWorkflowError("Public judge input exceeds the 64 MB batch bound")
                if len(raw) > MAX_INPUT_BYTES:
                    raise JudgeWorkflowError("Public judge input row exceeds byte bound")
                if not raw.strip():
                    continue
                if len(prepared) >= MAX_AUDIT_INPUT_ROWS:
                    raise JudgeWorkflowError("Public judge inputs exceed the 1000 record batch bound")
                record = _decode(raw)
                if not isinstance(record, dict) or set(record) != {"query", "prediction", "evidence"}:
                    raise JudgeWorkflowError("Public judge rows require exactly query/prediction/evidence")
                query, prediction, evidence, _, original = _inputs(
                    record["query"], record["prediction"], record["evidence"], MAX_INPUT_BYTES)
                if client.key and client.key in original:
                    raise JudgeWorkflowError("Credential material is forbidden in judge input")
                qid = query["id"]
                if qid in seen:
                    raise JudgeWorkflowError("Public judge query IDs must be unique")
                seen.add(qid)
                effective_split = query.get("split", split)
                if effective_split not in SPLIT_NAMES or (split is not None and effective_split != split):
                    raise JudgeWorkflowError("Public query split is missing or disagrees with requested split")
                query["split"] = effective_split
                group = query.get("group_id", qid)
                if not isinstance(group, str) or not group.strip() or len(group.encode("utf-8")) > 2048:
                    raise JudgeWorkflowError("Public query duplicate group must be a nonempty string")
                if group in groups and groups[group] != effective_split:
                    raise JudgeWorkflowError("Public duplicate group crosses judge input splits")
                groups[group] = effective_split
                prepared.append((query, prediction, evidence))
        if not prepared:
            raise JudgeWorkflowError("Public judge input batch is empty")
        if input_manifest is not None:
            declared = public_manifest["files"][inputs.name].get("records")
            if type(declared) is not int or declared != len(prepared):
                raise JudgeWorkflowError("Public input manifest record count mismatch")
        if sha256_file(inputs) != expected_sha256:
            raise JudgeWorkflowError("Public judge input changed during validation")
    except (JudgeWorkflowError, JudgeError):
        raise
    except Exception:
        raise JudgeWorkflowError("Cannot validate public judge inputs") from None
    identifiers = [query["id"] for query, _, _ in prepared]
    metadata = {"source_kind": "public_inputs", "inputs_sha256": expected_sha256,
                "input_manifest_sha256": public_manifest_hash, "query_count": len(prepared),
                "judgment_count": 0, "query_ids": identifiers, "source_query_ids": identifiers,
                "observation_namespace": None, "split": split,
                "splits": sorted({query["split"] for query, _, _ in prepared}),
                "public_queries_sha256": json_hash([query for query, _, _ in prepared])}

    def verify_sources():
        if (sha256_file(inputs) != expected_sha256 or (input_manifest is not None
                and sha256_file(input_manifest) != public_manifest_hash)):
            raise JudgeWorkflowError("Public input artifacts changed while judgments were running")

    return _execute(prepared=prepared, out=out, manifest_path=manifest_path, client=client,
                    budget=budget, metadata=metadata, identity_prefix="", verify_sources=verify_sources,
                    interval=interval)


def _execute(*, prepared, out, manifest_path, client, budget, metadata, identity_prefix, verify_sources,
             interval):
    judge = M3GroundednessJudge(client)
    metadata.update({"format": "magra-judge-run-v1", "schema_version": 1, "status": "running",
                "created_at": _now(), "model": getattr(client, "model_id", MODEL), "max_tokens": client.max_tokens,
                "reasoning_effort": getattr(client, "reasoning_effort", None),
                "runtime": judge_runtime(client), "runtime_sha256": json_hash(judge_runtime(client)),
                "prompt_usd_per_million": client.prompt_price,
                "completion_usd_per_million": client.output_price,
                "min_request_interval_seconds": interval,
                "rubric_sha256": json_hash(SYSTEM), "calibration_status": "not_human_audited",
                "environment": environment(), "spend_before": client.ledger.summary()})
    metadata.update(spend=metadata["spend_before"], judgments_sha256=None)
    metadata["resources"] = budget.check(len(_encoded(metadata)) * 2 + MAX_JUDGMENT_ROW_BYTES)
    out.parent.mkdir(parents=True, exist_ok=True)
                                                                              
    with manifest_path.open("xb") as stream:
        stream.write(_encoded(metadata))
    active_source_query_id = None
    previous_started = None
    try:
        with out.open("xb") as stream:
            for query, prediction, evidence in prepared:
                active_source_query_id = query["id"]
                metadata["resources"] = budget.check(MAX_JUDGMENT_ROW_BYTES + len(_encoded(metadata)) * 2)
                before_wait = time.monotonic()
                delay = max(0.0, interval - (before_wait - previous_started)) if previous_started is not None else 0.0
                if delay:
                    time.sleep(delay)
                previous_started = time.monotonic()
                queue_wait_ms = max(0.0, (previous_started - before_wait) * 1000) if delay else 0.0
                result = judge.judge(query, prediction, evidence)
                original_id = query["id"]
                identity = identity_prefix + original_id
                result.update(query_id=identity, observation_id=identity, source_query_id=original_id,
                              group_id=query.get("group_id", original_id), split=query["split"],
                              queue_wait_ms=queue_wait_ms,
                              confidence_kind="uncalibrated_model_self_report",
                              calibration_status="not_human_audited")
                if metadata["source_kind"] == "generated_run":
                    result["source_predictions_sha256"] = metadata["predictions_sha256"]
                else:
                    result["source_inputs_sha256"] = metadata["inputs_sha256"]
                for key in ("source", "domain", "setting", "track", "scenario", "synthetic"):
                    if key in query:
                        result[key] = query[key]
                row = _encoded(result)
                if len(row) > MAX_JUDGMENT_ROW_BYTES:
                    raise JudgeWorkflowError("Judgment row exceeds workflow output bound")
                stream.write(row)
                stream.flush()
                os.fsync(stream.fileno())
                metadata["judgment_count"] += 1
                metadata.update(updated_at=_now(), judgments_sha256=sha256_file(out),
                                spend=client.ledger.summary())
                _replace_manifest(manifest_path, metadata, budget)
        verify_sources()
        metadata.update(status="complete", completed_at=_now(), judgments_sha256=sha256_file(out),
                        resources=budget.check(), spend=client.ledger.summary())
        _replace_manifest(manifest_path, metadata, budget)
    except BaseException as error:
        metadata.update(status="failed", failed_at=_now(), failure=_failure(error),
                        failed_source_query_id=active_source_query_id)
                                                                              
                                                                               
        for name, read in (
                ("judgments_sha256", lambda: sha256_file(out) if out.exists() else None),
                ("spend", client.ledger.summary), ("resources", budget.check)):
            try:
                metadata[name] = read()
            except Exception:
                metadata["failure_reporting_incomplete"] = True
        try:
            _replace_manifest(manifest_path, metadata, budget)
        except Exception:
            pass
        if isinstance(error, (PaidHTTPError, RateLimitError, IncompleteResponseError)):
            raise error from None
        if isinstance(error, (JudgeError, JudgeWorkflowError, KeyboardInterrupt, SystemExit)):
            raise error from None
        raise JudgeWorkflowError("Judge run failed; partial output and failed manifest retained") from None
    return {"judgments": str(out), "manifest": str(manifest_path), "status": "complete",
            "query_count": len(prepared), "judgment_count": metadata["judgment_count"],
            "observation_namespace": metadata["observation_namespace"],
            "judgments_sha256": metadata["judgments_sha256"],
            "calibration_status": "not_human_audited", "resources": metadata["resources"],
            "runtime_sha256": metadata["runtime_sha256"],
            "spend": metadata["spend"]}
