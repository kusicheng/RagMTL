                                                                               

                                                                                
                                                                               
                                                                             
                                                                            
   

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Mapping
import urllib.error
import urllib.request
import uuid

from magra_rag.resources import BudgetExceeded

from .datasets import sha256_file
from .protocol import json_hash, read_jsonl
from .splits import group_queries


REPOSITORY = "https://github.com/ParticleMedia/RAGTruth"
RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAGTRUTH_FILES = {
    "LICENSE": {"bytes": 1071,
                "sha256": "b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244",
                "git_blob_sha1": "71ba6fcf69d6c0d3c246115c237d63ab98a5baad"},
    "README.md": {"bytes": 21234,
                  "sha256": "3686178d505a38cb20a94827917369bcb9d0aa44cf5154d7ddbac740ee06628d",
                  "git_blob_sha1": "aa15888b6349f8450b26fada228b90717a9b9bd6"},
    "dataset/response.jsonl": {"bytes": 21458735,
                               "sha256": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
                               "git_blob_sha1": "f9ae5913fb9976684f3e13be3cd3465062db54cc"},
    "dataset/source_info.jsonl": {"bytes": 15117971,
                                  "sha256": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
                                  "git_blob_sha1": "119277774009f110aa58596b93f9512f62594a89"},
}
DEFAULT_SEED = "magra-ragtruth-v1"
_MAX_INPUT_BYTES = 64_000_000


def _publisher_url(relative):
    return f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/{relative}"


def _check_publisher_url(url):
                                                                               
                                                                        
    if url not in {_publisher_url(relative) for relative in RAGTRUTH_FILES}:
        raise ValueError("RAGTruth acquisition requires an exact pinned publisher HTTPS URL")


class _RAGTruthRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_publisher_url(newurl)
        if newurl != req.full_url:
            raise ValueError("RAGTruth redirect cannot change the pinned file URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _check_pinned_file(path, expected):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected["bytes"]:
        raise ValueError("RAGTruth pinned file is missing or has an unexpected size")
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {expected['bytes']}\0".encode())
    received = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            received += len(chunk)
            if received > expected["bytes"]:
                raise ValueError("RAGTruth file changed while validating its bound")
            digest.update(chunk)
            blob.update(chunk)
    if received != expected["bytes"] or digest.hexdigest() != expected["sha256"] \
            or blob.hexdigest() != expected["git_blob_sha1"]:
        raise ValueError("RAGTruth publisher SHA-256 or Git blob checksum mismatch")


def _validate_existing_acquisition(target):
    manifest_path = target / "acquisition.json"
    if not target.is_dir() or not manifest_path.is_file() or manifest_path.stat().st_size > 1_000_000:
        raise ValueError("Existing RAGTruth destination is incomplete; choose a new destination")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("format") != "magra-ragtruth-acquisition-v1"
            or manifest.get("repository") != REPOSITORY or manifest.get("revision") != RAGTRUTH_REVISION
            or (manifest.get("status") is not None and manifest.get("status") != "complete")
            or not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(RAGTRUTH_FILES)):
        raise ValueError("Existing RAGTruth acquisition does not match the pinned publisher manifest")
    for relative, expected in RAGTRUTH_FILES.items():
        record = manifest["files"][relative]
        if not isinstance(record, dict) or record.get("url") != _publisher_url(relative) \
                or any(type(record.get(key)) is not type(value) or record.get(key) != value
                       for key, value in expected.items()):
            raise ValueError("Existing RAGTruth manifest pins differ from the built-in publisher pins")
        path = target / relative
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError("Existing RAGTruth file escapes its dataset directory")
        _check_pinned_file(path, expected)
    return target


def _remaining(deadline, timeout):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("RAGTruth acquisition deadline exceeded")
    return min(timeout, remaining)


def _download_pinned_file(opener, url, target, expected, *, timeout, deadline):
    _check_publisher_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "MAGRA-benchmark/0.1", "Accept-Encoding": "identity"})
    with opener.open(request, timeout=_remaining(deadline, timeout)) as response:
        if response.geturl() != url:
            raise ValueError("RAGTruth response URL differs from the exact pinned publisher URL")
        if response.status != 200:
            raise ValueError("RAGTruth publisher response was not HTTP 200")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("RAGTruth acquisition rejects encoded response bodies")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected["bytes"]:
            raise ValueError("RAGTruth declared response length differs from pinned byte count")
        received = 0
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {expected['bytes']}\0".encode())
        with target.open("xb") as output:
            while True:
                remaining_timeout = _remaining(deadline, timeout)
                                                                               
                                                                                
                                                                                  
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if sock is not None:
                    sock.settimeout(remaining_timeout)
                read = getattr(response, "read1", response.read)
                chunk = read(min(1_048_576, expected["bytes"] - received + 1))
                _remaining(deadline, timeout)
                if not chunk:
                    break
                received += len(chunk)
                if received > expected["bytes"]:
                    raise ValueError("RAGTruth response exceeds its pinned byte count")
                digest.update(chunk)
                blob.update(chunk)
                output.write(chunk)
        if (received != expected["bytes"] or digest.hexdigest() != expected["sha256"]
                or blob.hexdigest() != expected["git_blob_sha1"]):
            raise ValueError("RAGTruth publisher byte count, SHA-256, or Git blob checksum mismatch")


def acquire_ragtruth(destination, budget, *, max_bytes=40_000_000, timeout=60, deadline_seconds=300):
    pass                                                                         

                                                                              
                                                                                
                                                                             
                                                                                  
       
    total = sum(record["bytes"] for record in RAGTRUTH_FILES.values())
    if type(max_bytes) is not int or max_bytes < total:
        raise ValueError("RAGTruth combined byte budget is smaller than the pinned publication")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in (timeout, deadline_seconds)):
        raise ValueError("RAGTruth request timeout and total deadline must be finite and positive")
    target = budget.contains(destination)
    if target.exists():
        result = _validate_existing_acquisition(target)
        budget.check()
        return result
                                                                               
                                                                               
                                                       
    budget.check(total * 2 + 131_072)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ("." + target.name + ".acquire.lock")
    stage = target.parent / (".ragtruth-acquire-" + uuid.uuid4().hex)
    lock_handle = lock.open("x", encoding="utf-8")
    try:
        lock_handle.close()
        if target.exists():
            return _validate_existing_acquisition(target)
        stage.mkdir(mode=0o755)
        (stage / "dataset").mkdir(mode=0o755)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RAGTruthRedirect())
        deadline = time.monotonic() + deadline_seconds
        files, received = {}, 0
        for relative, expected in RAGTRUTH_FILES.items():
            budget.check(total * 2 - received + 131_072)
            url = _publisher_url(relative)
            _download_pinned_file(opener, url, stage / relative, expected, timeout=timeout, deadline=deadline)
            files[relative] = {"url": url, **expected}
            received += expected["bytes"]
        manifest = {
            "format": "magra-ragtruth-acquisition-v1", "status": "complete", "repository": REPOSITORY,
            "revision": RAGTRUTH_REVISION, "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files, "max_total_bytes": max_bytes, "downloaded_bytes": total, "license_file": "LICENSE",
            "purpose": "Separate human-annotated groundedness evaluator audit; not indexed as retrieval documents.",
        }
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        budget.check(len(raw))
        with (stage / "acquisition.json").open("xb") as stream:
            stream.write(raw)
        _validate_existing_acquisition(stage)
        if target.exists():
            raise FileExistsError("RAGTruth destination appeared during acquisition; refusing overwrite")
        stage.rename(target)
        budget.check()
        return target
    except (urllib.error.URLError, OSError) as error:
        if isinstance(error, (TimeoutError, FileExistsError)):
            raise
        raise RuntimeError("RAGTruth public acquisition failed: " + type(error).__name__) from None
    finally:
        lock_handle.close()
        if stage.exists():
            resolved = stage.resolve()
            if resolved.parent != target.parent.resolve() or not stage.name.startswith(".ragtruth-acquire-"):
                raise RuntimeError("Refusing to clean an unexpected RAGTruth staging path")
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def _source_paths(source_dir):
    source_dir = Path(source_dir).resolve()
    dataset = source_dir / "dataset" if (source_dir / "dataset").is_dir() else source_dir
    acquisition = source_dir / "acquisition.json"
    if not acquisition.exists():
        acquisition = dataset.parent / "acquisition.json"
    return dataset, acquisition


def _verify_acquisition(dataset, acquisition):
    if not acquisition.is_file() or acquisition.stat().st_size > 1_000_000:
        raise ValueError("RAGTruth requires a bounded publisher-pinned acquisition.json")
    manifest = json.loads(acquisition.read_text(encoding="utf-8"))
    revision = manifest.get("revision")
    if manifest.get("repository", "").rstrip("/") != REPOSITORY or not isinstance(revision, str) \
            or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("RAGTruth acquisition must pin the publisher repository and a full commit")
    files = {}
    for name in ("response.jsonl", "source_info.jsonl"):
        path = dataset / name
        metadata = manifest.get("files", {}).get("dataset/" + name, {})
        if not path.is_file() or not 0 < path.stat().st_size <= _MAX_INPUT_BYTES:
            raise ValueError("RAGTruth source files must be nonempty and bounded")
        digest = sha256_file(path)
        expected_url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{revision}/dataset/{name}"
        if (metadata.get("sha256") != digest or metadata.get("bytes") != path.stat().st_size
                or metadata.get("url") != expected_url):
            raise ValueError("RAGTruth source bytes or publisher pin differ from acquisition manifest")
        if metadata.get("git_blob_sha1") is not None:
            blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1_048_576):
                    blob.update(chunk)
            if blob.hexdigest() != metadata["git_blob_sha1"]:
                raise ValueError("RAGTruth Git blob hash does not match publisher acquisition")
        files[name] = {"bytes": path.stat().st_size, "sha256": digest}
    if sum(record["bytes"] for record in files.values()) > _MAX_INPUT_BYTES:
        raise ValueError("RAGTruth raw input exceeds the 64 MB adapter bound")
    return manifest, files


def _valid_spans(response, spans):
    if not isinstance(spans, list):
        return False
    for span in spans:
        if not isinstance(span, Mapping):
            return False
        start, end, text = span.get("start"), span.get("end"), span.get("text")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(response)
                or not isinstance(text, str) or response[start:end] != text
                or not isinstance(span.get("label_type"), str) or not span["label_type"].strip()):
            return False
        if "implicit_true" in span and type(span["implicit_true"]) is not bool:
            return False
        if "due_to_null" in span and type(span["due_to_null"]) is not bool:
            return False
        if "meta" in span and span["meta"] is not None and not isinstance(span["meta"], str):
            return False
    return True


def _duplicate_groups(sources, threshold):
    ids = sorted(sources)
    parents = {source_id: source_id for source_id in ids}

    def find(source_id):
        while parents[source_id] != source_id:
            parents[source_id] = parents[parents[source_id]]
            source_id = parents[source_id]
        return source_id

    for field in ("question", "passages"):
        grouped = group_queries(({"id": source_id, "question": sources[source_id][field]}
                                 for source_id in ids), similarity_threshold=threshold)
        members = {}
        for source_id in ids:
            previous = members.setdefault(grouped[source_id], source_id)
            left, right = find(previous), find(source_id)
            if left != right:
                parents[max(left, right)] = min(left, right)
    groups = defaultdict(list)
    for source_id in ids:
        groups[find(source_id)].append(source_id)
    result = {}
    for members in groups.values():
        group_id = "ragtruth-group:" + json_hash(sorted(members))[:24]
        for source_id in members:
            result[source_id] = group_id
    return result


def _select_balanced(candidates, quotas, seed):
    pass                                                                      
    selected, available = [], {}
    for split in ("dev", "eval"):
        groups = defaultdict(lambda: defaultdict(list))
        for row in candidates:
            if row["audit_split"] == split:
                groups[row["group_id"]][row["human_label"]].append(row)
        available[split] = {str(label): sum(label in options for options in groups.values()) for label in (0, 1)}
        remaining = {0: quotas[split], 1: quotas[split]}
        ordered = sorted(groups, key=lambda group: (json_hash([seed, split, group]), group))
        chosen = []
        for label in (0, 1):
            for group in ordered:
                if set(groups[group]) == {label} and remaining[label]:
                    chosen.append((group, label))
                    remaining[label] -= 1
        for group in ordered:
            if set(groups[group]) != {0, 1}:
                continue
            needed = [label for label in (0, 1) if remaining[label]]
            if not needed:
                break
            label = min(needed, key=lambda label: (-remaining[label], json_hash([seed, group, label])))
            chosen.append((group, label))
            remaining[label] -= 1
        if any(remaining.values()):
            raise ValueError(f"Insufficient independent RAGTruth {split} groups for requested balanced class counts; missing {remaining}")
        for group, label in chosen:
            row = min(groups[group][label], key=lambda row: (json_hash([seed, "response", row["id"]]), row["id"]))
            selected.append(row)
    return sorted(selected, key=lambda row: (row["audit_split"], row["group_id"], row["id"])), available


def _public_and_private(row, sources, acquisition, source_hashes):
    source = sources[row["source_id"]]
    query_id = "ragtruth:qa:" + json_hash([row["source_id"], row["id"]])[:24]
    context_id = "ragtruth-context:" + hashlib.sha256(source["passages"].encode()).hexdigest()
    public = {
        "query": {"id": query_id, "question": source["question"], "source": "ragtruth_qa",
                  "domain": "web_qa", "setting": "provided_context", "group_id": row["group_id"],
                  "split": row["audit_split"]},
        "prediction": {"query_id": query_id, "answer": row["response"], "citations": [], "abstained": False},
        "evidence": [{"doc_id": context_id, "sentence_id": 0, "title": "Provided passages",
                      "text": source["passages"], "source": "ragtruth_qa"}],
    }
    private = {
        "query_id": query_id, "split": row["audit_split"], "label": row["human_label"],
        "group_id": row["group_id"], "annotation_source": "RAGTruth published human annotations",
        "source": source["source"],
        "annotation_provenance": {
            "repository": REPOSITORY, "revision": acquisition["revision"], "task_type": "QA",
            "response_id": row["id"], "source_id": row["source_id"], "original_model": row.get("model"),
            "official_split": row["split"], "quality": row["quality"],
            "raw_file_hashes": {name: record["sha256"] for name, record in source_hashes.items()},
            "span_count": len(row["labels"]), "annotation_spans_hash": json_hash(row["labels"]),
            "contains_implicit_true": any(span.get("implicit_true", False) for span in row["labels"]),
            "label_rule": "1 iff the published hallucination-span list is empty; implicit_true spans remain unsupported",
        },
        "annotation_spans": row["labels"],
    }
    return public, private


def prepare_ragtruth_audit(
    source_dir, out, budget, *, dev_per_class=40, eval_per_class=40,
    seed=DEFAULT_SEED, similarity_threshold=0.85,
):
    pass                                                                      

                                                                             
                                                                              
                                                                            
                                                                               
                                                                   
       
    for name, count in (("dev_per_class", dev_per_class), ("eval_per_class", eval_per_class)):
        if type(count) is not int or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, str) or not seed or not 0 < similarity_threshold <= 1:
        raise ValueError("A nonempty seed and a similarity threshold in (0,1] are required")
    destination = budget.contains(out)
    if destination.exists():
        raise FileExistsError("RAGTruth audit output already exists; use a new directory")
    dataset, acquisition_path = _source_paths(source_dir)
    acquisition, source_hashes = _verify_acquisition(dataset, acquisition_path)
    telemetry = budget.check()
    if telemetry["process_peak_rss_bytes"] + sum(x["bytes"] for x in source_hashes.values()) * 12 >= budget.ram_limit:
        raise BudgetExceeded("RAGTruth adapter input would exceed the process RAM budget")
    exclusions, sources, source_ids = Counter(), {}, set()
    for source in read_jsonl(dataset / "source_info.jsonl"):
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise ValueError("RAGTruth source IDs must be unique nonempty strings")
        source_ids.add(source_id)
        if source.get("task_type") != "QA":
            exclusions["non_qa_source"] += 1
            continue
        info = source.get("source_info")
        if (not isinstance(info, dict) or set(info) != {"question", "passages"}
                or any(not isinstance(info[field], str) or not info[field].strip() for field in ("question", "passages"))):
            exclusions["unsupported_qa_source_schema"] += 1
            continue
        sources[source_id] = {**info, "source": str(source.get("source", "unknown"))}
    if not sources:
        raise ValueError("No unambiguous QA sources available for RAGTruth audit")
    groups = _duplicate_groups(sources, similarity_threshold)
    responses, response_ids, test_groups = [], set(), set()
    for row in read_jsonl(dataset / "response.jsonl"):
        response_id, source_id = row.get("id"), row.get("source_id")
        if not isinstance(response_id, str) or not response_id or response_id in response_ids:
            raise ValueError("RAGTruth response IDs must be unique nonempty strings")
        response_ids.add(response_id)
        if source_id not in source_ids:
            raise ValueError("RAGTruth response references an unknown source")
        if source_id not in sources:
            exclusions["non_qa_or_unsupported_source_response"] += 1
            continue
        if row.get("split") not in {"train", "test"}:
            raise ValueError("RAGTruth official response split must be train or test")
        if row["split"] == "test":
            test_groups.add(groups[source_id])
        if row.get("quality") != "good":
            exclusions["response_quality_not_good"] += 1
            continue
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            exclusions["empty_response"] += 1
            continue
        if not _valid_spans(row["response"], row.get("labels")):
            exclusions["ambiguous_or_invalid_annotations"] += 1
            continue
        responses.append({**row, "group_id": groups[source_id], "human_label": int(not row["labels"]),
                          "audit_split": "dev" if row["split"] == "train" else "eval"})
    candidates = []
    for row in responses:
        if row["split"] == "train" and row["group_id"] in test_groups:
            exclusions["train_group_touches_official_test"] += 1
        else:
            candidates.append(row)
    selected, available = _select_balanced(candidates, {"dev": dev_per_class, "eval": eval_per_class}, seed)
    public_rows, human_rows = {"dev": [], "eval": []}, {"dev": [], "eval": []}
    for row in selected:
        public, private = _public_and_private(row, sources, acquisition, source_hashes)
        public_rows[row["audit_split"]].append(public)
        human_rows[row["audit_split"]].append(private)
    for name, record in source_hashes.items():
        if sha256_file(dataset / name) != record["sha256"]:
            raise ValueError("RAGTruth raw source changed during audit preparation")
    payloads = {}
    for split in ("dev", "eval"):
        payloads[f"public/{split}.inputs.jsonl"] = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in public_rows[split]).encode()
        payloads[f"private/{split}.labels.jsonl"] = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in human_rows[split]).encode()
    payloads["private/human-labels.jsonl"] = payloads["private/dev.labels.jsonl"] + payloads["private/eval.labels.jsonl"]
    payloads["acquisition.json"] = acquisition_path.read_bytes()
    file_records = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in payloads.items()}
    public_manifest = {
        "format": "magra-ragtruth-public-inputs-v1",
        "files": {f"{split}.inputs.jsonl": {**file_records[f"public/{split}.inputs.jsonl"],
                                           "records": len(public_rows[split])} for split in ("dev", "eval")},
        "source_dataset": {"repository": REPOSITORY, "revision": acquisition["revision"],
                           "source_file_hashes": {name: record["sha256"] for name, record in source_hashes.items()}},
    }
    public_manifest["manifest_hash"] = json_hash(public_manifest)
    payloads["public/manifest.json"] = (json.dumps(public_manifest, sort_keys=True, indent=2) + "\n").encode()
    file_records["public/manifest.json"] = {"sha256": hashlib.sha256(payloads["public/manifest.json"]).hexdigest(),
                                            "bytes": len(payloads["public/manifest.json"])}
    manifest = {
        "format": "magra-ragtruth-human-audit-v1", "status": "prepared_not_judged",
        "repository": REPOSITORY, "revision": acquisition["revision"],
        "source_files": source_hashes, "acquisition_sha256": sha256_file(acquisition_path),
        "adapter_sha256": sha256_file(Path(__file__)),
        "grouping_sha256": sha256_file(Path(__file__).with_name("splits.py")),
        "seed": seed, "requested_per_class": {"dev": dev_per_class, "eval": eval_per_class},
        "selected_counts": {split: {str(label): sum(row["label"] == label for row in human_rows[split])
                                    for label in (0, 1)} for split in ("dev", "eval")},
        "available_independent_groups_per_class": available, "exclusions": dict(sorted(exclusions.items())),
        "files": file_records,
        "selection_policy": {
            "task": "QA only", "quality": "good only; nonempty response; exact valid annotation spans",
            "label": "1 iff no annotated hallucination spans; implicit_true remains unsupported",
            "development": "official train only", "evaluation": "official test only",
            "grouping": "source_id plus normalized exact/explicit token-Jaccard question OR context groups, transitive closure",
            "similarity_threshold": similarity_threshold,
            "cross_split_collision": "test wins; touching train responses excluded, never promoted into evaluation",
            "independence": "one response per duplicate group, at most one per source",
            "sampling": "balanced classes; exclusive-label groups first, then mixed-label groups, stable seeded SHA-256 order",
            "evidence": "full provided passages; no truncation or annotation-dependent evidence selection",
        },
        "limitations": [
            "Published humans annotated earlier model responses; this does not rate current MAGRA or MiniMax M3 answers.",
            "QA is from MARCO only; this external evaluator audit does not establish generality across domains.",
            "Balanced response selection changes class prevalence and response/model distribution; calibrated probabilities cannot transfer silently to production.",
            "Lexical grouping cannot detect every semantic paraphrase or context overlap.",
            "Annotation absence is the publisher-derived operational groundedness label, not proof of truth or completeness.",
            "No judge predictions, audit scores, calibration fit, or deployment acceptance is claimed by this preparation step.",
            "Private files are separated and owner-permissioned where supported; filesystem separation is not an ACL against the process owner.",
        ],
    }
    manifest["manifest_hash"] = json_hash(manifest)
    payloads["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    budget.check(sum(map(len, payloads.values())) + 1_048_576)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ragtruth-audit-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "audit"
        (stage / "public").mkdir(parents=True)
        (stage / "private").mkdir(mode=0o700)
        for relative, raw in payloads.items():
            path = stage / relative
            with path.open("xb") as stream:
                stream.write(raw)
            if relative.startswith("private/"):
                path.chmod(0o600)
        budget.check()
        stage.rename(destination)
    return {
        "directory": str(destination), "manifest": str(destination / "manifest.json"),
        "manifest_hash": manifest["manifest_hash"], "public_manifest": str(destination / "public/manifest.json"),
        "public_inputs": {split: str(destination / f"public/{split}.inputs.jsonl") for split in ("dev", "eval")},
        "human_labels": str(destination / "private/human-labels.jsonl"),
        "private_labels": {split: str(destination / f"private/{split}.labels.jsonl") for split in ("dev", "eval")},
        "selected_counts": manifest["selected_counts"], "resources": budget.check(),
        "status": "prepared_not_judged",
    }


                                                                                 
BROAD_STRATA = {
    ("QA", "MARCO"): ("marco_qa", "ragtruth_qa", "web_qa"),
    ("Summary", "CNN/DM"): ("cnn_dm_summary", "ragtruth_cnn_dm_summary", "news_summary"),
    ("Summary", "Recent News"): ("recent_news_summary", "ragtruth_recent_news_summary", "news_summary"),
    ("Data2txt", "Yelp"): ("yelp_data2txt", "ragtruth_yelp_data2txt", "local_business_data_to_text"),
}
_BROAD_PUBLIC_ROW_BYTES = 1_000_000                                              


def _broad_source(row):
    pass                                                                            
    key = (row.get("task_type"), row.get("source"))
    if key not in BROAD_STRATA:
        raise ValueError("Unsupported broad-audit task/source pair")
    info, prompt = row.get("source_info"), row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Original publisher instruction is required")
    result = {"task_type": key[0], "source": key[1],
              "stratum": BROAD_STRATA[key][0], "public_source": BROAD_STRATA[key][1],
              "domain": BROAD_STRATA[key][2], "business_identity": None}
    if key[0] == "QA":
        if (not isinstance(info, dict) or set(info) != {"question", "passages"}
                or any(not isinstance(info[k], str) or not info[k].strip() for k in info)
                or prompt.count(info["passages"]) != 1):
            raise ValueError("Ambiguous original QA instruction/context")
        before, after = prompt.split(info["passages"])
        if not before.startswith("Briefly answer the following question:\n" + info["question"]):
            raise ValueError("Original QA question does not match source_info")
        if not after.endswith("\noutput:"):
            raise ValueError("Original QA output boundary is missing")
        result.update(question=info["question"], passages=info["passages"],
                      instruction=(before + after.removesuffix("\noutput:")).strip(),
                      evidence=info["passages"], title="Provided passages")
    elif key[0] == "Summary":
        if not isinstance(info, str) or not info.strip() or prompt.count(info) != 1:
            raise ValueError("Ambiguous original summary article")
        before, after = prompt.split(info)
        if not re.fullmatch(r"Summarize the following news within \d+ words:\n", before) or after != "\noutput:":
            raise ValueError("Unrecognized original summary instruction/boundary")
        result.update(instruction=before.strip(), evidence=info, title="Provided news article")
    else:
        fields = {"name", "address", "city", "state", "categories", "hours", "attributes", "business_stars", "review_info"}
        if (not isinstance(info, dict) or set(info) != fields or not isinstance(info["review_info"], list)
                or any(not isinstance(review, dict) or set(review) != {"review_stars", "review_date", "review_text"}
                       for review in info["review_info"])):
            raise ValueError("Unsupported publisher business evidence schema")
        before, separator, after = prompt.partition("Structured data:\n")
        if not separator or after != str(info) + "\nOverview:" or not before.startswith("Instruction:\n"):
            raise ValueError("Original business instruction/context mismatch")
        from .splits import normalize_question
        identity = [normalize_question(info[k]) if isinstance(info[k], str) and info[k].strip() else ""
                    for k in ("name", "address", "city", "state")]
        result.update(instruction=before.strip(), title="Provided business and customer reviews",
                      evidence=json.dumps(info, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                      business_identity=tuple(identity) if all(identity) else None)
    return result


def _broad_groups(sources, threshold):
    parents = {identifier: identifier for identifier in sources}

    def find(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    def merge(groups):
        representatives = {}
        for identifier, group in groups.items():
            union(identifier, representatives.setdefault(group, identifier))

    qa = {identifier: source for identifier, source in sources.items() if source["task_type"] == "QA"}
    merge(_duplicate_groups(qa, threshold))
    for task in ("Summary", "Data2txt"):
        merge(group_queries(({"id": identifier, "question": source["evidence"]}
                             for identifier, source in sources.items() if source["task_type"] == task),
                            similarity_threshold=threshold))
    identities = {}
    for identifier, source in sources.items():
        if source["business_identity"] is not None:
            union(identifier, identities.setdefault(source["business_identity"], identifier))
    members = defaultdict(list)
    for identifier in sorted(sources):
        members[find(identifier)].append(identifier)
    groups = {"ragtruth-broad-group:" + json_hash(ids)[:24]: ids for ids in members.values()}
    return {identifier: group for group, ids in groups.items() for identifier in ids}, groups


def _broad_prior_exclusions(directory, sources, acquisition, source_hashes):
    pass                                                                              
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1_000_000:
        raise ValueError("Prior QA audit manifest is required and bounded")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "magra-ragtruth-human-audit-v1"
            or manifest.get("repository") != REPOSITORY or manifest.get("revision") != acquisition["revision"]
            or manifest.get("source_files") != source_hashes):
        raise ValueError("Prior QA audit refers to different raw source bytes")
    qa = {identifier: source for identifier, source in sources.items() if source["task_type"] == "QA"}
    groups = _duplicate_groups(qa, manifest["selection_policy"]["similarity_threshold"])
    known = set(groups.values())
    old_groups, file_hashes = set(), {}
    for name in ("public/manifest.json", "public/dev.inputs.jsonl", "public/eval.inputs.jsonl"):
        path = directory / name
        expected = manifest.get("files", {}).get(name, {})
        if (not path.is_file() or path.stat().st_size > _MAX_INPUT_BYTES
                or path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get("sha256")):
            raise ValueError("Prior QA public input hash mismatch")
        file_hashes[name] = sha256_file(path)
        if name.endswith(".jsonl"):
            for row in read_jsonl(path):
                query = row.get("query", {})
                group = query.get("group_id")
                if query.get("source") != "ragtruth_qa" or group not in known or group in old_groups:
                    raise ValueError("Prior QA public group is unknown or repeated")
                old_groups.add(group)
    if not old_groups:
        raise ValueError("Prior QA audit must exclude at least one verified group")
    excluded = sorted(identifier for identifier, group in groups.items() if group in old_groups)
    return set(excluded), {"manifest_sha256": sha256_file(manifest_path), "public_file_sha256": file_hashes,
                           "prior_group_count": len(old_groups), "excluded_source_ids": excluded,
                           "prior_group_ids": sorted(old_groups), "private_labels_opened": False}


def prepare_ragtruth_broad_audit(
    source_dir, out, budget, *, exclude_audit, dev_per_class=20, eval_per_class=20,
    seed="magra-ragtruth-broad-v1", similarity_threshold=0.85,
):
    pass                                                                        

                                                                                
                                                                            
                                                               
       
    for name, value in (("dev_per_class", dev_per_class), ("eval_per_class", eval_per_class)):
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError(f"{name} must be an integer in 1..100")
    if (not isinstance(seed, str) or not seed or isinstance(similarity_threshold, bool)
            or not isinstance(similarity_threshold, (int, float)) or not 0 < similarity_threshold <= 1):
        raise ValueError("A seed and finite similarity threshold in (0,1] are required")
    destination = budget.contains(out)
    if destination.exists():
        raise FileExistsError("Broad audit output already exists; choose a new immutable directory")
    dataset, acquisition_path = _source_paths(source_dir)
    acquisition, source_hashes = _verify_acquisition(dataset, acquisition_path)
    telemetry = budget.check()
    if telemetry["process_peak_rss_bytes"] + sum(record["bytes"] for record in source_hashes.values()) * 12 >= budget.ram_limit:
        raise BudgetExceeded("Broad RAGTruth input would exceed process RAM budget")
    sources, seen, exclusions = {}, set(), Counter()
    for row in read_jsonl(dataset / "source_info.jsonl"):
        identifier = row.get("source_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("RAGTruth source IDs must be unique nonempty strings")
        seen.add(identifier)
        sources[identifier] = _broad_source(row)                                                         
    strata = sorted(value[0] for value in BROAD_STRATA.values())
    if {source["stratum"] for source in sources.values()} != set(strata):
        raise ValueError("All four explicit broad-audit strata are required")
    excluded_sources, prior = _broad_prior_exclusions(exclude_audit, sources, acquisition, source_hashes)
    group_of, members = _broad_groups(sources, similarity_threshold)
    excluded_groups = {group_of[identifier] for identifier in excluded_sources}
    owners = {group: min({sources[i]["stratum"] for i in identifiers}, key=lambda s: (json_hash([seed, "stratum", group, s]), s))
              for group, identifiers in members.items()}
    source_splits, response_ids, rows = defaultdict(set), set(), []
    for row in read_jsonl(dataset / "response.jsonl"):
        identifier, source_id = row.get("id"), row.get("source_id")
        if not isinstance(identifier, str) or not identifier or identifier in response_ids:
            raise ValueError("RAGTruth response IDs must be unique nonempty strings")
        response_ids.add(identifier)
        if source_id not in sources or row.get("split") not in {"train", "test"}:
            raise ValueError("Response source and official train/test split are required")
        source_splits[source_id].add(row["split"])
        if row.get("quality") != "good":
            exclusions["response_quality_not_good"] += 1
            continue
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            exclusions["empty_response"] += 1
            continue
        if not _valid_spans(row["response"], row.get("labels")):
            exclusions["ambiguous_or_invalid_annotations"] += 1
            continue
        rows.append({**row, "group_id": group_of[source_id], "human_label": int(not row["labels"]),
                     "stratum": sources[source_id]["stratum"], "audit_split": "dev" if row["split"] == "train" else "eval"})
    if set(source_splits) != set(sources):
        raise ValueError("Every broad-audit source requires a publisher response split")
    test_groups = {group_of[i] for i, splits in source_splits.items() if "test" in splits}
    candidates = []
    for row in rows:
        group = row["group_id"]
        if group in excluded_groups:
            exclusions["prior_audit_group_response"] += 1
        elif row["split"] == "train" and group in test_groups:
            exclusions["train_group_touches_official_test"] += 1
        elif row["stratum"] != owners[group]:
            exclusions["cross_stratum_group_owned_elsewhere"] += 1
        else:
            candidates.append(row)
    selected, available = [], {}
    for stratum in strata:
        chosen, available[stratum] = _select_balanced(
            [row for row in candidates if row["stratum"] == stratum],
            {"dev": dev_per_class, "eval": eval_per_class}, seed + ":" + stratum)
        selected.extend(chosen)
    if len({row["group_id"] for row in selected}) != len(selected):
        raise ValueError("Broad selection repeated a context group")
    public_rows, human_rows = {"dev": [], "eval": []}, {"dev": [], "eval": []}
    for row in sorted(selected, key=lambda r: (r["audit_split"], r["stratum"], r["group_id"], r["id"])):
        source = sources[row["source_id"]]
        qid = "ragtruth:broad:" + json_hash([row["source_id"], row["id"]])[:24]
        public_rows[row["audit_split"]].append({
            "query": {"id": qid, "question": source["instruction"], "source": source["public_source"],
                      "domain": source["domain"], "setting": "provided_context", "group_id": row["group_id"], "split": row["audit_split"]},
            "prediction": {"query_id": qid, "answer": row["response"], "citations": [], "abstained": False},
            "evidence": [{"doc_id": "ragtruth-context:" + hashlib.sha256(source["evidence"].encode()).hexdigest(),
                          "sentence_id": 0, "title": source["title"], "text": source["evidence"], "source": source["public_source"]}]})
        _, private = _public_and_private(row, {row["source_id"]: {**source, "question": "", "passages": source["evidence"]}}, acquisition, source_hashes)
        private.update(query_id=qid, stratum=row["stratum"])
        private["annotation_provenance"]["task_type"] = source["task_type"]
        human_rows[row["audit_split"]].append(private)
    counts = {split: {stratum: {str(label): sum(row["stratum"] == stratum and row["label"] == label for row in human_rows[split])
                               for label in (0, 1)} for stratum in strata} for split in ("dev", "eval")}
    grouping = {"format": "magra-ragtruth-broad-grouping-v1", "threshold": similarity_threshold,
                "policy": {"QA": "question OR full passages", "Summary": "full article only, across both news sources; never instruction",
                           "Data2txt": "canonical full JSON context OR complete normalized name/address/city/state identity",
                           "closure": "transitive", "cross_split": "test wins; exclude touching train responses",
                           "cross_stratum": "one stable seeded stratum owner per content group, independent of labels"},
                "sources": len(sources), "groups": len(members), "prior_audit": prior,
                "excluded_broad_groups": sorted(excluded_groups),
                "cross_official_split_groups": sum(any("train" in source_splits[i] for i in ids) and any("test" in source_splits[i] for i in ids) for ids in members.values()),
                "groups_audit": [{"group_id": group, "source_ids": ids, "owner_stratum": owners[group],
                                  "official_splits": sorted(set().union(*(source_splits[i] for i in ids))),
                                  "excluded_by_prior_audit": group in excluded_groups}
                                 for group, ids in sorted(members.items())],
                "selected_unique_groups": len(selected), "selected_counts": counts,
                "available_independent_groups_per_class": available, "exclusions": dict(sorted(exclusions.items()))}
    limitations = [
        "Published humans annotated older GPT-3.5/GPT-4, Llama2 and Mistral responses, not current MAGRA/M3 answers.",
        "Four source strata span web QA, news summaries and Yelp business descriptions; this does not establish five-domain, scientific or medical coverage.",
        "Yelp source coverage is California and predominantly restaurants; news has no publisher subject-domain taxonomy.",
        "Balanced selection changes prevalence and model/response mix; calibrated probabilities cannot transfer silently to deployment.",
        "No annotated spans is the publisher-derived operational groundedness label, not proof of truth, completeness or instruction compliance; implicit_true stays unsupported.",
        "Lexical/context grouping cannot detect all semantic paraphrases; one response per group does not prove population independence.",
        "Forty evaluation items per stratum at defaults give limited precision; report 95% intervals and every stratum even if an aggregate passes.",
        "Private filesystem separation and owner permissions where supported are not an ACL against the process owner.",
        "Preparation performs no judge calls, predictions, calibration fit, acceptance decision or changes to the earlier final QA report.",
    ]
    payloads = {}
    for split in ("dev", "eval"):
        for kind, records in ((f"public/{split}.inputs.jsonl", public_rows[split]), (f"private/{split}.labels.jsonl", human_rows[split])):
            lines = [json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in records]
            if kind.startswith("public/") and any(len(line.encode()) > _BROAD_PUBLIC_ROW_BYTES for line in lines):
                raise ValueError("Complete broad public input exceeds the 1 MB judge bound; no truncation is allowed")
            payloads[kind] = "".join(lines).encode()
    payloads["private/human-labels.jsonl"] = payloads["private/dev.labels.jsonl"] + payloads["private/eval.labels.jsonl"]
    payloads["grouping-audit.json"] = (json.dumps(grouping, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    payloads["acquisition.json"] = acquisition_path.read_bytes()

    def records():
        return {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in payloads.items()}

    public_manifest = {"format": "magra-ragtruth-public-inputs-v1", "files": {
        f"{split}.inputs.jsonl": {**records()[f"public/{split}.inputs.jsonl"], "records": len(public_rows[split])}
        for split in ("dev", "eval")}, "source_dataset": {"repository": REPOSITORY, "revision": acquisition["revision"],
        "source_file_hashes": {name: record["sha256"] for name, record in source_hashes.items()}}}
    public_manifest["manifest_hash"] = json_hash(public_manifest)
    payloads["public/manifest.json"] = (json.dumps(public_manifest, sort_keys=True, indent=2) + "\n").encode()
    plan = {"format": "magra-ragtruth-broad-proposed-audit-plan-v1", "status": "prepared_not_judged",
            "public_inputs": public_manifest["files"], "public_manifest_sha256": records()["public/manifest.json"]["sha256"],
            "private_labels_sha256": {split: records()[f"private/{split}.labels.jsonl"]["sha256"] for split in ("dev", "eval")},
            "strata": {value[0]: {"task_type": key[0], "publisher_source": key[1], "query_source": value[1], "domain": value[2]}
                       for key, value in BROAD_STRATA.items()}, "selected_counts": counts,
            "proposed_gates": {"aggregate": {"min_balanced_accuracy": 0.8, "min_coverage": 0.8},
                               "per_stratum": {s: {"min_balanced_accuracy": 0.8, "min_coverage": 0.8} for s in strata},
                               "rule": "All aggregate AND per-stratum point-estimate gates required; missing/abstained/unavailable judgments reduce coverage; both judged classes required.",
                               "coverage_denominator": "All planned inputs in the scored split/stratum; BA is among covered judgments, reported alongside coverage."},
            "confidence_intervals": {"level": 0.95, "coverage": "Wilson binomial interval",
                                     "balanced_accuracy": "Percentile bootstrap over independent selected context groups, stratified by human class and source stratum",
                                     "bootstrap_samples": 2000, "seed": seed + ":intervals",
                                     "decision_use": "Report intervals; proposed .8 gates apply to point estimates, not interval lower bounds; no multiplicity correction claimed."},
            "sequence": ["Review inputs and proposed plan; freeze package, judge identity/rubric/runtime and plan before calls.",
                         "Run DEV only, fit one pooled DEV isotonic calibrator/threshold with existing fit_judge_dev; persist exact fit and hashes.",
                         "Freeze fit before EVAL calls; score EVAL once using unchanged pooled fit for aggregate and every stratum, without per-stratum refitting.",
                         "Report BA, coverage, 95% intervals, confusion, AUROC, Brier, ECE and kappa with all failures; preserve earlier QA v3 report separately."],
            "limitations": limitations}
    plan["plan_hash"] = json_hash(plan)
    payloads["proposed-plan.json"] = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    manifest = {"format": "magra-ragtruth-broad-human-audit-v1", "status": "prepared_not_judged",
                "repository": REPOSITORY, "revision": acquisition["revision"], "source_files": source_hashes,
                "acquisition_sha256": sha256_file(acquisition_path), "adapter_sha256": sha256_file(Path(__file__)),
                "grouping_sha256": sha256_file(Path(__file__).with_name("splits.py")), "seed": seed,
                "requested_per_class_per_stratum": {"dev": dev_per_class, "eval": eval_per_class},
                "selected_counts": counts, "selected_total": len(selected), "exclusions": dict(sorted(exclusions.items())),
                "prior_audit": prior, "selection_policy": grouping["policy"], "files": records(), "limitations": limitations}
    manifest["manifest_hash"] = json_hash(manifest)
    payloads["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    for name, record in source_hashes.items():
        if sha256_file(dataset / name) != record["sha256"]:
            raise ValueError("Raw publisher input changed during broad audit preparation")
    budget.check(sum(map(len, payloads.values())) + 1_048_576)
    destination.parent.mkdir(parents=True, exist_ok=True)
                                                                                
                                                                            
    import os
    stage = destination.parent / (".ragtruth-broad-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o755)
    try:
        (stage / "public").mkdir()
        (stage / "private").mkdir(mode=0o755 if os.name == "nt" else 0o700)
        for relative, raw in payloads.items():
            path = stage / relative
            with path.open("xb") as stream:
                stream.write(raw)
            if relative.startswith("private/"):
                path.chmod(0o600)
        budget.check()
        if destination.exists():
            raise FileExistsError("Broad audit destination appeared during preparation")
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"directory": str(destination), "manifest": str(destination / "manifest.json"),
            "manifest_hash": manifest["manifest_hash"], "public_manifest": str(destination / "public/manifest.json"),
            "public_inputs": {s: str(destination / f"public/{s}.inputs.jsonl") for s in ("dev", "eval")},
            "private_labels": {s: str(destination / f"private/{s}.labels.jsonl") for s in ("dev", "eval")},
            "human_labels": str(destination / "private/human-labels.jsonl"),
            "grouping_audit": str(destination / "grouping-audit.json"), "proposed_plan": str(destination / "proposed-plan.json"),
            "selected_counts": counts, "selected_total": len(selected), "resources": budget.check(), "status": "prepared_not_judged"}
