                                                                              

                                                                               
                                                                              
                                                                            
                                                                              
   

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping

from magra_rag.resources import ResourceBudget


MIRROR_REPOSITORY = "hotpotqa/hotpot_qa"
MIRROR_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
MIRROR_SOURCE = f"https://huggingface.co/datasets/{MIRROR_REPOSITORY}"
CONVERTER_VERSION = "magra.hotpot_hf_original_schema.v1"
MIRROR_DATASETS = {
    "dev_distractor": {
        "configuration": "distractor", "mirror_split": "validation", "official_subset": "dev",
        "filename": "hotpot_dev_distractor_v1.jsonl", "rows": 7405,
        "files": [{"path": "distractor/validation-00000-of-00001.parquet", "bytes": 27452575,
                   "sha256": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"}],
    },
    "train": {
        "configuration": "distractor", "mirror_split": "train", "official_subset": "train",
        "filename": "hotpot_train_v1.1.jsonl", "rows": 90447,
        "files": [
            {"path": "distractor/train-00000-of-00002.parquet", "bytes": 165624177,
             "sha256": "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68"},
            {"path": "distractor/train-00001-of-00002.parquet", "bytes": 166162479,
             "sha256": "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f"},
        ],
    },
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pinned_url(spec: Mapping[str, Any]) -> str:
    return f"{MIRROR_SOURCE}/resolve/{MIRROR_REVISION}/{spec['path']}"


def _string(value: Any, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"Hotpot mirror {field} must be a {'nonempty ' if nonempty else ''}string")
    return value


def _pairs(value: Any, first: str, second: str, field: str) -> list[tuple[Any, Any]]:
    pass                                                                          
    if isinstance(value, Mapping):
        left, right = value.get(first), value.get(second)
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            raise ValueError(f"Hotpot mirror {field} has mismatched arrays")
        return list(zip(left, right))
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, Mapping) and set(item) == {first, second}:
                result.append((item[first], item[second]))
            else:
                raise ValueError(f"Hotpot mirror {field} has an invalid record")
        return result
    raise ValueError(f"Hotpot mirror {field} must be structured sequences")


def normalize_hotpot_row(row: Mapping[str, Any]) -> dict[str, Any]:
    pass                                                                        

                                                                                
                                                                         
       
    if not isinstance(row, Mapping):
        raise ValueError("Hotpot mirror rows must be mappings")
    identifier = _string(row.get("id", row.get("_id")), "id", nonempty=True)
    if "id" in row and "_id" in row and row["id"] != row["_id"]:
        raise ValueError("Hotpot mirror has conflicting row IDs")
    result = {"_id": identifier, "question": _string(row.get("question"), "question", nonempty=True)}
    if "answer" in row:
        result["answer"] = _string(row["answer"], "answer")
    for field in ("type", "level"):
        if field in row:
            result[field] = _string(row[field], field)
    context = []
    for title, sentences in _pairs(row.get("context"), "title", "sentences", "context"):
        _string(title, "context title", nonempty=True)
        if not isinstance(sentences, list) or not all(isinstance(sentence, str) for sentence in sentences):
            raise ValueError("Hotpot mirror context sentences must be lists of strings")
        context.append([title, list(sentences)])
    supports = []
    for title, sentence_id in _pairs(row.get("supporting_facts"), "title", "sent_id", "supporting_facts"):
        _string(title, "support title", nonempty=True)
        if type(sentence_id) is not int or sentence_id < 0:
            raise ValueError("Hotpot mirror support sentence IDs must be nonnegative integers")
        supports.append([title, sentence_id])
    result.update(context=context, supporting_facts=supports)
    return result


def iter_mirror_rows(paths: Iterable[str | Path], *, batch_size: int = 64):
    pass                                                                            
    if not 1 <= batch_size <= 256:
        raise ValueError("Parquet batch size must be 1..256")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Hotpot Parquet fallback requires optional pyarrow") from error
    for path in paths:
        with parquet.ParquetFile(path) as source:
            for batch in source.iter_batches(batch_size=batch_size, use_threads=False):
                for row in batch.to_pylist():
                    yield normalize_hotpot_row(row)


def _download(spec: Mapping[str, Any], target: Path, *, timeout: float, opener) -> None:
    request = urllib.request.Request(_pinned_url(spec), headers={"User-Agent": "MagraBenchmark/1.0", "Accept-Encoding": "identity"})
    with opener(request, timeout=timeout) as response:
                                                                               
                                                                 
        final = urllib.parse.urlsplit(response.geturl())
        hostname = (final.hostname or "").lower()
        if final.scheme != "https" or not (
            hostname == "huggingface.co" or hostname.endswith(".huggingface.co")
            or hostname.endswith(".hf.co") or hostname == "cas-bridge.xethub.hf.co"
        ):
            raise ValueError("Hotpot mirror redirected outside publisher distribution hosts")
        encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding not in {"identity", ""}:
            raise ValueError("Compressed HTTP transfer is unsupported for pinned Parquet bytes")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != spec["bytes"]:
            raise ValueError("Hotpot mirror Content-Length differs from pinned bytes")
        remaining = spec["bytes"]
        with target.open("xb") as output:
            while remaining:
                chunk = response.read(min(65_536, remaining))
                if not chunk:
                    raise ValueError("Hotpot mirror response ended before pinned file size")
                if len(chunk) > remaining:
                    raise ValueError("Hotpot mirror response exceeds pinned file size")
                output.write(chunk)
                remaining -= len(chunk)
    if _sha256(target) != spec["sha256"]:
        raise ValueError("Hotpot mirror SHA-256 check failed")


def acquire_hotpot_mirror(
    name: str, destination: str | Path, *, local_files: Iterable[str | Path] | None = None,
    max_download_bytes: int = 400_000_000, max_output_bytes: int = 800_000_000,
    timeout: float = 60.0, budget: ResourceBudget | None = None, opener=None,
) -> Path:
    pass                                                                            

                                                                                
                                                                              
                                                                                  
                                                                                  
       
    if name not in MIRROR_DATASETS:
        raise ValueError(f"Unknown pinned Hotpot mirror split: {name}")
    if max_download_bytes < 1 or max_output_bytes < 1 or not 0 < timeout <= 300:
        raise ValueError("Invalid mirror acquisition bounds")
    destination = Path(destination).resolve()
    budget = budget or ResourceBudget(destination)
    budget.contains(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spec = MIRROR_DATASETS[name]
    total_download = sum(item["bytes"] for item in spec["files"])
    if total_download > max_download_bytes:
        raise ValueError("Pinned mirror split exceeds download budget before acquisition")
    final_directory = destination / f"hotpot_hf_{name}_{MIRROR_REVISION[:12]}"
    output_name = spec["filename"]
    if final_directory.exists():
        acquisition = json.loads((final_directory / "acquisition.json").read_text(encoding="utf-8"))
        output = final_directory / output_name
        if (acquisition.get("format") != "magra-hotpot-publisher-mirror-v1"
            or acquisition.get("origin") != "publisher_huggingface_mirror"
            or acquisition.get("official_original_bytes") is not False
            or acquisition.get("full_wikipedia_corpus") is not False
            or acquisition.get("repository") != MIRROR_REPOSITORY
            or acquisition.get("revision") != MIRROR_REVISION
            or acquisition.get("source_files") != [{**item, "url": _pinned_url(item)} for item in spec["files"]]
            or acquisition.get("query_count") != spec["rows"]
            or acquisition.get("configuration") != spec["configuration"]
            or acquisition.get("mirror_split") != spec["mirror_split"]
            or acquisition.get("output_bytes") != output.stat().st_size
            or output.stat().st_size > max_output_bytes
            or acquisition.get("converter") != CONVERTER_VERSION
            or acquisition.get("output_sha256") != _sha256(output)):
            raise ValueError("Existing Hotpot mirror acquisition differs from its pinned manifest")
        budget.check()
        return output
    locals_list = None if local_files is None else [budget.contains(path) for path in local_files]
    if locals_list is not None:
        if len(locals_list) != len(spec["files"]):
            raise ValueError("Local mirror file count differs from the pinned shard count")
        for path, item in zip(locals_list, spec["files"]):
            if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
                raise ValueError("Local Hotpot mirror bytes or SHA-256 differ from pin")
    budget.check((total_download if locals_list is None else 0) + max_output_bytes + 1_000_000)
    try:
        import pyarrow
    except ImportError as error:
        raise RuntimeError("Hotpot Parquet fallback requires optional pyarrow") from error
    with tempfile.TemporaryDirectory(prefix=".hotpot-mirror-", dir=destination) as temporary:
        stage = Path(temporary)
        paths = locals_list
        if paths is None:
            paths = []
            for item in spec["files"]:
                path = stage / Path(item["path"]).name
                _download(item, path, timeout=timeout, opener=opener or urllib.request.urlopen)
                paths.append(path)
                budget.check()
        output = stage / output_name
        count, written = 0, 0
        seen: set[str] = set()
        with output.open("xb") as stream:
            for row in iter_mirror_rows(paths):
                identifier = row["_id"]
                if identifier in seen:
                    raise ValueError("Duplicate Hotpot mirror query ID across shards")
                seen.add(identifier)
                encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                if len(encoded) > 16_000_000:
                    raise ValueError("Converted Hotpot record exceeds per-record bound")
                if written + len(encoded) > max_output_bytes:
                    raise ValueError("Converted Hotpot JSONL exceeds output budget")
                stream.write(encoded)
                written += len(encoded)
                count += 1
                if count > spec["rows"]:
                    raise ValueError("Hotpot mirror row count exceeds pinned split count")
                if count % 256 == 0:
                    budget.check()
        if count != spec["rows"]:
            raise ValueError("Hotpot mirror row count differs from pinned complete split")
        manifest = {
            "format": "magra-hotpot-publisher-mirror-v1", "origin": "publisher_huggingface_mirror",
            "source": MIRROR_SOURCE, "repository": MIRROR_REPOSITORY, "revision": MIRROR_REVISION,
            "upstream_reference": "https://hotpotqa.github.io/", "official_original_bytes": False,
            "configuration": spec["configuration"], "mirror_split": spec["mirror_split"],
            "official_subset": spec["official_subset"], "query_count": count,
            "source_files": [{**item, "url": _pinned_url(item)} for item in spec["files"]],
            "output_filename": output_name, "output_bytes": written, "output_sha256": _sha256(output),
            "converter": CONVERTER_VERSION, "pyarrow_version": pyarrow.__version__,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "license": "cc-by-sa-4.0", "contains_private_qa_annotations": True,
            "full_wikipedia_corpus": False,
            "limitations": ["Publisher Parquet mirror converted to original Hotpot schema; this is not an original-file byte checksum.",
                            "Context records are per-question evidence, not the independent Wikipedia full-corpus collection.",
                            "Only document iterators may be indexed; questions, answers and annotations remain separate."],
        }
        (stage / "acquisition.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        budget.check()
        stage.rename(final_directory)
    return final_directory / output_name
