                                                                               

                                                                                    
                                                                      
   

from __future__ import annotations

import bz2
import csv
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import nullcontext
from html.parser import HTMLParser


BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
BEIR_SOURCE = "https://github.com/beir-cellar/beir/wiki/Datasets-available"
HOTPOT_SOURCE = "https://hotpotqa.github.io/"
BEIR_DATASETS = {
    "scifact": {"domain": "scientific_claims", "md5": "5f7d1de60b170fc8027bb7898e2efca1", "documents": 5183},
    "nfcorpus": {"domain": "nutrition_medicine", "md5": "a89dba18a62ef92f7d323ec890a0d38d", "documents": 3633},
    "fiqa": {"domain": "finance", "md5": "17918ed23cd04fb15047f73e6c3bd9d9", "documents": 57638},
    "arguana": {"domain": "argumentation", "md5": "8ad3e3c2a5867cdced806d6503f29b99", "documents": 8674},
}
HOTPOT_DATASETS = {
    "train": {"filename": "hotpot_train_v1.1.json", "queries": 90447},
    "dev_distractor": {"filename": "hotpot_dev_distractor_v1.json", "queries": 7405},
    "dev_fullwiki": {"filename": "hotpot_dev_fullwiki_v1.json", "queries": 7405},
    "test_fullwiki": {"filename": "hotpot_test_fullwiki_v1.json", "queries": 7405},
    "wiki": {"filename": "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
             "bytes": 1_553_565_403, "md5": "01edf64cd120ecc03a2745352779514c"},
}
HOTPOT_QUESTIONS_BASE_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/"
HOTPOT_WIKI_URL = "https://nlp.stanford.edu/projects/hotpotqa/" + HOTPOT_DATASETS["wiki"]["filename"]


def _archive_path(name: str, destination: Path) -> Path:
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    relative = PurePosixPath(name)
    if (not name or any(ord(character) < 32 for character in name) or "\\" in name or ":" in name
            or relative.is_absolute()
            or any(part in {"..", "."} or part.rstrip(" .") != part for part in name.split("/") if part)
            or any(part.split(".", 1)[0].casefold() in reserved for part in relative.parts)):
        raise ValueError("Unsafe archive member")
    target = destination.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(destination):
        raise ValueError("Archive member escapes destination")
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_lines(path: Path, max_record_bytes: int = 16_000_000) -> Iterator[dict]:
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        while line := stream.readline(max_record_bytes + 1):
            if len(line.encode("utf-8")) > max_record_bytes:
                raise ValueError(f"Oversized JSON record in {path.name}")
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL records must be objects")
            yield row


def _json_array(path: Path, max_record_chars: int = 16_000_000) -> Iterator[dict]:
    pass                                                                         
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        eof = False

        def more() -> None:
            nonlocal buffer, eof
            block = stream.read(65536)
            eof = not block
            buffer += block

        more()
        buffer = buffer.lstrip()
        if not buffer.startswith("["):
            raise ValueError("HotpotQA input must be a JSON array")
        buffer = buffer[1:]
        first = True
        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                more()
                buffer = buffer.lstrip()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                if buffer.strip():
                    raise ValueError("Trailing content after JSON array")
                while block := stream.read(65536):
                    if block.strip():
                        raise ValueError("Trailing content after JSON array")
                return
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Missing comma in JSON array")
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError("Invalid or truncated JSON array") from error
                    if len(buffer) >= max_record_chars:
                        raise ValueError("Oversized HotpotQA record") from error
                    more()
                    buffer = buffer.lstrip()
            if end > max_record_chars:
                raise ValueError("Oversized HotpotQA record")
            if not isinstance(row, dict):
                raise ValueError("HotpotQA records must be objects")
            yield row
            buffer = buffer[end:]
            first = False


def _beir_identity(directory: Path) -> tuple[str, str]:
    name = directory.name
    manifest = directory / "acquisition.json"
    if manifest.exists():
        name = json.loads(manifest.read_text(encoding="utf-8"))["dataset"]
    return name, BEIR_DATASETS.get(name, {}).get("domain", "unspecified")


def iter_beir_documents(directory: str | Path) -> Iterator[dict]:
    pass                                                                          
    directory = Path(directory)
    source, domain = _beir_identity(directory)
    for row in _json_lines(directory / "corpus.jsonl"):
        raw_id = str(row["_id"])
        title, text = row.get("title", ""), row["text"]
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("BEIR document title and text must be strings")
        yield {"id": f"{source}:{raw_id}", "title": title, "text": text,
               "metadata": {"source": source, "domain": domain, "original_id": raw_id}}


def iter_beir_queries(directory: str | Path, subset: str = "test") -> Iterator[dict]:
    pass                                                                               
    if subset not in {"train", "dev", "test"}:
        raise ValueError("BEIR subset must be train, dev, or test")
    directory = Path(directory)
    source, domain = _beir_identity(directory)
    relevance: dict[str, dict[str, float]] = {}
    with (directory / "qrels" / f"{subset}.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Unexpected BEIR qrels header")
        for row in reader:
            score = float(row["score"])
            if not math.isfinite(score) or score < 0:
                raise ValueError("Qrels grades must be finite and nonnegative")
            judged = relevance.setdefault(str(row["query-id"]), {})
            doc_id = f"{source}:{row['corpus-id']}"
            if doc_id in judged and judged[doc_id] != score:
                raise ValueError("Conflicting duplicate qrels annotation")
            judged[doc_id] = score
    seen: set[str] = set()
    for row in _json_lines(directory / "queries.jsonl"):
        raw_id = str(row["_id"])
        if raw_id not in relevance:
            continue
        if raw_id in seen:
            raise ValueError("Duplicate BEIR query ID")
        seen.add(raw_id)
        if not isinstance(row["text"], str):
            raise ValueError("BEIR query text must be a string")
        yield {"id": f"{source}:{raw_id}", "question": row["text"],
               "relevance": relevance[raw_id], "source": source, "domain": domain,
               "official_split": subset, "corpus_source": source,
               **({"exclude_doc_ids": [f"{source}:{raw_id}"]} if source == "arguana" else {})}
    if seen != relevance.keys():
        raise ValueError("Qrels reference missing query text")


class _OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != "public.ukp.informatik.tu-darmstadt.de":
            raise ValueError("Dataset redirected outside the configured official HTTPS host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_extract_zip(archive: str | Path, destination: str | Path, *,
                     max_uncompressed_bytes: int = 8_000_000_000,
                     max_members: int = 10000) -> None:
    pass                                                                      

                                                                                
                                                                              
       
    destination = Path(destination).resolve()
    if max_uncompressed_bytes < 1 or max_members < 1:
        raise ValueError("Extraction budgets must be positive")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("ZIP extraction destination must be empty")
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > max_members or sum(i.file_size for i in members) > max_uncompressed_bytes:
            raise ValueError("ZIP exceeds extraction budget")
        names: set[str] = set()
        for member in members:
                                                                            
                                                                              
            name = member.orig_filename
            mode = member.external_attr >> 16
            target = _archive_path(name, destination)
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise ValueError("Unsafe ZIP member")
            key = str(target.relative_to(destination)).casefold()
            if key in names:
                raise ValueError("Duplicate ZIP destination")
            names.add(key)
        destination.mkdir(parents=True, exist_ok=True)
        written = 0
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("xb") as output:
                while block := source.read(1024 * 1024):
                    written += len(block)
                    if written > max_uncompressed_bytes:
                        raise ValueError("ZIP stream exceeds extraction budget")
                    output.write(block)


def download_beir(name: str, destination: str | Path, *, max_bytes: int = 1_000_000_000,
                  max_uncompressed_bytes: int = 8_000_000_000, timeout: float = 60,
                  archive_path: str | Path | None = None) -> Path:
    pass                                                                              

                                                                               
                                                                                
                                                                                   
                                                                                
                                                                                   
       
    if name not in BEIR_DATASETS:
        raise ValueError(f"Dataset must be one of {', '.join(BEIR_DATASETS)}")
    if max_bytes < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Download budgets must be positive")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    if target.exists():
        raise FileExistsError(f"Dataset destination exists: {target}")
    spec = BEIR_DATASETS[name]
    url = BEIR_BASE_URL + name + ".zip"
    with tempfile.TemporaryDirectory(prefix=".magra-download-", dir=destination) as temporary:
        staging = Path(temporary)
        archive = Path(archive_path).resolve() if archive_path is not None else staging / "corpus.zip"
        digest = hashlib.md5(usedforsecurity=False)
        received = 0
        if archive_path is not None:
            if archive.stat().st_size > max_bytes:
                raise ValueError("Local archive exceeds compressed byte budget")
            with archive.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    received += len(block)
                    if received > max_bytes:
                        raise ValueError("Local archive exceeds compressed byte budget")
                    digest.update(block)
        else:
            opener = urllib.request.build_opener(_OfficialRedirect())
            request = urllib.request.Request(url, headers={"User-Agent": "MAGRA-benchmark/0.1"})
            with opener.open(request, timeout=timeout) as response, archive.open("xb") as output:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected dataset content encoding")
                length = response.headers.get("Content-Length")
                expected_length = int(length) if length is not None else None
                if expected_length is not None and (expected_length < 0 or expected_length > max_bytes):
                    raise ValueError("Download exceeds compressed byte budget")
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > max_bytes:
                        raise ValueError("Download exceeds compressed byte budget")
                    digest.update(block)
                    output.write(block)
            if expected_length is not None and received != expected_length:
                raise ValueError("Truncated dataset download")
        if digest.hexdigest() != spec["md5"]:
            raise ValueError("Published dataset MD5 mismatch")
        extracted = staging / "extracted"
        safe_extract_zip(archive, extracted, max_uncompressed_bytes=max_uncompressed_bytes)
        corpus = extracted / name
        if not (corpus / "corpus.jsonl").is_file() or not (corpus / "queries.jsonl").is_file():
            raise ValueError("Official dataset layout is missing")
        count = sum(1 for _ in iter_beir_documents(corpus))
        if count != spec["documents"]:
            raise ValueError("Downloaded corpus is incomplete or has unexpected size")
        files = {str(path.relative_to(corpus)).replace("\\", "/"): sha256_file(path)
                 for path in sorted(corpus.rglob("*")) if path.is_file()}
        manifest = {"format": "magra-acquisition-v1", "dataset": name, "domain": spec["domain"],
                    "source_url": url, "source_documentation": BEIR_SOURCE,
                    "archive_md5": digest.hexdigest(), "archive_sha256": sha256_file(archive),
                    "archive_bytes": received, "corpus_documents": count, "complete_corpus": True,
                    "acquisition_method": "verified_local_archive" if archive_path is not None else "official_https",
                    "files_sha256": files}
        (corpus / "acquisition.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        corpus.rename(target)
    return target


def safe_extract_tar(archive: str | Path, destination: str | Path, *,
                     max_uncompressed_bytes: int = 12_000_000_000,
                     max_members: int = 100_000) -> None:
    pass                                                                        

                                                                             
                                                                                
                                                                                
       
    destination = Path(destination).resolve()
    if max_uncompressed_bytes < 1 or max_members < 1:
        raise ValueError("Extraction budgets must be positive")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Tar extraction destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    declared = written = members = 0
    names: set[str] = set()
    with tarfile.open(archive, mode="r|*") as tar:
        for member in tar:
            members += 1
            if members > max_members or member.size < 0:
                raise ValueError("Tar exceeds member budget")
            declared += member.size
            if declared > max_uncompressed_bytes:
                raise ValueError("Tar exceeds extraction byte budget")
            name = member.name
            while name.startswith("./"):
                name = name[2:]
            if member.isdir() and name in {"", "."}:
                continue
            target = _archive_path(name, destination)
            key = str(target.relative_to(destination)).casefold()
            if key in names:
                raise ValueError("Duplicate tar destination")
            names.add(key)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issparse():
                raise ValueError("Tar links, devices, and sparse files are forbidden")
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Missing tar member content")
            target.parent.mkdir(parents=True, exist_ok=True)
            file_written = 0
            with source, target.open("xb") as output:
                while block := source.read(1024 * 1024):
                    file_written += len(block)
                    written += len(block)
                    if written > max_uncompressed_bytes or file_written > member.size:
                        raise ValueError("Tar stream exceeds extraction budget")
                    output.write(block)
            if file_written != member.size:
                raise ValueError("Truncated tar member")


class _HotpotRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {"curtis.ml.cmu.edu", "nlp.stanford.edu"}:
            raise ValueError("HotpotQA redirected outside configured official HTTPS hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_hotpot(name: str, destination: str | Path, *, max_bytes: int = 2_000_000_000,
                    max_uncompressed_bytes: int = 12_000_000_000, timeout: float = 60,
                    expected_sha256: str | None = None,
                    archive_path: str | Path | None = None) -> Path:
    pass                                                                        

                                                                            
                                                                              
                                                                                
                                                                           
                                                                   
                                                                              
                                                                           
       
    if name not in HOTPOT_DATASETS:
        raise ValueError(f"Hotpot dataset must be one of {', '.join(HOTPOT_DATASETS)}")
    if max_bytes < 1 or max_uncompressed_bytes < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Download and extraction budgets must be positive")
    if expected_sha256 is not None and (len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Expected SHA256 must be a lowercase 64-character hex digest")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spec = HOTPOT_DATASETS[name]
    wiki = name == "wiki"
    target = destination / ("hotpotqa-wikipedia" if wiki else spec["filename"])
    manifest_target = target / "acquisition.json" if wiki else target.with_suffix(target.suffix + ".acquisition.json")
    if target.exists() or manifest_target.exists():
        raise FileExistsError(f"Hotpot acquisition destination exists: {target}")
    if wiki and spec["bytes"] > max_bytes:
        raise ValueError("Official wiki archive exceeds compressed byte budget")
    url = HOTPOT_WIKI_URL if wiki else HOTPOT_QUESTIONS_BASE_URL + spec["filename"]
    with tempfile.TemporaryDirectory(prefix=".magra-hotpot-", dir=destination) as temporary:
        stage = Path(temporary)
        downloaded = Path(archive_path).resolve() if archive_path is not None and wiki else stage / spec["filename"]
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        received = 0
        if archive_path is not None:
            local = Path(archive_path).resolve()
            if local.stat().st_size > max_bytes:
                raise ValueError("Local HotpotQA input exceeds compressed byte budget")
            with local.open("rb") as source, (nullcontext(None) if wiki else downloaded.open("xb")) as output:
                while block := source.read(1024 * 1024):
                    received += len(block)
                    if received > max_bytes:
                        raise ValueError("Local HotpotQA input exceeds compressed byte budget")
                    md5.update(block)
                    sha256.update(block)
                    if output is not None:
                        output.write(block)
        else:
            opener = urllib.request.build_opener(_HotpotRedirect())
            request = urllib.request.Request(url, headers={"User-Agent": "MAGRA-benchmark/0.1"})
            with opener.open(request, timeout=timeout) as response, downloaded.open("xb") as output:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected dataset content encoding")
                length = response.headers.get("Content-Length")
                expected_length = int(length) if length is not None else None
                if expected_length is not None and (expected_length < 0 or expected_length > max_bytes):
                    raise ValueError("Download exceeds compressed byte budget")
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > max_bytes:
                        raise ValueError("Download exceeds compressed byte budget")
                    md5.update(block)
                    sha256.update(block)
                    output.write(block)
            if expected_length is not None and received != expected_length:
                raise ValueError("Truncated dataset download")
        if expected_sha256 is not None and sha256.hexdigest() != expected_sha256:
            raise ValueError("Pinned HotpotQA SHA256 mismatch")
        manifest = {"format": "magra-acquisition-v1", "dataset": "hotpotqa", "variant": name,
                    "domain": "encyclopedia", "source_url": url, "source_documentation": HOTPOT_SOURCE,
                    "download_sha256": sha256.hexdigest(), "download_bytes": received,
                    "acquisition_method": "verified_local_input" if archive_path is not None else "official_https",
                    "license": "CC-BY-SA-4.0", "complete_corpus": False}
        if wiki:
            if received != spec["bytes"] or md5.hexdigest() != spec["md5"]:
                raise ValueError("Published HotpotQA wiki size or MD5 mismatch")
            extracted = stage / "wiki"
            safe_extract_tar(downloaded, extracted, max_uncompressed_bytes=max_uncompressed_bytes)
            count = sum(1 for _ in iter_hotpot_wikipedia(extracted))
            if count < 1:
                raise ValueError("Official wiki archive has no corpus documents")
            files = {str(path.relative_to(extracted)).replace("\\", "/"): sha256_file(path)
                     for path in sorted(extracted.rglob("*")) if path.is_file()}
            manifest.update({"archive_md5": md5.hexdigest(), "corpus_documents": count,
                             "complete_corpus": True, "setting": "full_corpus",
                             "snapshot_id": "enwiki-2017-10-01-hotpot-intro", "files_sha256": files,
                             "source_documentation": "https://hotpotqa.github.io/wiki-readme.html"})
            (extracted / "acquisition.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            extracted.rename(target)
        else:
            count = 0
            ids: set[str] = set()
            for row in _json_array(downloaded):
                if not isinstance(row.get("question"), str) or not row["question"].strip() or "_id" not in row:
                    raise ValueError("Invalid official HotpotQA question schema")
                query_id = str(row["_id"])
                if query_id in ids:
                    raise ValueError("Duplicate official HotpotQA query ID")
                ids.add(query_id)
                count += 1
            if count != spec["queries"]:
                raise ValueError("Downloaded HotpotQA question set is incomplete or unexpected")
            manifest.update({"query_records": count, "publisher_checksum_available": False,
                             "sha256_pinned": expected_sha256 is not None})
            temporary_manifest = stage / "acquisition.json"
            temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            downloaded.rename(target)
            temporary_manifest.rename(manifest_target)
    return target


def hotpot_document_id(title: str, question_id: str | None = None) -> str:
    pass                                                                      

                                                                             
                                                                         
       
    title = unicodedata.normalize("NFC", title)
    encoded = urllib.parse.quote(title, safe="")
    if question_id is None:
        return "hotpotqa:wiki:" + encoded
    return "hotpotqa:context:" + urllib.parse.quote(str(question_id), safe="") + ":" + encoded


def _hotpot_rows(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if path.suffix in {".jsonl", ".bz2"}:
        yield from _json_lines(path)
    else:
        yield from _json_array(path)


def _setting(setting: str) -> None:
    if setting not in {"provided_context", "full_corpus"}:
        raise ValueError("Hotpot setting must be provided_context or full_corpus")


def iter_hotpot(path: str | Path, setting: str = "provided_context", *,
                 subset: str | None = None) -> Iterator[dict]:
    pass                                                                           

                                                                              
                                                                                 
       
    _setting(setting)
    for row in _hotpot_rows(path):
        raw_id = str(row["_id"])
        document_id = lambda title: hotpot_document_id(title, raw_id if setting == "provided_context" else None)
        supports = [{"doc_id": document_id(title), "sentence_id": int(sentence)}
                    for title, sentence in row.get("supporting_facts", [])]
        if any(fact["sentence_id"] < 0 for fact in supports):
            raise ValueError("Supporting sentence IDs must be nonnegative")
        result = {"id": f"hotpotqa:{setting}:{raw_id}", "question": row["question"],
                  "source": "hotpotqa", "domain": "encyclopedia", "setting": setting,
                  "group_id": f"hotpotqa:{raw_id}", "supporting_facts": supports,
                  "relevance": {fact["doc_id"]: 1 for fact in supports}}
        if "answer" in row:
            result["answer"] = row["answer"]
        if setting == "provided_context":
            result["context_doc_ids"] = [document_id(title) for title, _ in row.get("context", [])]
        if subset is not None:
            if subset not in {"train", "dev", "test"}:
                raise ValueError("Hotpot subset must be train, dev, or test")
            result["official_split"] = subset
        yield result


def iter_hotpot_documents(path: str | Path, setting: str = "provided_context") -> Iterator[dict]:
    pass                                                                           
    _setting(setting)
    if setting == "full_corpus":
        raise ValueError("Import independent Wikipedia using iter_hotpot_wikipedia; a context union is not full corpus")
    for row in _hotpot_rows(path):
        raw_id = str(row["_id"])
        for title, sentences in row.get("context", []):
            if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
                raise ValueError("Hotpot context sentences must be strings")
            yield {"id": hotpot_document_id(title, raw_id), "title": title,
                   "text": "".join(sentences), "metadata": {"source": "hotpotqa",
                   "domain": "encyclopedia", "context_id": raw_id,
                   "sentences": sentences}}


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _strip_links(sentence: str) -> str:
    parser = _PlainText()
    parser.feed(sentence)
    parser.close()
    return "".join(parser.parts)


def iter_hotpot_wikipedia(path: str | Path) -> Iterator[dict]:
    pass                                                                          

                                                                              
                                                                               
                                                                                  
       
    path = Path(path)
    paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".bz2", ".jsonl"}) if path.is_dir() else [path]
    if not paths:
        raise ValueError("No Wikipedia JSONL/bz2 shards found")
    for shard in paths:
        for row in _json_lines(shard):
            sentences = row["text"]
            if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
                raise ValueError("Import the Hotpot introductory-paragraph corpus, not nested full articles")
            sentences = [_strip_links(sentence) for sentence in sentences]
            yield {"id": hotpot_document_id(row["title"]), "title": row["title"],
                   "text": "".join(sentences), "metadata": {"source": "hotpotqa",
                   "domain": "encyclopedia", "sentences": sentences,
                   "original_id": str(row.get("id", "")), "url": row.get("url", ""),
                   "snapshot_id": "enwiki-2017-10-01-hotpot-intro"}}
