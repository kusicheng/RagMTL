                                                                                  

                                                                                 
                                                                               
                                                                             
                                                                             
                                                                      
   

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import queue
import re
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit


MAX_FOOTPRINT = 2_000_000_000_000
MAX_RESPONSE_BYTES = 16_000_000
MAX_SNAPSHOTS = 10_000
_REDIRECTS = {301, 302, 303, 307, 308}
_TEXT_TYPES = {"text/plain", "text/html", "application/xhtml+xml", "application/json"}
_HASH = re.compile(r"^[0-9a-f]{64}$")


class WebError(RuntimeError):
    pass                                                                 


class WebPolicyError(WebError):
    pass


class WebBudgetError(WebError):
    pass


class WebNetworkError(WebError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _host(value: str) -> str:
    try:
        normalized = value.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise WebPolicyError("invalid hostname") from error
    if not normalized or any(character in normalized for character in "/\\@%\r\n\t "):
        raise WebPolicyError("invalid hostname")
    try:
        return str(ipaddress.ip_address(normalized.strip("[]")))
    except ValueError:
        if len(normalized) > 253 or any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in normalized.split(".")
        ):
            raise WebPolicyError("invalid hostname")
        return normalized


def _public_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise WebPolicyError("DNS returned an invalid address") from error
    if (
        not address.is_global or address.is_multicast or address.is_unspecified
        or address.is_loopback or address.is_link_local or address.is_reserved
        or getattr(address, "ipv4_mapped", None) is not None
        or (address.version == 6 and (
            getattr(address, "sixtofour", None) is not None
            or getattr(address, "teredo", None) is not None
            or address in ipaddress.ip_network("64:ff9b::/96")
            or address in ipaddress.ip_network("64:ff9b:1::/48")
        ))
    ):
        raise WebPolicyError("non-public destination address is forbidden")
    return str(address)


def _resolve(hostname: str, port: int) -> Iterable[str]:
    return [row[4][0] for row in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)]


def _pinned_socket(address: str, port: int, timeout: float) -> socket.socket:
                                                                                
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    connection = socket.socket(family, socket.SOCK_STREAM)
    try:
        connection.settimeout(timeout)
        connection.connect((address, port))
        if ipaddress.ip_address(connection.getpeername()[0]) != ipaddress.ip_address(address):
            raise WebPolicyError("connected peer does not match pinned address")
        return connection
    except BaseException:
        connection.close()
        raise


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, hostname: str, port: int, address: str, timeout: float):
        super().__init__(hostname, port=port, timeout=timeout)
        self.address = address

    def connect(self) -> None:
        self.sock = _pinned_socket(self.address, self.port, self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, port: int, address: str, timeout: float):
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self) -> None:
        raw = _pinned_socket(self.address, self.port, self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []
        self.title_fragments: list[str] = []
        self.hidden = 0
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.fragments.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False
        if tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.fragments.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.fragments.append(data)
            if self.in_title:
                self.title_fragments.append(data)


@dataclass(frozen=True)
class WebResponse:
    original_url: str
    final_url: str
    status: int
    body: bytes
    content_type: str
    fetched_at: str
    redirect_chain: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    last_modified: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    def to_document(self, *, track: str = "live_web", document_date: str | None = None) -> dict[str, Any]:
        header = Message()
        header["content-type"] = self.content_type
        try:
            decoded = self.body.decode(header.get_content_charset() or "utf-8", errors="replace")
        except LookupError:
            decoded = self.body.decode("utf-8", errors="replace")
        title = urlsplit(self.final_url).hostname or self.final_url
        if header.get_content_type() in {"text/html", "application/xhtml+xml"}:
            parser = _TextExtractor()
            parser.feed(decoded)
            title = " ".join("".join(parser.title_fragments).split()) or title
            decoded = "\n".join(line for line in (" ".join(line.split()) for line in "".join(parser.fragments).splitlines()) if line)
        return {
            "id": f"web:{self.sha256}", "title": title, "text": decoded,
            "metadata": {
                "track": track, "source_url": self.original_url, "final_url": self.final_url,
                "fetched_at": self.fetched_at, "sha256": self.sha256,
                "document_date": document_date, "content_type": self.content_type,
                "last_modified": self.last_modified,
            },
        }


class BoundedWebClient:
    pass                                                                         

                                                                                
                                                                              
                                                                     
       

    def __init__(
        self, allowed_domains: Iterable[str], *, timeout: float = 10.0,
        max_requests: int = 8, max_bytes: int = 10_000_000,
        max_response_bytes: int = 2_000_000, max_redirects: int = 3,
        resolver: Callable[[str, int], Iterable[str]] | None = None,
    ):
        self.allowed_domains = frozenset(_host(domain) for domain in allowed_domains)
        if not self.allowed_domains:
            raise ValueError("at least one explicitly allowed domain is required")
        if not 0 < timeout <= 120:
            raise ValueError("timeout must be greater than zero and at most 120 seconds")
        if not 1 <= max_requests <= 1000 or not 0 <= max_redirects <= 20:
            raise ValueError("invalid request or redirect budget")
        if not 0 < max_response_bytes <= min(max_bytes, MAX_RESPONSE_BYTES) or max_bytes > MAX_FOOTPRINT:
            raise ValueError("invalid byte budget")
        self.timeout = timeout
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects
        self.resolver = resolver or _resolve
        self.requests_used = 0
        self.bytes_used = 0

    def _validate_url(self, url: str) -> tuple[str, str, int]:
        if len(url) > 8192:
            raise WebPolicyError("URL exceeds length bound")
        if any(ord(character) < 32 or ord(character) == 127 for character in url):
            raise WebPolicyError("control characters in URL are forbidden")
        try:
            parts = urlsplit(url)
            port = parts.port
            hostname = parts.hostname
        except ValueError as error:
            raise WebPolicyError("invalid URL") from error
        if parts.scheme not in {"http", "https"} or not hostname:
            raise WebPolicyError("only absolute HTTP and HTTPS URLs are permitted")
        if parts.username is not None or parts.password is not None:
            raise WebPolicyError("URL credentials are forbidden")
        hostname = _host(hostname)
        if hostname not in self.allowed_domains:
            raise WebPolicyError("destination is outside the exact domain allowlist")
        expected_port = 443 if parts.scheme == "https" else 80
        if port is not None and port != expected_port:
            raise WebPolicyError("nonstandard destination ports are forbidden")
        authority = f"[{hostname}]" if ":" in hostname else hostname
        normalized = urlunsplit((parts.scheme, authority, parts.path or "/", parts.query, ""))
        return normalized, hostname, expected_port

    def _addresses(self, hostname: str, port: int, deadline: float) -> list[str]:
                                                                              
                                                                             
                                                                                
        result: queue.Queue[Any] = queue.Queue(maxsize=1)

        def lookup() -> None:
            try:
                addresses: list[str] = []
                for address in self.resolver(hostname, port):
                    if len(addresses) >= 64:
                        raise WebPolicyError("DNS answer exceeds address bound")
                    addresses.append(address)
                result.put((True, addresses))
            except Exception as error:
                result.put((False, error))

        threading.Thread(target=lookup, daemon=True, name="magra-bounded-dns").start()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WebNetworkError("request deadline exceeded during DNS resolution")
        try:
            success, value = result.get(timeout=remaining)
        except queue.Empty as error:
            raise WebNetworkError("request deadline exceeded during DNS resolution") from error
        if not success:
            raise value
        return sorted({_public_address(address) for address in value})

    def _request(self, url: str, address: str, deadline: float) -> tuple[int, dict[str, str], bytes]:
        parts = urlsplit(url)
        port = 443 if parts.scheme == "https" else 80
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WebNetworkError("request deadline exceeded")
        connection_type = _PinnedHTTPSConnection if parts.scheme == "https" else _PinnedHTTPConnection
        connection = connection_type(parts.hostname, port, address, remaining)
        response = None
        chunks: list[bytes] = []
        consumed = 0
        try:
            path = urlunsplit(("", "", parts.path or "/", parts.query, ""))
            connection.request("GET", path, headers={"User-Agent": "MagraBenchmark/1.0", "Accept-Encoding": "identity", "Accept": "text/plain,text/html,application/json,application/xhtml+xml"})
            active_socket = connection.sock
            if active_socket:
                active_socket.settimeout(max(0.001, deadline - time.monotonic()))
            response = connection.getresponse()
            headers = {key.lower(): value for key, value in response.getheaders()}
            if response.status in _REDIRECTS:
                return response.status, headers, b""
            if not 200 <= response.status < 300:
                raise WebNetworkError(f"HTTP status {response.status}")
            if headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
                raise WebPolicyError("compressed responses are forbidden")
            content_type = headers.get("content-type", "application/octet-stream").split(";", 1)[0].strip().lower()
            if content_type not in _TEXT_TYPES:
                raise WebPolicyError("only bounded textual content is accepted")
            limit = min(self.max_response_bytes, self.max_bytes - self.bytes_used)
            length = headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as error:
                    raise WebPolicyError("invalid Content-Length") from error
                if declared < 0:
                    raise WebPolicyError("invalid Content-Length")
                if declared > limit:
                    raise WebBudgetError("response exceeds byte budget")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WebNetworkError("request deadline exceeded")
                if active_socket:
                    active_socket.settimeout(remaining)
                if consumed >= limit:
                    if response.length == 0 or response.isclosed():
                        break
                                                                            
                                                                              
                    raise WebBudgetError("response reached streaming byte budget")
                                                                           
                                                                              
                chunk = response.read1(min(65_536, limit - consumed))
                if not chunk:
                    break
                consumed += len(chunk)
                self.bytes_used += len(chunk)
                if consumed > limit:
                    raise WebBudgetError("response exceeds byte budget")
                chunks.append(chunk)
            return response.status, headers, b"".join(chunks)
        finally:
            if response is not None:
                response.close()
            connection.close()

    def fetch(self, url: str) -> WebResponse:
        original, _, _ = self._validate_url(url)
        current = original
        redirects: list[str] = []
        started = time.monotonic()
        deadline = started + self.timeout
        for hop in range(self.max_redirects + 1):
            current, hostname, port = self._validate_url(current)
            if self.requests_used >= self.max_requests:
                raise WebBudgetError("request budget exhausted")
            if self.bytes_used >= self.max_bytes:
                raise WebBudgetError("total response-byte budget exhausted")
            self.requests_used += 1
            try:
                addresses = self._addresses(hostname, port, deadline)
                if not addresses:
                    raise WebNetworkError("DNS returned no addresses")
                if time.monotonic() >= deadline:
                    raise WebNetworkError("request deadline exceeded during DNS resolution")
                status, headers, body = self._request(current, addresses[0], deadline)
            except WebError:
                raise
            except (OSError, http.client.HTTPException, UnicodeError, ValueError) as error:
                raise WebNetworkError(f"{type(error).__name__}: {error}") from error
            if status in _REDIRECTS:
                if hop >= self.max_redirects:
                    raise WebBudgetError("redirect budget exhausted")
                location = headers.get("location")
                if not location:
                    raise WebNetworkError("redirect response has no Location")
                redirects.append(current)
                current = urljoin(current, location)
                continue
            return WebResponse(
                original_url=original, final_url=current, status=status,
                body=body, content_type=headers.get("content-type", "text/plain"),
                fetched_at=_now(), redirect_chain=tuple(redirects),
                elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                last_modified=headers.get("last-modified"),
            )
        raise AssertionError("unreachable redirect loop")

    def fetch_document(self, url: str) -> dict[str, Any]:
        return self.fetch(url).to_document()


class SnapshotStore:
    pass                                                                            

                                                                             
                                                                                
                                                                                
       

    def __init__(self, root: str | Path, *, max_bytes: int = 1_000_000_000):
        if not 0 < max_bytes <= MAX_FOOTPRINT:
            raise ValueError("snapshot quota must be in (0, 2 TB decimal]")
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("blobs", "snapshots"):
            path = self.root / name
            path.mkdir(exist_ok=True)
            if not path.resolve().is_relative_to(self.root):
                raise WebPolicyError("snapshot directory escapes its root")

    def _used(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise WebPolicyError("symlinks are forbidden inside snapshot storage")
            if path.is_file():
                total += path.stat().st_size
        return total

    def manifest(self) -> dict[str, Any]:
        entries = []
        for path in sorted((self.root / "snapshots").glob("*.json")):
            if len(entries) >= MAX_SNAPSHOTS:
                raise WebBudgetError("snapshot count exceeds bound")
            if not _HASH.fullmatch(path.stem) or path.is_symlink():
                raise WebPolicyError("invalid snapshot metadata path")
            if path.stat().st_size > 65_536:
                raise WebBudgetError("snapshot metadata exceeds limit")
            raw = path.read_bytes()
            record = json.loads(raw)
            if hashlib.sha256(raw).hexdigest() != path.stem:
                raise WebPolicyError("snapshot metadata integrity check failed")
            entries.append({"snapshot_id": path.stem, **record})
        return {"version": 1, "track": "frozen_web", "snapshots": entries}

    def save(self, response: WebResponse, *, document_date: str | None = None) -> dict[str, Any]:
        self._used()
        lock_path = self.root / ".writer.lock"
        try:
            lock = lock_path.open("xb")
        except FileExistsError as error:
            raise WebPolicyError("snapshot store already has a writer lock") from error
        try:
            with lock:
                return self._save(response, document_date=document_date)
        finally:
            lock_path.unlink()

    def _save(self, response: WebResponse, *, document_date: str | None = None) -> dict[str, Any]:
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise WebBudgetError("snapshot exceeds maximum response bound")
        if document_date is not None:
            date.fromisoformat(document_date)
        timestamp = datetime.fromisoformat(response.fetched_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("fetched_at must include a timezone")
        record = {
            "original_url": response.original_url, "final_url": response.final_url,
            "fetched_at": response.fetched_at, "sha256": response.sha256,
            "document_date": document_date, "content_type": response.content_type,
            "status": response.status, "bytes": len(response.body),
            "redirect_chain": list(response.redirect_chain), "last_modified": response.last_modified,
        }
        raw_record = _canonical_json(record)
        if len(raw_record) > 65_536:
            raise WebBudgetError("snapshot metadata exceeds limit")
        snapshot_id = hashlib.sha256(raw_record).hexdigest()
        blob_path = self.root / "blobs" / f"{response.sha256}.bin"
        record_path = self.root / "snapshots" / f"{snapshot_id}.json"
        manifest = self.manifest()
        if not record_path.exists():
            if len(manifest["snapshots"]) >= MAX_SNAPSHOTS:
                raise WebBudgetError("snapshot count exceeds bound")
            manifest["snapshots"].append({"snapshot_id": snapshot_id, **record})
            manifest["snapshots"].sort(key=lambda entry: entry["snapshot_id"])
        raw_manifest = _canonical_json(manifest)
        used = self._used()
        additional = (0 if blob_path.exists() else len(response.body)) + (0 if record_path.exists() else len(raw_record)) + len(raw_manifest)
        if used + additional > self.max_bytes:
            raise WebBudgetError("snapshot physical-footprint budget exhausted")
        if blob_path.exists():
            if blob_path.stat().st_size != len(response.body) or hashlib.sha256(blob_path.read_bytes()).hexdigest() != response.sha256:
                raise WebPolicyError("existing snapshot blob integrity check failed")
        if not blob_path.exists():
            with blob_path.open("xb") as handle:
                handle.write(response.body)
        if not record_path.exists():
            with record_path.open("xb") as handle:
                handle.write(raw_record)
        temporary_manifest = self.root / "manifest.pending"
        with temporary_manifest.open("xb") as handle:
            handle.write(raw_manifest)
        os.replace(temporary_manifest, self.root / "manifest.json")
        return self.load(snapshot_id)

    def capture(self, url: str, client: BoundedWebClient, *, document_date: str | None = None) -> dict[str, Any]:
        return self.save(client.fetch(url), document_date=document_date)

    def load(self, snapshot_id: str) -> dict[str, Any]:
        if not _HASH.fullmatch(snapshot_id):
            raise ValueError("invalid snapshot id")
        self._used()                                                              
        metadata_path = self.root / "snapshots" / f"{snapshot_id}.json"
        if metadata_path.stat().st_size > 65_536:
            raise WebBudgetError("snapshot metadata exceeds limit")
        raw = metadata_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != snapshot_id:
            raise WebPolicyError("snapshot metadata integrity check failed")
        record = json.loads(raw)
        digest = record["sha256"]
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise WebPolicyError("invalid snapshot blob id")
        blob_path = self.root / "blobs" / f"{digest}.bin"
        if blob_path.stat().st_size > min(self.max_bytes, MAX_RESPONSE_BYTES):
            raise WebBudgetError("snapshot blob exceeds storage budget")
        body = blob_path.read_bytes()
        if len(body) != record["bytes"] or hashlib.sha256(body).hexdigest() != digest:
            raise WebPolicyError("snapshot blob integrity check failed")
        response = WebResponse(
            original_url=record["original_url"], final_url=record["final_url"],
            status=record["status"], body=body, content_type=record["content_type"],
            fetched_at=record["fetched_at"], redirect_chain=tuple(record["redirect_chain"]),
            last_modified=record["last_modified"],
        )
        document = response.to_document(track="frozen_web", document_date=record["document_date"])
        document["id"] = f"web:{snapshot_id}"
        document["metadata"]["snapshot_id"] = snapshot_id
        return document


class LiveWebTrack:
    pass                                                                             

    def __init__(self, client: BoundedWebClient):
        self.client = client
        self.records: list[dict[str, Any]] = []

    def acquire(self, url: str, *, query_id: str | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "track": "live_web", "query_id": query_id, "url": url,
            "attempted_at": _now(), "replayable": False, "document": None,
        }
        try:
            response = self.client.fetch(url)
            record.update(status="ok", document=response.to_document(), elapsed_ms=response.elapsed_ms)
        except WebError as error:
            status = "network_error"
            if isinstance(error, WebPolicyError):
                status = "policy_blocked"
            elif isinstance(error, WebBudgetError):
                status = "budget_exhausted"
            record.update(status=status, error={"type": type(error).__name__, "message": str(error)})
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        counts = {status: sum(record["status"] == status for record in self.records) for status in ("ok", "network_error", "policy_blocked", "budget_exhausted")}
        network_attempts = counts["ok"] + counts["network_error"]
        return {
            "track": "live_web", "attempts": len(self.records), **counts,
            "network_availability": counts["ok"] / network_attempts if network_attempts else None,
            "replayable": False,
        }
