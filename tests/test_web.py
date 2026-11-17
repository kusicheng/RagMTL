import hashlib
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from magra_rag.web import (
    BoundedWebClient, LiveWebTrack, SnapshotStore, WebBudgetError,
    WebNetworkError, WebPolicyError, WebResponse, _pinned_socket,
)


class FakeResponse:
    def __init__(self, body=b"Evidence.", *, status=200, headers=None, known_length=True):
        self.status = status
        self.body = body
        self.offset = 0
        self.length = len(body) if known_length else None
        self.closed = False
        self.headers = {"content-type": "text/plain; charset=utf-8", **(headers or {})}
        if known_length:
            self.headers.setdefault("content-length", str(len(body)))

    def getheaders(self):
        return list(self.headers.items())

    def read1(self, size):
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        if self.length is not None:
            self.length -= len(chunk)
        if not chunk:
            self.closed = True
        return chunk

    def isclosed(self):
        return self.closed

    def close(self):
        self.closed = True


def connection_factory(responses, calls):
    class FakeConnection:
        def __init__(self, hostname, port, address, timeout):
            calls.append({"hostname": hostname, "port": port, "address": address, "timeout": timeout})
            self.sock = mock.Mock()

        def request(self, method, path, headers):
            calls[-1].update(method=method, path=path, headers=headers)

        def getresponse(self):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def close(self):
            pass
    return FakeConnection


def public_resolver(hostname, port):
    return ["8.8.8.8"]


class BoundedWebTests(unittest.TestCase):
    def client(self, **kwargs):
        return BoundedWebClient(["example.com"], resolver=public_resolver, **kwargs)

    def test_fetch_pins_public_address_preserves_host_and_extracts_text(self):
        calls = []
        body = b"<html><title>Frozen title</title><script>bad()</script><p>Owned evidence &amp; facts.</p></html>"
        response = FakeResponse(body, headers={"content-type": "text/html; charset=utf-8"})
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([response], calls)):
            client = self.client()
            document = client.fetch_document("https://EXAMPLE.com/article?q=1#fragment")
        self.assertEqual(calls[0]["hostname"], "example.com")
        self.assertEqual(calls[0]["address"], "8.8.8.8")
        self.assertEqual(calls[0]["path"], "/article?q=1")
        self.assertEqual(calls[0]["headers"]["Accept-Encoding"], "identity")
        self.assertEqual(document["title"], "Frozen title")
        self.assertIn("Owned evidence & facts.", document["text"])
        self.assertNotIn("bad()", document["text"])
        self.assertEqual(document["metadata"]["sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(client.bytes_used, len(body))

    def test_exact_allowlist_credentials_schemes_ports_and_controls(self):
        for url in (
            "https://example.com.evil.test/", "https://sub.example.com/",
            "https://user:password@example.com/", "file:///etc/passwd",
            "http://example.com:8080/", "https://example.com/\r\nHost: evil",
        ):
            with self.subTest(url=url), self.assertRaises(WebPolicyError):
                self.client().fetch(url)

    def test_nonpublic_and_mixed_dns_answers_are_rejected_before_connection(self):
        unsafe = ["127.0.0.1", "10.1.2.3", "169.254.169.254", "::1", "::ffff:8.8.8.8", "64:ff9b::7f00:1", "224.0.0.1"]
        for address in unsafe:
            client = BoundedWebClient(["example.com"], resolver=lambda host, port, address=address: ["8.8.8.8", address])
            with self.subTest(address=address), mock.patch("magra_rag.web._PinnedHTTPSConnection") as connection:
                with self.assertRaises(WebPolicyError):
                    client.fetch("https://example.com/")
                connection.assert_not_called()

    def test_every_redirect_is_rechecked_for_domain_and_public_dns(self):
        calls = []
        first = FakeResponse(status=302, headers={"location": "https://private.test/secret"})
        client = BoundedWebClient(["example.com", "private.test"], resolver=lambda host, port: ["10.0.0.1"] if host == "private.test" else ["8.8.8.8"])
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([first], calls)):
            with self.assertRaises(WebPolicyError):
                client.fetch("https://example.com/")
        self.assertEqual(len(calls), 1)
        second = FakeResponse(status=302, headers={"location": "https://evil.test/"})
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([second], [])):
            with self.assertRaises(WebPolicyError):
                self.client().fetch("https://example.com/")

    def test_relative_redirect_is_bounded_and_provenance_is_kept(self):
        calls = []
        responses = [FakeResponse(status=302, headers={"location": "/final"}), FakeResponse(b"Final evidence")]
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory(responses, calls)):
            response = self.client().fetch("https://example.com/start")
        self.assertEqual(response.original_url, "https://example.com/start")
        self.assertEqual(response.final_url, "https://example.com/final")
        self.assertEqual(response.redirect_chain, ("https://example.com/start",))
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([FakeResponse(status=302, headers={"location": "/again"})], [])):
            with self.assertRaises(WebBudgetError):
                self.client(max_redirects=0).fetch("https://example.com/")

    def test_streaming_never_reads_past_response_or_total_byte_cap(self):
        response = FakeResponse(b"123456789", known_length=False)
        client = self.client(max_bytes=5, max_response_bytes=5)
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([response], [])):
            with self.assertRaises(WebBudgetError):
                client.fetch("https://example.com/")
        self.assertEqual(response.offset, 5)
        self.assertEqual(client.bytes_used, 5)
        with self.assertRaises(WebBudgetError):
            client.fetch("https://example.com/")
        client = self.client(max_bytes=3, max_response_bytes=3)
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([FakeResponse(b"12"), FakeResponse(b"34")], [])):
            self.assertEqual(client.fetch("https://example.com/").body, b"12")
            with self.assertRaises(WebBudgetError):
                client.fetch("https://example.com/")
        self.assertEqual(client.bytes_used, 2)

    def test_request_budget_includes_failed_requests(self):
        client = self.client(max_requests=1)
        with mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([socket.timeout("offline")], [])):
            with self.assertRaises(WebNetworkError):
                client.fetch("https://example.com/")
        self.assertEqual(client.requests_used, 1)
        with self.assertRaises(WebBudgetError):
            client.fetch("https://example.com/")

    def test_dns_is_deadline_bounded_without_live_network(self):
        finished = threading.Event()
        def slow_dns(hostname, port):
            finished.wait(1)
            return ["8.8.8.8"]
        client = BoundedWebClient(["example.com"], resolver=slow_dns, timeout=0.03)
        started = time.monotonic()
        try:
            with self.assertRaises(WebNetworkError):
                client.fetch("https://example.com/")
        finally:
            finished.set()
        self.assertLess(time.monotonic() - started, 0.5)

    def test_compression_binary_status_and_declared_oversize_are_rejected(self):
        cases = [
            (FakeResponse(headers={"content-encoding": "gzip"}), WebPolicyError),
            (FakeResponse(headers={"content-type": "application/octet-stream"}), WebPolicyError),
            (FakeResponse(status=503), WebNetworkError),
            (FakeResponse(b"123456"), WebBudgetError),
        ]
        for response, error in cases:
            with self.subTest(error=error), mock.patch("magra_rag.web._PinnedHTTPSConnection", connection_factory([response], [])):
                client = self.client(max_bytes=5, max_response_bytes=5)
                with self.assertRaises(error):
                    client.fetch("https://example.com/")
                self.assertEqual(client.bytes_used, 0)

    def test_socket_connection_never_re_resolves_the_approved_address(self):
        fake_socket = mock.Mock()
        fake_socket.getpeername.return_value = ("8.8.8.8", 443)
        with mock.patch("magra_rag.web.socket.socket", return_value=fake_socket), mock.patch("magra_rag.web.socket.getaddrinfo") as dns:
            self.assertIs(_pinned_socket("8.8.8.8", 443, 1), fake_socket)
            fake_socket.connect.assert_called_once_with(("8.8.8.8", 443))
            dns.assert_not_called()

    def test_live_failures_are_recorded_and_scored_separately(self):
        client = mock.Mock()
        client.fetch.side_effect = [WebNetworkError("timeout"), WebPolicyError("private"), WebBudgetError("limit"), WebResponse("https://example.com/", "https://example.com/", 200, b"Evidence", "text/plain", "2025-01-01T00:00:00Z")]
        track = LiveWebTrack(client)
        for number in range(4):
            record = track.acquire("https://example.com/", query_id=str(number))
            self.assertEqual(record["track"], "live_web")
            self.assertFalse(record["replayable"])
        self.assertEqual([record["status"] for record in track.records], ["network_error", "policy_blocked", "budget_exhausted", "ok"])
        self.assertEqual(track.summary()["network_availability"], 0.5)
        self.assertNotIn("answer_quality", track.summary())


class SnapshotTests(unittest.TestCase):
    def response(self, fetched_at="2025-01-01T00:00:00Z"):
        return WebResponse("https://example.com/original", "https://example.com/final", 200, b"Frozen evidence", "text/plain", fetched_at, ("https://example.com/original",), last_modified="Tue, 31 Dec 2024 00:00:00 GMT")

    def test_snapshot_is_replayable_content_addressed_and_manifest_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory)
            first = store.save(self.response(), document_date="2024-12-31")
            manifest = (Path(directory) / "manifest.json").read_bytes()
            second = store.save(self.response(), document_date="2024-12-31")
            self.assertEqual(first, second)
            self.assertEqual(manifest, (Path(directory) / "manifest.json").read_bytes())
            self.assertEqual(store.load(first["metadata"]["snapshot_id"]), first)
            self.assertEqual(first["text"], "Frozen evidence")
            self.assertEqual(first["metadata"]["track"], "frozen_web")
            self.assertEqual(first["metadata"]["source_url"], "https://example.com/original")
            self.assertEqual(first["metadata"]["document_date"], "2024-12-31")
            self.assertEqual(first["metadata"]["fetched_at"], "2025-01-01T00:00:00Z")
            later = store.save(self.response("2025-01-02T00:00:00Z"))
            self.assertNotEqual(later["id"], first["id"])
            self.assertEqual(len(list((Path(directory) / "blobs").glob("*.bin"))), 1)
            self.assertEqual(len(store.manifest()["snapshots"]), 2)

    def test_quota_checks_peak_before_writing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory, max_bytes=20)
            with self.assertRaises(WebBudgetError):
                store.save(self.response())
            self.assertEqual(list((Path(directory) / "blobs").iterdir()), [])
            self.assertFalse((Path(directory) / ".writer.lock").exists())
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory)
            store.save(self.response())
            used = sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())
            store.max_bytes = used
            with self.assertRaises(WebBudgetError):
                store.save(self.response())                                               

    def test_corruption_and_path_traversal_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory)
            document = store.save(self.response())
            snapshot_id = document["metadata"]["snapshot_id"]
            blob = Path(directory) / "blobs" / f"{document['metadata']['sha256']}.bin"
            blob.write_bytes(b"Changed evidence")
            with self.assertRaises(WebPolicyError):
                store.load(snapshot_id)
            with self.assertRaises(ValueError):
                store.load("../../outside")
            metadata = Path(directory) / "snapshots" / f"{snapshot_id}.json"
            metadata.write_text("{}", encoding="utf-8")
            with self.assertRaises(WebPolicyError):
                store.manifest()

    def test_capture_fetches_only_the_requested_page_and_owns_no_crawler(self):
        with tempfile.TemporaryDirectory() as directory:
            client = mock.Mock()
            client.fetch.return_value = self.response()
            document = SnapshotStore(directory).capture("https://example.com/original", client)
            client.fetch.assert_called_once_with("https://example.com/original")
            self.assertEqual(document["metadata"]["track"], "frozen_web")


if __name__ == "__main__":
    unittest.main()
