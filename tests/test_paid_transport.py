                                                                                   

from contextlib import closing
import io
import json
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from magra_rag import configuration as cfg
from magra_rag.generation import API_ROOT, GenerationError, OpenRouterM3, PaidHTTPError, RateLimitError, SpendLedger
from magra_rag.resources import ResourceBudget
from magra_benchmark import judge as judge_module


FAKE_KEY = "invented-paid-transport-key"
INITIAL_CHARGE = 0.012345
EVIDENCE = [{"doc_id": "invented-document", "sentence_id": 0,
             "text": "The invented sample contains seven items."}]
QUERY = {"id": "invented-query", "question": "How many items are in the sample?"}
PREDICTION = {"answer": "The sample contains seven items.", "abstained": False,
              "citations": [{"doc_id": "invented-document", "sentence_id": 0}]}


class _Response(io.BytesIO):
    def __init__(self, raw):
        super().__init__(raw)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


def _json_response(value):
    return _Response(json.dumps(value, allow_nan=False).encode("utf-8"))


class PaidTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="paid-transport-fake-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger_path = self.root / "paid" / "runs" / "spend.sqlite"
        for replacement in (
            patch.object(cfg, "PAID_LEDGER_PATH", self.ledger_path),
            patch.object(cfg, "_dotenv", side_effect=AssertionError("No environment file access")),
            patch("urllib.request.urlopen", side_effect=AssertionError("No live urlopen")),
            patch.object(socket, "create_connection", side_effect=AssertionError("No live connection")),
            patch.object(socket.socket, "connect", side_effect=AssertionError("No live socket")),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        initial = SpendLedger(
            self.ledger_path, 20,
            budget=ResourceBudget(self.root / "paid", disk_bytes=256_000_000,
                                  reserve_bytes=4_000_000))
        reservation = initial.reserve(0.05)
        initial.settle(reservation, INITIAL_CHARGE)
        self.client = self.new_client()
        self.initial_rows = self.rows()

    def new_client(self):
        return OpenRouterM3(data_root=self.root / "owned-run", profile="paid",
                            api_key=FAKE_KEY, env_file=None)

    def rows(self):
        with closing(sqlite3.connect(self.ledger_path)) as database:
            return database.execute(
                "SELECT id,reserved,charged,state,created_at FROM spend ORDER BY rowid"
            ).fetchall()

    def catalog(self):
        return {"data": [{"id": cfg.PAID_MODEL,
                          "context_length": 1048576,
                          "pricing": {"prompt": "0.00000015", "completion": "0.00000060"},
                          "supported_parameters": ["response_format", "reasoning"],
                          "reasoning": {"supported_efforts": ["low"]}}]}

    def routed_opener(self, post):
        pass                                                                           
        calls = []

        def open_request(request, *, timeout):
            calls.append((request.get_method(), request.full_url))
            self.assertEqual(timeout, self.client.timeout)
            if calls[-1] == ("GET", API_ROOT + "/models"):
                return _json_response(self.catalog())
            if calls[-1] == ("GET", API_ROOT + "/key"):
                return _json_response({"data": {"limit": 20, "usage": 0.1,
                                                "limit_remaining": 19.9,
                                                "is_free_tier": False,
                                                "is_management_key": False}})
            self.assertEqual(calls[-1], ("POST", API_ROOT + "/chat/completions"))
            self.assertEqual(request.get_header("Authorization"), "Bearer " + FAKE_KEY)
            body = json.loads(request.data)
            self.assertEqual(body["model"], cfg.PAID_MODEL)
            self.assertFalse(body["provider"]["allow_fallbacks"])
            self.assertEqual(body["provider"]["max_price"], {"prompt": 0.15, "completion": 0.60})
            return post(request)

        return Mock(open=Mock(side_effect=open_request)), calls

    def test_exact_allowed_endpoints_install_redirect_refusal(self):
        for method, suffix in (("GET", "/models"), ("GET", "/key"),
                               ("POST", "/chat/completions")):
            with self.subTest(method=method, suffix=suffix):
                request = urllib.request.Request(
                    API_ROOT + suffix, method=method,
                    data=b"{}" if method == "POST" else None)
                response = _json_response({"received": suffix})
                opener = Mock(open=Mock(return_value=response))
                with patch("urllib.request.build_opener", return_value=opener) as build:
                    self.assertEqual(self.client._paid_json_request(request, 7), {"received": suffix})
                build.assert_called_once()
                handler, = build.call_args.args
                self.assertIsInstance(handler, urllib.request.HTTPRedirectHandler)
                self.assertIsNone(handler.redirect_request(
                    request, None, 302, "Found", {"Location": "https://redirect.invalid/"},
                    "https://redirect.invalid/"))
                opener.open.assert_called_once_with(request, timeout=7)
                self.assertEqual(response.read_sizes, [8_000_001])
                self.assertTrue(response.closed)

    def test_other_origins_paths_and_methods_are_rejected_before_open(self):
        cases = [("GET", "http://openrouter.ai/api/v1/models"),
                 ("GET", "https://other.invalid/api/v1/models"),
                 ("GET", API_ROOT + "/models?extra=1"),
                 ("GET", API_ROOT + "/models#fragment"),
                 ("GET", API_ROOT + "/models/"),
                 ("POST", API_ROOT + "/models"),
                 ("GET", API_ROOT + "/chat/completions"),
                 ("DELETE", API_ROOT + "/key")]
        with patch("urllib.request.build_opener") as build:
            for method, url in cases:
                with self.subTest(method=method, url=url), self.assertRaises(GenerationError):
                    self.client._paid_json_request(urllib.request.Request(url, method=method), 7)
            build.assert_not_called()
        self.assertEqual(self.rows(), self.initial_rows)

    def test_redirect_response_never_retries_or_forwards_credentials(self):
        request = urllib.request.Request(
            API_ROOT + "/chat/completions", data=b"{}", method="POST",
            headers={"Authorization": "Bearer " + FAKE_KEY})
        error = urllib.error.HTTPError(request.full_url, 302, FAKE_KEY,
                                      {"Location": "https://redirect.invalid/"}, None)
        opener = Mock(open=Mock(side_effect=error))
        with patch("urllib.request.build_opener", return_value=opener) as build:
            with self.assertRaises(GenerationError) as caught:
                self.client._paid_json_request(request, 7)
        self.assertNotIn(FAKE_KEY, str(caught.exception))
        build.assert_called_once()
        opener.open.assert_called_once_with(request, timeout=7)

    def test_duplicate_nonfinite_nonobject_and_malformed_json_are_rejected(self):
        cases = [b'{"cost":1,"cost":2}', b'{"usage":{"cost":1,"cost":2}}',
                 b'{"cost":NaN}', b'{"cost":Infinity}', b'{"cost":-Infinity}',
                 b'[]', b'null', b'{', b'\xff']
        request = urllib.request.Request(API_ROOT + "/models")
        for raw in cases:
            with self.subTest(raw=raw):
                response = _Response(raw)
                opener = Mock(open=Mock(return_value=response))
                with patch("urllib.request.build_opener", return_value=opener):
                    with self.assertRaises(GenerationError):
                        self.client._paid_json_request(request, 7)
                opener.open.assert_called_once_with(request, timeout=7)
                self.assertEqual(response.read_sizes, [8_000_001])
                self.assertTrue(response.closed)

    def test_response_byte_cap_accepts_exact_limit_and_rejects_one_more(self):
        request = urllib.request.Request(API_ROOT + "/models")
        for extra in (0, 1):
            with self.subTest(bytes=8_000_000 + extra):
                response = _Response(b"{}" + b" " * (8_000_000 - 2 + extra))
                opener = Mock(open=Mock(return_value=response))
                with patch("urllib.request.build_opener", return_value=opener):
                    if extra:
                        with self.assertRaises(GenerationError):
                            self.client._paid_json_request(request, 7)
                    else:
                        self.assertEqual(self.client._paid_json_request(request, 7), {})
                self.assertEqual(response.read_sizes, [8_000_001])
                self.assertTrue(response.closed)
                opener.open.assert_called_once_with(request, timeout=7)

    def test_generation_lost_response_keeps_reservation_and_prevents_second_call(self):
        def lose_response(request):
            pending = self.rows()[-1]
            self.assertGreater(pending[1], 0)
            self.assertEqual(pending[2:4], (None, "reserved"))
            raise urllib.error.URLError("Invented lost response " + FAKE_KEY)

        opener, calls = self.routed_opener(lose_response)
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(GenerationError) as caught:
                self.client.generate(QUERY["question"], EVIDENCE)
            self.assertNotIn(FAKE_KEY, str(caught.exception))
            self.assertEqual(calls, [("GET", API_ROOT + "/models"),
                                     ("GET", API_ROOT + "/key"),
                                     ("POST", API_ROOT + "/chat/completions")])
            failed_rows = self.rows()
            self.assertEqual(failed_rows[:-1], self.initial_rows)
            self.assertEqual(failed_rows[-1][2:4], (None, "reserved"))
            summary = self.client.ledger.summary()
            self.assertEqual(summary["unconfirmed_requests"], 1)
            self.assertAlmostEqual(summary["accounted_usd"], INITIAL_CHARGE + failed_rows[-1][1])
            with self.assertRaises(GenerationError):
                self.client.generate("A second invented question?", EVIDENCE)
            with self.assertRaises(GenerationError):
                self.new_client()
            self.assertEqual(self.rows(), failed_rows)
            self.assertEqual(opener.open.call_count, 3)

    def test_paid402_status_is_preserved_without_body_url_or_secret(self):
        request = urllib.request.Request(API_ROOT + "/chat/completions", data=b"{}", method="POST")
        body = _Response(("unread-private-body-" + FAKE_KEY).encode())
        error = urllib.error.HTTPError(request.full_url, 402, FAKE_KEY, {"arbitrary": FAKE_KEY}, body)
        opener = Mock(open=Mock(side_effect=error))
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(PaidHTTPError) as caught:
                self.client._paid_json_request(request, 7)
        self.assertEqual(caught.exception.status_code, 402)
        self.assertIn("HTTP 402", str(caught.exception))
        self.assertNotIn(FAKE_KEY, str(caught.exception))
        self.assertNotIn(request.full_url, str(caught.exception))
        self.assertEqual(body.read_sizes, [])
        self.assertTrue(body.closed)
        opener.open.assert_called_once_with(request, timeout=7)

    def test_paid429_retains_rate_limit_type_and_safe_headers(self):
        request = urllib.request.Request(API_ROOT + "/models")
        body = _Response(FAKE_KEY.encode())
        error = urllib.error.HTTPError(request.full_url, 429, FAKE_KEY,
                                      {"Retry-After": "60", "X-RateLimit-Remaining": "0"}, body)
        opener = Mock(open=Mock(side_effect=error))
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(RateLimitError) as caught:
                self.client._paid_json_request(request, 7)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.retry_after_seconds, 60)
        self.assertEqual(caught.exception.rate_limit, {"x-ratelimit-remaining": 0})
        self.assertEqual(body.read_sizes, [])
        self.assertTrue(body.closed)
        opener.open.assert_called_once_with(request, timeout=7)

    def test_generation402_retains_reservation_and_global_unknown_halt(self):
        def exhausted(request):
            raise urllib.error.HTTPError(request.full_url, 402, FAKE_KEY, None, io.BytesIO(FAKE_KEY.encode()))
        opener, calls = self.routed_opener(exhausted)
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(PaidHTTPError) as caught:
                self.client.generate(QUERY["question"], EVIDENCE)
            self.assertEqual(caught.exception.status_code, 402)
            failed_rows = self.rows()
            self.assertEqual(failed_rows[:-1], self.initial_rows)
            self.assertEqual(failed_rows[-1][2:4], (None, "reserved"))
            self.assertEqual(self.client.ledger.summary()["unconfirmed_requests"], 1)
            with self.assertRaises(GenerationError):
                self.client.generate("Another fictional question?", EVIDENCE)
            with self.assertRaises(GenerationError):
                self.new_client()
            self.assertEqual(self.rows(), failed_rows)
            self.assertEqual(len(calls), 3)

    def test_judge_known_charge_is_committed_before_schema_validation_fails(self):
        charge = 0.000123
        opener, calls = self.routed_opener(lambda request: _json_response({
            "id": "invented-paid-judge-response", "model": cfg.PAID_MODEL,
            "usage": {"cost": charge, "total_tokens": 12},
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}))
        original_judgment = judge_module._judgment

        def observe_committed_charge(content, limit):
            rows = self.rows()
            self.assertEqual(rows[:-1], self.initial_rows)
            self.assertEqual(rows[-1][2:4], (charge, "settled"))
            return original_judgment(content, limit)

        judge = judge_module.M3GroundednessJudge(self.client)
        with patch("urllib.request.build_opener", return_value=opener), patch.object(
                judge_module, "_judgment", side_effect=observe_committed_charge) as validate:
            with self.assertRaisesRegex(judge_module.JudgeError, "structured judge response"):
                judge.judge(QUERY, PREDICTION, EVIDENCE)
        validate.assert_called_once()
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1], ("POST", API_ROOT + "/chat/completions"))
        self.assertEqual(self.client.ledger.summary()["unconfirmed_requests"], 0)
        self.assertAlmostEqual(self.client.ledger.summary()["accounted_usd"], INITIAL_CHARGE + charge)
        reopened = self.new_client()
        self.assertEqual(reopened.ledger.path, self.ledger_path.resolve())
        self.assertAlmostEqual(reopened.ledger.summary()["accounted_usd"], INITIAL_CHARGE + charge)


if __name__ == "__main__":
    unittest.main()
