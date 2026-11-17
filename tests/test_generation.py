                                                                                    

from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import traceback
import unittest
from unittest.mock import patch
import urllib.error

from magra_benchmark.evaluation import compare_runs, evaluate_run
from magra_rag.generation import API_ROOT, GenerationError, OpenRouterM3, SpendLedger
from magra_rag.resources import ResourceBudget


MODEL = "minimax/minimax-m3"                                                          
FAKE_KEY = "sk-or-v1-OFFLINE-TEST-SECRET-NEVER-A-REAL-CREDENTIAL"
EVIDENCE = [{"doc_id": "owned:aurora", "sentence_id": 0, "title": "Aurora",
             "text": "Aurora was founded in 1998."}]
QUESTION = "When was Aurora founded?"


def catalog():
    return {"data": [{"id": MODEL, "supported_parameters": ["response_format"], "pricing": {
        "prompt": "0.000001", "completion": "0.000002", "request": "0",
        "image": "0", "web_search": "0", "internal_reasoning": "0",
    }}]}


def answer():
    return {"answer": "1998", "citations": [{"doc_id": "owned:aurora", "sentence_id": 0}],
            "confidence": 0.8, "abstained": False, "reason": "Supported by the supplied sentence."}


def completion(structured=None, **updates):
    payload = {"id": "offline-completion", "model": MODEL,
               "usage": {"total_tokens": 25, "cost": 0.0001},
               "choices": [{"finish_reason": "stop", "message": {
                   "content": json.dumps(answer() if structured is None else structured)}}]}
    payload.update(updates)
    return payload


class OfflineResponse:
    def __init__(self, payload=None, *, raw=None):
        self.raw = raw if raw is not None else json.dumps(payload).encode()
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        self.read_limits.append(limit)
        return self.raw[:limit]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.budget = ResourceBudget(self.root)
                                                                            
                                                                                
        self.transport = patch("magra_rag.generation.urllib.request.urlopen",
                               side_effect=AssertionError("Offline test attempted an unmocked request"))
        self.urlopen = self.transport.start()
        self.addCleanup(self.transport.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def client(self, **overrides):
        kwargs = {"data_root": self.root, "max_total_usd": 1.0, "max_tokens": 16,
                  "timeout": 5, "prompt_usd_per_million": 2,
                  "completion_usd_per_million": 4, "api_key": FAKE_KEY, "budget": self.budget,
                  "model_id": MODEL, "env_file": None}
        kwargs.update(overrides)
        return OpenRouterM3(**kwargs)

    def responses(self, *payloads):
        self.urlopen.side_effect = [OfflineResponse(payload) for payload in payloads]

    def request_counts(self):
        requests = [call.args[0] for call in self.urlopen.call_args_list]
        return sum(request.get_method() == "GET" for request in requests), \
            sum(request.get_method() == "POST" for request in requests)

    def test_preflight_uses_public_catalog_without_authorization_and_exact_model(self):
        self.responses(catalog())
        result = self.client().preflight()
        request = self.urlopen.call_args.args[0]
        self.assertEqual(request.full_url, API_ROOT + "/models")
        self.assertEqual(request.get_method(), "GET")
        self.assertNotIn("authorization", {name.lower() for name, _ in request.header_items()})
        self.assertEqual(result["id"], MODEL)
        self.assertEqual(len(result["catalog_sha256"]), 64)
        self.assertNotIn(FAKE_KEY, json.dumps(result))

    def test_success_is_one_paid_request_with_price_bounds_and_no_returned_secret(self):
        self.responses(catalog(), completion())
        client = self.client()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = client.generate(QUESTION, EVIDENCE, as_of="2026-01-01")
        self.assertEqual(self.request_counts(), (1, 1))
        request = self.urlopen.call_args.args[0]
        self.assertEqual(request.full_url, API_ROOT + "/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + FAKE_KEY)
        body = json.loads(request.data)
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["max_tokens"], 16)
        self.assertFalse(body["stream"])
        self.assertEqual(body["provider"]["max_price"], {"prompt": 2, "completion": 4})
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertNotIn(FAKE_KEY, request.data.decode())
        evidence_body = json.loads(body["messages"][1]["content"])
        self.assertEqual(evidence_body["evidence"], EVIDENCE)
        self.assertEqual(evidence_body["as_of"], "2026-01-01")
        self.assertEqual(result["answer"], "1998")
        self.assertEqual(result["cost_usd"], 0.0001)
        self.assertGreater(result["reserved_usd"], result["cost_usd"])
        self.assertEqual(client.ledger.summary()["unconfirmed_requests"], 0)
        self.assertNotIn(FAKE_KEY, json.dumps(result) + stdout.getvalue() + stderr.getvalue())
        self.assertNotIn(FAKE_KEY.encode(), client.ledger.path.read_bytes())

    def test_missing_exact_model_never_falls_back_or_sends_paid_request(self):
        unavailable = catalog()
        unavailable["data"][0]["id"] = MODEL + "-preview"
        self.responses(unavailable)
        client = self.client()
        with self.assertRaisesRegex(GenerationError, "absent"):
            client.generate(QUESTION, EVIDENCE)
        self.assertEqual(self.request_counts(), (1, 0))
        self.assertEqual(client.ledger.summary()["accounted_usd"], 0)

    def test_catalog_price_above_either_ceiling_is_rejected_before_reservation(self):
        for field, price in (("prompt", "0.000003"), ("completion", "0.000005")):
            with self.subTest(field=field):
                self.urlopen.reset_mock()
                expensive = catalog()
                expensive["data"][0]["pricing"][field] = price
                self.responses(expensive)
                client = self.client()
                with self.assertRaisesRegex(GenerationError, "ceiling"):
                    client.generate(QUESTION, EVIDENCE)
                self.assertEqual(self.request_counts(), (1, 0))
                self.assertEqual(client.ledger.summary()["accounted_usd"], 0)

    def test_unaccounted_or_invalid_catalog_pricing_fails_closed(self):
        for field, value in (("request", "0.01"), ("image", "NaN"),
                             ("web_search", "-1"), ("internal_reasoning", FAKE_KEY)):
            with self.subTest(field=field, category="invalid-price"):
                self.urlopen.reset_mock()
                invalid = catalog()
                invalid["data"][0]["pricing"][field] = value
                self.responses(invalid)
                client = self.client()
                with self.assertRaises(GenerationError) as caught:
                    client.generate(QUESTION, EVIDENCE)
                self.assertNotIn(FAKE_KEY, "".join(traceback.format_exception(caught.exception)))
                self.assertEqual(self.request_counts(), (1, 0))
                self.assertEqual(client.ledger.summary()["accounted_usd"], 0)

    def test_wrong_response_model_is_rejected_without_free_billing_assumption(self):
        self.responses(catalog(), completion(model="another-provider/another-model"))
        client = self.client()
        with self.assertRaisesRegex(GenerationError, "model"):
            client.generate(QUESTION, EVIDENCE)
        self.assertEqual(self.request_counts(), (1, 1))
        self.assertAlmostEqual(client.ledger.summary()["accounted_usd"], 0.0001)

    def test_bad_citations_and_boolean_fields_do_not_trigger_paid_retries(self):
        variants = [
            {"citations": [{"doc_id": "outside", "sentence_id": 0}]},
            {"citations": [{"doc_id": "owned:aurora", "sentence_id": True}]},
            {"citations": [{"doc_id": "owned:aurora", "sentence_id": 0.0}]},
            {"citations": [{"doc_id": False, "sentence_id": 0}]},
            {"citations": ["owned:aurora"]}, {"citations": "owned:aurora"},
            {"confidence": True}, {"confidence": float("nan")}, {"abstained": 0},
            {"abstained": True}, {"citations": []},
        ]
        for variation in variants:
            with self.subTest(variation=variation):
                self.urlopen.reset_mock()
                self.responses(catalog(), completion({**answer(), **variation}))
                with self.assertRaises(GenerationError):
                    self.client().generate(QUESTION, EVIDENCE)
                self.assertEqual(self.request_counts(), (1, 1))

    def test_malformed_or_truncated_structured_output_is_not_retried(self):
        malformed = completion()
        malformed["choices"][0]["message"]["content"] = "This is not JSON"
        truncated = completion()
        truncated["choices"][0]["finish_reason"] = "length"
        for payload in (malformed, truncated, completion(structured=[]), completion(choices=[])):
            with self.subTest(payload_type="invalid-output"):
                self.urlopen.reset_mock()
                self.responses(catalog(), payload)
                with self.assertRaises(GenerationError):
                    self.client().generate(QUESTION, EVIDENCE)
                self.assertEqual(self.request_counts(), (1, 1))

    def test_no_evidence_abstains_with_zero_cost_and_no_network(self):
        client = self.client()
        result = client.generate(QUESTION, [])
        self.urlopen.assert_not_called()
        self.assertTrue(result["abstained"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["cost_usd"], 0)
        self.assertEqual(result["tokens"], 0)
        self.assertEqual(client.ledger.summary()["accounted_usd"], 0)

    def test_unknown_reported_cost_retains_reservation_across_client_restart(self):
        unknown_cost = completion(usage={"total_tokens": 25})
        self.responses(catalog(), unknown_cost)
        client = self.client()
        result = client.generate(QUESTION, EVIDENCE)
        summary = client.ledger.summary()
        self.assertIsNone(result["cost_usd"])
        self.assertAlmostEqual(summary["accounted_usd"], result["reserved_usd"])
        self.assertEqual(summary["unconfirmed_requests"], 1)
        restarted = self.client()
        self.assertEqual(restarted.ledger.summary(), summary)
        with closing(sqlite3.connect(client.ledger.path)) as database:
            self.assertEqual(database.execute("SELECT state FROM spend").fetchone()[0], "unconfirmed")
        bounded = SpendLedger(client.ledger.path, summary["accounted_usd"] * 1.5, budget=self.budget)
        with self.assertRaises(GenerationError):
            bounded.reserve(summary["accounted_usd"])

    def test_transport_failure_has_no_paid_retry_or_secret_in_traceback_and_keeps_unknown_spend(self):
        client = self.client()
        self.urlopen.side_effect = [OfflineResponse(catalog()), urllib.error.URLError(FAKE_KEY)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(GenerationError) as caught:
                client.generate(QUESTION, EVIDENCE)
        diagnostic = "".join(traceback.format_exception(caught.exception)) + stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(FAKE_KEY, diagnostic)
        self.assertEqual(self.request_counts(), (1, 1))
        summary = client.ledger.summary()
        self.assertGreater(summary["accounted_usd"], 0)
        self.assertEqual(summary["unconfirmed_requests"], 1)
        self.assertEqual(self.client().ledger.summary(), summary)

    def test_malformed_reported_cost_is_sanitized_and_keeps_reservation(self):
        for value in (FAKE_KEY, True, -1, float("nan")):
            with self.subTest(category="invalid-usage-cost"):
                self.urlopen.reset_mock()
                self.responses(catalog(), completion(usage={"total_tokens": 25, "cost": value}))
                client = self.client()
                with self.assertRaises(GenerationError) as caught:
                    client.generate(QUESTION, EVIDENCE)
                self.assertNotIn(FAKE_KEY, "".join(traceback.format_exception(caught.exception)))
                self.assertEqual(self.request_counts(), (1, 1))
                self.assertGreater(client.ledger.summary()["unconfirmed_requests"], 0)

    def test_malformed_top_level_json_and_oversized_api_response_are_rejected(self):
        for raw in (b"[]", b"null", b"x" * 8_000_001):
            with self.subTest(category="invalid-response-envelope"):
                response = OfflineResponse(raw=raw)
                self.urlopen.side_effect = [response]
                with self.assertRaises(GenerationError):
                    self.client().preflight()
                self.assertEqual(response.read_limits, [8_000_001])

    def test_spend_limit_is_checked_before_post(self):
        self.responses(catalog())
        client = self.client(max_total_usd=0.0000001)
        with self.assertRaisesRegex(GenerationError, "spending limit"):
            client.generate(QUESTION, EVIDENCE)
        self.assertEqual(self.request_counts(), (1, 0))
        self.assertEqual(client.ledger.summary()["accounted_usd"], 0)


class ComparisonCompletenessTests(unittest.TestCase):
    def test_missing_baseline_context_outputs_cannot_support_acceptance(self):
        def labels(split):
            return [{"id": f"{split}-{setting}-{i}", "question": "Fixture?", "split": split,
                     "source": "hotpotqa", "domain": "encyclopedia", "setting": setting,
                     "relevance": {"document": 1}, "answer": "expected"}
                    for setting in ("full_corpus", "provided_context") for i in range(2)]

        def predictions(rows):
            return [{"query_id": row["id"], "ranked_doc_ids": ["document"], "answer": "expected",
                     "cost_usd": 0.01, "latency_ms": 1, "tokens": 10} for row in rows]

        dev, evaluation = labels("dev"), labels("eval")
        baseline = evaluate_run(dev, predictions(dev))
        incomplete = predictions([row for row in evaluation if row["setting"] == "full_corpus"])
        comparison = compare_runs(evaluation, {"baseline": incomplete, "candidate": predictions(evaluation)},
                                  baseline_names=["baseline"], candidate_name="candidate",
                                  development_reports={"baseline": baseline}, bootstrap_samples=20)
        provided = [group for group in comparison["groups"].values()
                    if group["stratum"]["setting"] == "provided_context"]
        self.assertEqual(len(provided), 1)
        self.assertFalse(provided[0]["complete_predictions"])
        self.assertFalse(comparison["generality_claim_supported"])
        self.assertEqual(comparison["acceptance_status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
