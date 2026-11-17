                                                                                     

from contextlib import closing
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import traceback
import unittest
from unittest.mock import patch
import urllib.error

from magra_rag.configuration import DEFAULT_MODEL, load_openrouter_settings
from magra_rag.generation import (GenerationError, IncompleteResponseError, OpenRouterM3,
                                  RateLimitError, REASONING_EFFORTS, SpendLedger)
from magra_rag.resources import ResourceBudget


SECRET = "offline-only-credential-no-real-key"
EVIDENCE = [{"doc_id": "d", "sentence_id": 0, "text": "Aurora was founded in 1998."}]


class Response:
    def __init__(self, value):
        self.raw = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.raw[:limit]


def catalog(model=DEFAULT_MODEL, **overrides):
    model_data = {"id": model, "pricing": {"prompt": "0", "completion": "0"},
                  "supported_parameters": ["response_format"]}
    model_data.update(overrides)
    return {"data": [model_data]}


def completion(model=DEFAULT_MODEL):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"answer": "1998", "citations": [{"doc_id": "d", "sentence_id": 0}],
                               "abstained": False, "reason": "Supplied evidence", "confidence": 0.8})}}],
        "usage": {"cost": 0, "total_tokens": 20}}


class ModelConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.env = self.root / ".env"
        self.addCleanup(self.temporary.cleanup)

    def settings(self, **kwargs):
        options = {"env_file": self.env, "environ": {}}
        options.update(kwargs)
        return load_openrouter_settings(**options)

    def test_existing_alias_loads_without_evaluation_or_repr_exposure(self):
        self.env.write_text('ai-model-key="' + SECRET + '" # local credential\n'
                            'IGNORED=$(never_execute)\nOPENROUTER_MODEL=example/switch:free\n')
        config = self.settings()
        self.assertEqual(config.api_key, SECRET)
        self.assertEqual(config.model_id, "example/switch:free")
        self.assertTrue(config.key_configured)
        self.assertNotIn(SECRET, repr(config))

    def test_precedence_explicit_then_actual_environment_then_dotenv(self):
        self.env.write_text('OPENROUTER_API_KEY=file-credential\nOPENROUTER_MODEL=file/model:free\n')
        actual = {"ai-model-key": "actual-credential", "OPENROUTER_MODEL": "actual/model:free"}
        self.assertEqual(self.settings(environ=actual).api_key, "actual-credential")
        self.assertEqual(self.settings(environ=actual).model_id, "actual/model:free")
        explicit = self.settings(environ=actual, api_key="explicit-credential", model_id="explicit/model:free")
        self.assertEqual(explicit.api_key, "explicit-credential")
        self.assertEqual(explicit.model_id, "explicit/model:free")
        self.assertEqual(actual, {"ai-model-key": "actual-credential", "OPENROUTER_MODEL": "actual/model:free"})

    def test_empty_actual_credential_does_not_fall_back_to_file(self):
        self.env.write_text('ai-model-key=' + SECRET)
        self.assertIsNone(self.settings(environ={"OPENROUTER_API_KEY": ""}).api_key)

    def test_missing_file_has_safe_default(self):
        config = self.settings()
        self.assertEqual(config.model_id, "minimax/minimax-m3.1-flash:free")
        self.assertFalse(config.key_configured)

    def test_literal_values_are_not_interpolated(self):
        self.env.write_text("export ai-model-key='${NEVER_EVALUATE}'\nOPENROUTER_MODEL=valid/model:free # comment\n")
        self.assertEqual(self.settings().api_key, "${NEVER_EVALUATE}")
        self.assertEqual(self.settings().model_id, "valid/model:free")

    def test_malformed_or_oversized_file_fails_without_secret_traceback(self):
        for content in ('ai-model-key="' + SECRET, 'ai-model-key="' + SECRET + '" trailing',
                        'ai-model-key=' + SECRET + 'x' * 65_536):
            self.env.write_text(content)
            with self.subTest(case="invalid-file"), self.assertRaises(ValueError) as caught:
                self.settings()
            self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_invalid_model_and_credential_rejected_without_echo(self):
        for model in ("model id", "bad\nmodel", "", "a" * 201):
            with self.subTest(model="invalid"), self.assertRaises(ValueError):
                self.settings(model_id=model)
        with self.assertRaises(ValueError):
            self.settings(api_key=SECRET + "\n")


class FreeTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.budget = ResourceBudget(self.root)
        self.addCleanup(self.temporary.cleanup)
        self.transport = patch("magra_rag.generation.urllib.request.urlopen",
                               side_effect=AssertionError("Unplanned offline request"))
        self.urlopen = self.transport.start()
        self.addCleanup(self.transport.stop)

    def client(self, **kwargs):
        options = {"data_root": self.root, "api_key": SECRET, "env_file": None,
                   "budget": self.budget}
        options.update(kwargs)
        return OpenRouterM3(**options)

    def test_default_free_model_has_zero_price_routing_no_monetary_cap(self):
        self.urlopen.side_effect = [Response(catalog()), Response(completion())]
        client = self.client(prompt_usd_per_million=10, completion_usd_per_million=10)
        result = client.generate("When was Aurora founded?", EVIDENCE)
        body = json.loads(self.urlopen.call_args.args[0].data)
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertEqual(body["provider"]["max_price"], {"prompt": 0, "completion": 0})
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(result["cost_usd"], 0)
        self.assertIsNone(client.ledger.summary()["limit_usd"])

    def test_free_model_catalog_positive_price_rejected_even_with_paid_ceiling(self):
        self.urlopen.side_effect = [Response(catalog(pricing={"prompt": "0.000001", "completion": "0"}))]
        with self.assertRaisesRegex(GenerationError, "Free model"):
            self.client(prompt_usd_per_million=10).generate("When?", EVIDENCE)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_exact_model_switch_is_persisted_and_no_substitution_allowed(self):
        selected = "other-provider/other-model:free"
        self.urlopen.side_effect = [Response(catalog(selected)), Response(completion(selected))]
        result = self.client(model_id=selected).generate("When?", EVIDENCE)
        self.assertEqual(result["model"], selected)
        self.assertEqual(result["catalog"]["id"], selected)
        self.assertEqual(json.loads(self.urlopen.call_args.args[0].data)["model"], selected)
        self.urlopen.side_effect = [Response(catalog(selected)), Response(completion(DEFAULT_MODEL))]
        with self.assertRaisesRegex(GenerationError, "model"):
            self.client(model_id=selected).generate("When?", EVIDENCE)

    def test_schema_only_sent_when_supported(self):
        schema = {"type": "object"}
        plain = self.client().response_format({"supported_parameters": ["response_format"]}, schema=schema)
        self.assertEqual(plain, {"type": "json_object"})
        structured = self.client().response_format({"supported_parameters": ["structured_outputs"]}, schema=schema)
        self.assertEqual(structured["type"], "json_schema")
        self.assertEqual(structured["json_schema"]["schema"], schema)

    def test_model_without_json_support_stops_before_reservation(self):
        self.urlopen.side_effect = [Response(catalog(supported_parameters=[]))]
        client = self.client()
        with self.assertRaisesRegex(GenerationError, "JSON"):
            client.generate("When?", EVIDENCE)
        self.assertEqual(self.urlopen.call_count, 1)
        with closing(sqlite3.connect(client.ledger.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM spend").fetchone()[0], 0)

    def test_account_limits_returns_allowed_counters_and_drops_key_label(self):
        self.urlopen.side_effect = [Response({"data": {"label": SECRET, "hash": SECRET,
            "is_free_tier": True, "limit": None, "usage_daily": 0,
            "free_model_daily_requests": {"used": 17, "limit": 1000, "remaining": 983, "other": SECRET},
            "rate_limit": {"requests": 20, "interval": "1m", "arbitrary": SECRET}}})]
        result = self.client().account_limits()
        self.assertEqual(result["data"]["free_model_daily_requests"], {"used": 17, "limit": 1000, "remaining": 983})
        self.assertNotIn("rate_limit", result["data"])
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertEqual(self.urlopen.call_args.args[0].full_url.rsplit("/", 1)[-1], "key")

    def test_429_retains_safe_retry_metadata_without_retrying_or_reading_body(self):
        error = urllib.error.HTTPError("https://openrouter.ai", 429, SECRET,
            {"Retry-After": "60", "X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0",
             "X-RateLimit-Reset": SECRET}, io.BytesIO(SECRET.encode()))
        self.urlopen.side_effect = [Response(catalog()), error]
        client = self.client()
        with self.assertRaises(RateLimitError) as caught:
            client.generate("When?", EVIDENCE)
        self.assertEqual(caught.exception.retry_after_seconds, 60)
        self.assertEqual(caught.exception.rate_limit, {"x-ratelimit-limit": 20, "x-ratelimit-remaining": 0})
        self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))
        self.assertEqual(self.urlopen.call_count, 2)
        self.assertEqual(client.ledger.summary()["unconfirmed_requests"], 1)

    def test_missing_or_malformed_retry_after_is_unavailable(self):
        for value in (SECRET, "NaN", "-3"):
            self.urlopen.side_effect = urllib.error.HTTPError("https://openrouter.ai", 429, SECRET,
                                                             {"Retry-After": value}, None)
            with self.subTest(case="invalid-header"), self.assertRaises(RateLimitError) as caught:
                self.client().account_limits()
            self.assertIsNone(caught.exception.retry_after_seconds)

    def test_unlimited_ledger_still_accounts_and_zero_limit_allows_only_free(self):
        ledger = SpendLedger(self.root / "no-cap.sqlite", None, budget=self.budget)
        ledger.reserve(100)
        self.assertEqual(ledger.summary()["accounted_usd"], 100)
        zero = SpendLedger(self.root / "zero.sqlite", 0, budget=self.budget)
        zero.reserve(0)
        with self.assertRaises(GenerationError):
            zero.reserve(0.000001)

    def test_invalid_price_ceiling_or_limit_still_rejected(self):
        for value in (True, -1, float("nan"), float("inf"), "bad"):
            with self.subTest(field="limit"), self.assertRaises(ValueError):
                self.client(max_total_usd=value)
            with self.subTest(field="price"), self.assertRaises(ValueError):
                self.client(prompt_usd_per_million=value)

    def test_omitted_effort_retains_provider_default_without_reasoning_body(self):
        self.urlopen.side_effect = [Response(catalog()), Response(completion())]
        client = self.client()
        client.generate("When?", EVIDENCE)
        self.assertIsNone(client.reasoning_effort)
        self.assertEqual(client.reasoning_options({}), {})
        self.assertNotIn("reasoning", json.loads(self.urlopen.call_args.args[0].data))

    def test_all_gateway_efforts_validate_and_invalid_values_are_sanitized(self):
        for effort in REASONING_EFFORTS:
            with self.subTest(effort=effort):
                self.assertEqual(self.client(reasoning_effort=effort).reasoning_effort, effort)
        for effort in (True, [], "", "LOW", SECRET):
            with self.subTest(case="invalid"), self.assertRaises(ValueError) as caught:
                self.client(reasoning_effort=effort)
            self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_low_effort_payload_and_sanitized_model_reasoning_capabilities(self):
        model = catalog(supported_parameters=["response_format", "reasoning"], reasoning={
            "supported_efforts": ["high", "medium", "low"], "mandatory": True,
            "default_effort": "medium", "default_enabled": True, "supports_max_tokens": False,
            "raw_provider_note": SECRET})
        self.urlopen.side_effect = [Response(model), Response(completion())]
        client = self.client(reasoning_effort="low", prompt_usd_per_million=50)
        result = client.generate("When?", EVIDENCE)
        body = json.loads(self.urlopen.call_args.args[0].data)
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["provider"]["max_price"], {"prompt": 0, "completion": 0})
        self.assertEqual(result["catalog"]["reasoning"], {
            "supported_efforts": ["high", "medium", "low"], "mandatory": True,
            "default_effort": "medium", "default_enabled": True, "supports_max_tokens": False})
        self.assertNotIn(SECRET, json.dumps(result))

    def test_reasoning_effort_parameter_alias_and_null_effort_list_supported(self):
        self.urlopen.side_effect = [Response(catalog(
            supported_parameters=["response_format", "reasoning_effort"],
            reasoning={"supported_efforts": None, "mandatory": False}))]
        preflight = self.client(reasoning_effort="none").preflight()
        self.assertEqual(preflight["reasoning"]["supported_efforts"], None)
        self.assertEqual(self.client(reasoning_effort="none").reasoning_options(preflight),
                         {"reasoning": {"effort": "none"}})

    def test_requested_effort_fails_before_spend_when_capability_is_unavailable(self):
        variants = [catalog(), catalog(supported_parameters=["response_format", "reasoning"],
                                      reasoning={"supported_efforts": ["high"]}),
                    catalog(supported_parameters=["response_format", "reasoning"],
                            reasoning={"mandatory": False}),
                    catalog(supported_parameters=["response_format", "reasoning"],
                            reasoning={"supported_efforts": []})]
        for model in variants:
            self.urlopen.reset_mock()
            self.urlopen.side_effect = [Response(model)]
            client = self.client(reasoning_effort="low")
            with self.subTest(case="unsupported"), self.assertRaisesRegex(GenerationError, "reasoning"):
                client.generate("When?", EVIDENCE)
            self.assertEqual(self.urlopen.call_count, 1)
            with closing(sqlite3.connect(client.ledger.path)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM spend").fetchone()[0], 0)

    def test_mandatory_reasoning_never_silently_disables(self):
        self.urlopen.side_effect = [Response(catalog(
            supported_parameters=["response_format", "reasoning"],
            reasoning={"supported_efforts": None, "mandatory": True}))]
        with self.assertRaisesRegex(GenerationError, "requires reasoning"):
            self.client(reasoning_effort="none").generate("When?", EVIDENCE)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_malformed_reasoning_catalog_metadata_cannot_leak_content(self):
        for reasoning in (SECRET, {"supported_efforts": SECRET}, {"supported_efforts": [SECRET]},
                          {"mandatory": SECRET}, {"default_enabled": SECRET},
                          {"supports_max_tokens": SECRET}, {"default_effort": SECRET}):
            self.urlopen.side_effect = [Response(catalog(reasoning=reasoning))]
            with self.subTest(case="malformed-capabilities"), self.assertRaises(GenerationError) as caught:
                self.client().preflight()
            self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))

    def test_incomplete_response_retains_only_safe_finish_and_usage_after_settlement(self):
        payload = completion()
        payload["choices"][0] = {"finish_reason": "length", "native_finish_reason": SECRET,
                                  "message": {"content": SECRET, "reasoning": SECRET}}
        payload["usage"] = {"cost": 0, "prompt_tokens": 10, "completion_tokens": 4096,
                            "total_tokens": 4106, "completion_tokens_details": {
                                "reasoning_tokens": 4090, "provider_note": SECRET}, "raw": SECRET}
        self.urlopen.side_effect = [Response(catalog()), Response(payload)]
        client = self.client()
        with self.assertRaises(IncompleteResponseError) as caught:
            client.generate("When?", EVIDENCE)
        self.assertEqual(caught.exception.finish_reason, "length")
        self.assertEqual(caught.exception.as_dict(), {"type": "incomplete_response",
            "finish_reason": "length", "token_usage": {"prompt_tokens": 10, "completion_tokens": 4096,
            "total_tokens": 4106, "reasoning_tokens": 4090, "visible_completion_tokens": 6}})
        self.assertNotIn(SECRET, "".join(traceback.format_exception(caught.exception)))
        self.assertNotIn(SECRET, json.dumps(caught.exception.as_dict()))
        self.assertEqual(client.ledger.summary()["unconfirmed_requests"], 0)
        self.assertEqual(self.urlopen.call_count, 2)

    def test_incomplete_diagnostics_sanitize_unknown_reasons_and_bad_token_counts(self):
        for reason in (SECRET, [], None):
            error = IncompleteResponseError(reason, {"prompt_tokens": True, "completion_tokens": SECRET,
                "total_tokens": -1, "completion_tokens_details": {"reasoning_tokens": float("nan")}})
            self.assertIsNone(error.finish_reason)
            self.assertEqual(error.token_usage, {})
            self.assertNotIn(SECRET, str(error) + json.dumps(error.as_dict()))
        for reason in ("tool_calls", "content_filter", "error", "length", "stop"):
            self.assertEqual(IncompleteResponseError(reason).finish_reason, reason)
        invalid = IncompleteResponseError("length", {"completion_tokens": 4,
            "completion_tokens_details": {"reasoning_tokens": 5}})
        self.assertEqual(invalid.token_usage, {"completion_tokens": 4, "reasoning_tokens": 5})


if __name__ == "__main__":
    unittest.main()
