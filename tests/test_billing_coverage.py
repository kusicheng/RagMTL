                                                                               

from copy import deepcopy
from contextlib import closing
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from magra_rag import billing_coverage as coverage


ROW = ["8d02b5b6-db29-4835-be2d-53653703d333", .159744, None, "unconfirmed", "2026-11-01T02:35:21.092304+00:00"]
R = Decimal(".159744")
NOW = datetime.fromisoformat("2026-11-02T00:00:00+00:00")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class BillingCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psutil
        process = psutil.Process()
        process.cpu_affinity([process.cpu_affinity()[0]])

    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        evidence_root = root / "proof"
        evidence_root.mkdir()
        body = {"model": "example/model", "max_tokens": 4096, "stream": False,
                "messages": [{"role": "user", "content": "QUERY_CANARY with escaped quote \" and emoji \U0001f600"}],
                "provider": {"only": ["example"], "require_parameters": True, "allow_fallbacks": False,
                             "max_price": {"prompt": .15, "completion": .60}}}
        raw_body = canonical(body)
        raw_response = b"RAW_RESPONSE_CANARY: never decoded, even as JSON"
        intent = {"reservation_id": ROW[0], "POST_attempts": 1, "reserved_USD": ".159744", "model": body["model"],
                  "max_tokens": 4096, "no_retry": True, "provider_only": ["example"], "body_sha256": digest(raw_body)}
        receipt = {"HTTP_status": 429, "body_complete": True, "over_capture_bound": False, "raw_before_parse_and_billing": True,
                   "request_sha256": digest(raw_body), "sha256": digest(raw_response), "bytes": len(raw_response)}
        catalog = {"endpoints": {"chosen": {"model_id": body["model"], "tag": "example", "context_length": 1048576,
                    "max_completion_tokens": 943718, "pricing": {"prompt": ".00000015", "completion": ".0000006", "discount": "0", "input_cache_read": ".000000015"},
                    "supported_parameters": ["max_tokens"], "unrelated": "QUERY_CANARY"}}}
        values = {"request_body": raw_body, "response_body": raw_response, "request_intent": canonical(intent),
                  "response_receipt": canonical(receipt), "catalog": canonical(catalog)}
        pins = {}
        for role, raw in values.items():
            path = evidence_root / (role + ".bin")
            path.write_bytes(raw)
            pins[role] = {"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": digest(raw)}
        manifest = {"schema": "magra.billing-coverage.v1", "authorized_at_utc": "2026-11-01T03:00:00+00:00", "monthly_ceiling_usd": 100,
                    "confirmed_exhausted": False, "covered_request": {"row": list(ROW), "row_sha256": digest(canonical(ROW)), "actual_cost_usd": None,
                    "upper_bound_kind": "catalog_context_plus_max_output", "catalog_endpoint_path": ["endpoints", "chosen"], "evidence": pins}}
        return root, manifest, values

    def load(self, root, manifest):
        path = root / "coverage.json"
        raw = canonical(manifest)
        path.write_bytes(raw)
        return coverage.load_billing_coverage(root, path, expected_sha256=digest(raw))

    def replace(self, root, manifest, role, value):
        raw = canonical(value)
        pin = manifest["covered_request"]["evidence"][role]
        (root / pin["path"]).write_bytes(raw)
        pin.update(bytes=len(raw), sha256=digest(raw))

    def test_pinned_complete_proof_preserves_real_unknown_and_skips_payloads(self):
        root, manifest, _ = self.fixture()
        self.assertEqual(manifest["covered_request"]["row_sha256"], "b38cdcb7908882e82f86bf79256e81977467b4896001d294a98b8e21d8f5eca5")
        strict = coverage._strict
        def guarded(raw):
            self.assertNotIn(b"QUERY_CANARY", raw)
            self.assertNotIn(b"RAW_RESPONSE_CANARY", raw)
            return strict(raw)
        with patch.object(coverage, "_strict", side_effect=guarded):
            policy = self.load(root, manifest)
        report = policy.validate_rows_and_reserved_carry([ROW], NOW)
        self.assertEqual(policy.conservative_bound_usd, R)
        self.assertEqual((report.unconfirmed_requests, report.covered_unconfirmed_requests, report.uncovered_unconfirmed_requests), (1, 1, 0))
        self.assertEqual(report.full_unknown_reserved_usd, R)
        self.assertEqual(report.known_charged_usd, 0)
        self.assertEqual(report.as_dict()["full_unknown_reserved_usd"], "0.159744")
        self.assertTrue(report.admission_permitted)
        self.assertIsNone(ROW[2])
        with self.assertRaises(TypeError):
            coverage.BillingCoveragePolicy()

    def test_strict_default_extra_unknown_and_summary_counts(self):
        root, manifest, _ = self.fixture()
        policy = self.load(root, manifest)
        with self.assertRaises(coverage.BillingCoverageError):
            coverage.validate_rows_and_reserved_carry([ROW], NOW)
        extra = ["new-unknown", .2, None, "reserved", "2026-10-01T00:00:00+00:00"]
        rows = [ROW, extra]
        with self.assertRaises(coverage.BillingCoverageError) as failure:
            policy.validate_rows_and_reserved_carry(rows, NOW)
        self.assertEqual((failure.exception.unconfirmed_requests, failure.exception.covered_unconfirmed_requests,
                          failure.exception.uncovered_unconfirmed_requests), (2, 1, 1))
        summary = policy.validate_rows_and_reserved_carry(rows, NOW, for_admission=False)
        self.assertEqual((summary.unconfirmed_requests, summary.covered_unconfirmed_requests, summary.uncovered_unconfirmed_requests), (2, 1, 1))
        self.assertEqual(summary.full_unknown_reserved_usd, R + Decimal(".2"))
        self.assertEqual(summary.period_reserved_carry_usd, Decimal(".2"))
        self.assertFalse(summary.admission_permitted)
        default_summary = coverage.validate_rows_and_reserved_carry(rows, NOW, for_admission=False)
        self.assertEqual(default_summary.covered_unconfirmed_requests, 0)
        self.assertEqual(default_summary.uncovered_unconfirmed_requests, 2)
        for flag in (0, 1, None, "false"):
            with self.assertRaises(coverage.BillingCoverageError):
                policy.validate_rows_and_reserved_carry(rows, NOW, for_admission=flag)

    def test_cross_month_full_reservation_carries_without_double_billing(self):
        root, manifest, _ = self.fixture()
        policy = self.load(root, manifest)
        known = ["known", 1., .1, "settled", "2026-11-01T04:00:00+00:00"]
        same = policy.validate_rows_and_reserved_carry([ROW, known], NOW)
        self.assertEqual(same.period_reserved_carry_usd, 0)
        self.assertEqual(same.period_recorded_usd, R + Decimal(".1"))
        for date in ("2026-12-01T00:00:00+00:00", "2028-01-01T00:00:00+00:00"):
            report = policy.validate_rows_and_reserved_carry([ROW, known], datetime.fromisoformat(date))
            self.assertEqual(report.period_reserved_carry_usd, R)
            self.assertEqual(report.period_effective_accounted_usd, R)
            self.assertEqual(report.lifetime_accounted_usd, same.lifetime_accounted_usd)
            self.assertEqual(report.known_charged_usd, Decimal(".1"))

    def test_cap_boundary_and_over_limit_summary_never_approves(self):
        root, manifest, _ = self.fixture()
        policy = self.load(root, manifest)
        december = datetime.fromisoformat("2026-12-02T00:00:00+00:00")
        exact = ["known", 100., 99.840256, "settled", "2026-12-01T00:00:00+00:00"]
        report = policy.validate_rows_and_reserved_carry([ROW, exact], december)
        self.assertEqual(report.period_effective_accounted_usd, Decimal(100))
        self.assertEqual(report.period_remaining_usd, 0)
        over = list(exact); over[2] = 99.840257
        with self.assertRaises(coverage.BillingCoverageError):
            policy.validate_rows_and_reserved_carry([ROW, over], december)
        summary = policy.validate_rows_and_reserved_carry([ROW, over], december, for_admission=False)
        self.assertEqual(summary.period_effective_accounted_usd, Decimal("100.000001"))
        self.assertEqual(summary.period_remaining_usd, 0)
        self.assertFalse(summary.admission_permitted)

    def test_corrupt_evidence_and_covered_row_changes_remain_blocked_in_summary(self):
        root, manifest, _ = self.fixture()
        policy = self.load(root, manifest)
        for index, replacement in ((1, .1), (2, 0.), (3, "reserved"), (4, "2026-11-01T02:35:22+00:00")):
            changed = list(ROW); changed[index] = replacement
            for mode in (True, False):
                with self.assertRaises(coverage.BillingCoverageError):
                    policy.validate_rows_and_reserved_carry([changed], NOW, for_admission=mode)
        for mode in (True, False):
            with self.assertRaises(coverage.BillingCoverageError):
                policy.validate_rows_and_reserved_carry([], NOW, for_admission=mode)
        (root / manifest["covered_request"]["evidence"]["response_body"]["path"]).write_bytes(b"corrupt")
        with self.assertRaises(coverage.BillingCoverageError):
            self.load(root, manifest)

    def test_paths_hash_authorization_and_linked_evidence_reject(self):
        root, manifest, _ = self.fixture()
        for path in ("../outside.json", ".env", "proof/../response_body.bin", "proof\\response_body.bin", "C:/outside.json"):
            altered = deepcopy(manifest)
            altered["covered_request"]["evidence"]["response_body"]["path"] = path
            with self.assertRaises(coverage.BillingCoverageError):
                self.load(root, altered)
        body = root / manifest["covered_request"]["evidence"]["request_body"]["path"]
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda path: path == body or original(path)):
            with self.assertRaises(coverage.BillingCoverageError):
                self.load(root, manifest)
        path = root / "coverage.json"; path.write_bytes(canonical(manifest))
        for expected in (None, "0" * 64):
            with self.assertRaises(coverage.BillingCoverageError):
                coverage.load_billing_coverage(root, path, expected_sha256=expected)

    def test_catalog_bound_and_complete_HTTP429_proof_are_mandatory(self):
        for kind in ("price", "capacity", "receipt", "fallback", "intent", "bound"):
            root, manifest, data = self.fixture()
            if kind == "price":
                catalog = json.loads(data["catalog"]); catalog["endpoints"]["chosen"]["pricing"]["prompt"] = ".000001"
                self.replace(root, manifest, "catalog", catalog)
            elif kind == "capacity":
                catalog = json.loads(data["catalog"]); catalog["endpoints"]["chosen"]["max_completion_tokens"] = 4095
                self.replace(root, manifest, "catalog", catalog)
            elif kind == "receipt":
                receipt = json.loads(data["response_receipt"]); receipt["body_complete"] = False
                self.replace(root, manifest, "response_receipt", receipt)
            elif kind == "intent":
                intent = json.loads(data["request_intent"]); intent["reservation_id"] = "another-row"
                self.replace(root, manifest, "request_intent", intent)
            elif kind == "fallback":
                body = json.loads(data["request_body"]); body["provider"]["allow_fallbacks"] = True
                self.replace(root, manifest, "request_body", body)
                intent = json.loads(data["request_intent"]); intent["body_sha256"] = manifest["covered_request"]["evidence"]["request_body"]["sha256"]
                self.replace(root, manifest, "request_intent", intent)
                receipt = json.loads(data["response_receipt"]); receipt["request_sha256"] = intent["body_sha256"]
                self.replace(root, manifest, "response_receipt", receipt)
            else:
                manifest["covered_request"]["row"][1] = .159743
                manifest["covered_request"]["row_sha256"] = digest(canonical(manifest["covered_request"]["row"]))
                intent = json.loads(data["request_intent"]); intent["reserved_USD"] = ".159743"
                self.replace(root, manifest, "request_intent", intent)
            with self.subTest(kind=kind), self.assertRaises(coverage.BillingCoverageError):
                self.load(root, manifest)

    def test_unaccounted_pricing_rejects_and_cache_must_fit_prompt_bound(self):
        for category, amount in (("request", ".01"), ("image", ".01"), ("web_search", ".01"), ("internal_reasoning", ".0000001"),
                                 ("unknown_fee", ".0000001"), ("discount", ".1"), ("input_cache_write", ".000001")):
            root, manifest, data = self.fixture()
            catalog = json.loads(data["catalog"])
            catalog["endpoints"]["chosen"]["pricing"][category] = amount
            self.replace(root, manifest, "catalog", catalog)
            with self.subTest(category=category), self.assertRaises(coverage.BillingCoverageError):
                self.load(root, manifest)


class PaidBillingCoverageIntegrationTests(unittest.TestCase):
    pass                                                                     

    fixture = BillingCoverageTests.fixture
    load = BillingCoverageTests.load

    def setUp(self):
        import psutil
        from magra_rag import generation
        self.generation = generation
        process = psutil.Process()
        process.cpu_affinity([process.cpu_affinity()[0]])
        for target in ("urlopen", "build_opener"):
            blocker = patch.object(generation.urllib.request, target, side_effect=AssertionError("unplanned network access"))
            blocker.start(); self.addCleanup(blocker.stop)
        class FixedDatetime(datetime):
            current = NOW
            @classmethod
            def now(cls, tz=None):
                return cls.current
        self.clock = FixedDatetime
        patched = patch.object(generation, "datetime", FixedDatetime)
        patched.start(); self.addCleanup(patched.stop)

    def ledger_fixture(self, additional=()):
        from magra_rag.resources import ResourceBudget
        root, manifest, _ = self.fixture()
        policy = self.load(root, manifest)
        path = root / "runs/spend.sqlite"
        path.parent.mkdir()
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE spend (id TEXT PRIMARY KEY,reserved REAL NOT NULL,charged REAL,state TEXT NOT NULL,created_at TEXT NOT NULL)")
            db.executemany("INSERT INTO spend VALUES(?,?,?,?,?)", [ROW, *additional])
        return root, manifest, policy, path, ResourceBudget(root)

    def rows(self, path):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            return db.execute("SELECT id,reserved,charged,state,created_at FROM spend ORDER BY rowid").fetchall()

    def ledger(self, path, budget, policy=None):
        return self.generation.SpendLedger(path, 100, budget=budget, require_existing=True,
            halt_on_unconfirmed=True, monthly=True, billing_coverage=policy)

    def client(self, root, path, budget, policy):
        from magra_rag import configuration
        settings = configuration.OpenRouterSettings(api_key="FICTIONAL_KEY", model_id=configuration.PAID_MODEL, profile="paid")
        with patch.object(configuration, "PAID_LEDGER_PATH", path), patch.object(self.generation, "load_openrouter_settings", return_value=settings):
            return self.generation.OpenRouterM3(data_root=root, budget=budget, profile="paid", api_key="FICTIONAL_KEY", env_file=None,
                shared_ledger_path=path, billing_coverage=policy)

    def test_default_strict_constructor_and_explicit_coverage_preserve_history(self):
        root, _, policy, path, budget = self.ledger_fixture()
        before = self.rows(path); before_bytes = path.read_bytes()
        with self.assertRaises(self.generation.GenerationError):
            self.ledger(path, budget)
        self.assertEqual(path.read_bytes(), before_bytes)
        ledger = self.ledger(path, budget, policy)
        self.assertEqual(self.rows(path), before)
        self.assertEqual(ledger.summary()["unconfirmed_requests"], 1)
        for amount in (None, 0.):
            with self.assertRaises(self.generation.GenerationError):
                ledger.settle(ROW[0], amount)
        self.assertEqual(path.read_bytes(), before_bytes)
        identity = ledger.reserve(.01)
        ledger.settle(identity, .005)
        self.assertEqual(self.rows(path)[:len(before)], before)
        self.assertIsNone(self.rows(path)[0][2])
        self.assertEqual(ledger.summary()["billing_coverage"]["covered_unconfirmed_requests"], 1)
        with self.assertRaises(self.generation.GenerationError):
            self.ledger(path, budget)

    def test_atomic_cap_boundary_includes_old_month_carry(self):
        self.clock.current = datetime.fromisoformat("2026-12-02T00:00:00+00:00")
        known = ["known", 100., 99.83, "settled", "2026-12-01T00:00:00+00:00"]
        _, _, policy, path, budget = self.ledger_fixture([known])
        ledger = self.ledger(path, budget, policy)
        before = self.rows(path)
        identity = ledger.reserve(.010256)
        ledger.settle(identity, .010256)
        self.assertEqual(ledger.summary()["period_remaining_usd"], 0.)
        completed = self.rows(path)
        with self.assertRaises(self.generation.GenerationError):
            ledger.reserve(.000001)
        self.assertEqual(self.rows(path), completed)
        self.assertEqual(self.rows(path)[:len(before)], before)

    def test_new_unknown_summary_is_honest_and_reserve_preflight_stop_before_API(self):
        root, _, policy, path, budget = self.ledger_fixture()
        client = self.client(root, path, budget, policy)
        before = self.rows(path)
        identity = client.ledger.reserve(.01)
        client.ledger.settle(identity, None)
        summary = client.ledger.summary()
        self.assertEqual(summary["unconfirmed_requests"], 2)
        self.assertEqual(summary["billing_coverage"]["uncovered_unconfirmed_requests"], 1)
        self.assertFalse(summary["billing_coverage"]["admission_permitted"])
        captured = self.rows(path)
        with self.assertRaises(self.generation.GenerationError):
            client.ledger.reserve(.01)
        client._json_request = Mock(side_effect=AssertionError("API reached despite unknown billing"))
        with self.assertRaises(self.generation.GenerationError):
            client.preflight()
        client._json_request.assert_not_called()
        self.assertEqual(self.rows(path), captured)
        self.assertEqual(captured[:len(before)], before)

    def test_HTTP402_survives_real_generate_then_summary_of_reserved_unknown(self):
        root, _, policy, path, budget = self.ledger_fixture()
        client = self.client(root, path, budget, policy)
        before = self.rows(path)
        def wire(request, timeout):
            if request.get_method() == "POST":
                raise self.generation.PaidHTTPError(402)
            if request.full_url.endswith("/models"):
                return {"data": [{"id": client.model_id, "context_length": 1048576, "pricing": {"prompt": ".00000015", "completion": ".0000006"},
                                  "supported_parameters": ["response_format", "reasoning", "reasoning_effort"]}]}
            return {"data": {"limit": 100., "limit_remaining": 100., "usage_monthly": 0., "limit_reset": "monthly"}}
        client._json_request = Mock(side_effect=wire)
        with self.assertRaises(self.generation.PaidHTTPError) as failure:
            client.generate("fictional question", [{"doc_id": "fictional", "sentence_id": 0, "title": "", "text": "fictional evidence"}])
        summary = client.ledger.summary()
        self.assertEqual(failure.exception.status_code, 402)
        self.assertEqual(summary["unconfirmed_requests"], 2)
        self.assertEqual(summary["billing_coverage"]["uncovered_unconfirmed_requests"], 1)
        self.assertFalse(summary["billing_coverage"]["admission_permitted"])
        self.assertEqual(self.rows(path)[:len(before)], before)
        self.assertEqual(sum(call.args[0].get_method() == "POST" for call in client._json_request.call_args_list), 1)

    def test_CLI_pair_and_stale_SHA_fail_before_client_or_API(self):
        from magra_rag import cli, configuration
        root, _, _, path, budget = self.ledger_fixture()
        args = SimpleNamespace(command="generate", data_root=root, shared_ledger=path, max_usd=100., prompt_price=.15,
                               completion_price=.60, max_output_tokens=4096, reasoning_effort="low", billing_coverage=None, billing_coverage_sha256=None)
        settings = configuration.OpenRouterSettings(api_key="FICTIONAL_KEY", model_id=configuration.PAID_MODEL, profile="paid")
        with patch.object(configuration, "PAID_LEDGER_PATH", path), patch.object(cli, "__file__", str(root / "magra_rag/cli.py")),\
             patch.object(self.generation, "OpenRouterM3", side_effect=AssertionError("client created before validation")) as constructor:
            self.assertIsNone(cli._generation_options(args, settings)["billing_coverage"])
            for selected, pin in ((root / "coverage.json", None), (None, "0" * 64), (root / "coverage.json", "0" * 64)):
                args.billing_coverage, args.billing_coverage_sha256 = selected, pin
                with self.assertRaises(ValueError):
                    cli._generation_client(args, budget, settings=settings)
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
