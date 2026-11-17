                                                                                
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from magra_rag import configuration as cfg
from magra_rag import generation as gen
from magra_rag.resources import ResourceBudget


class MonthlyPaidTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="paid-monthly-fictional-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "paid/runs/spend.sqlite"
        self.path.parent.mkdir(parents=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE spend (id TEXT PRIMARY KEY, reserved REAL NOT NULL, charged REAL, state TEXT NOT NULL, created_at TEXT NOT NULL)")
        self.now = datetime(2027, 2, 1, tzinfo=timezone.utc)
        for replacement in (
            patch.object(cfg, "PAID_LEDGER_PATH", self.path),
            patch.object(cfg, "_dotenv", side_effect=AssertionError("No environment file read")),
            patch.object(socket.socket, "connect", side_effect=AssertionError("No live network")),
            patch("urllib.request.urlopen", side_effect=AssertionError("No live API")),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        clock = patch.object(gen, "datetime", wraps=datetime).start()
        clock.now.side_effect = lambda _tz: self.now
        self.addCleanup(patch.stopall)

    def seed(self, amount, *, reserved=None, state="settled", created="2027-01-31T23:59:59+00:00"):
        with closing(sqlite3.connect(self.path)) as db, db:
            identifier = "fixture-%d" % db.execute("SELECT COUNT(*) FROM spend").fetchone()[0]
            db.execute("INSERT INTO spend VALUES(?,?,?,?,?)", (identifier, amount if reserved is None else reserved, amount, state, created))

    def rows(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT * FROM spend ORDER BY rowid").fetchall()

    def client(self, **options):
        return gen.OpenRouterM3(data_root=self.root / "owned-run", profile="paid", api_key="invented-monthly-key", env_file=None, **options)

    @staticmethod
    def catalog():
        return {"data": [{"id": cfg.PAID_MODEL, "context_length": 1048576, "pricing": {"prompt": "0.00000015", "completion": "0.00000060"}, "supported_parameters": ["response_format", "reasoning"]}]}

    def test_month_rollover_preserves_lifetime_rows_and_starts_new_period(self):
        self.seed(99.9)
        original = self.rows()
        client = self.client()
        reservation = client.ledger.reserve(1)
        client.ledger.settle(reservation, 1)
        summary = client.ledger.summary()
        self.assertAlmostEqual(summary["accounted_usd"], 100.9)
        self.assertEqual((summary["budget_period"], summary["period"], summary["period_accounted_usd"]), ("monthly_utc", "2027-02", 1))
        self.assertEqual(self.rows()[:1], original)
        self.now = datetime(2027, 3, 1, tzinfo=timezone.utc)
        reopened = self.client()
        reservation = reopened.ledger.reserve(100)
        reopened.ledger.settle(reservation, 100)
        self.assertEqual(reopened.ledger.summary()["period_remaining_usd"], 0)
        self.assertAlmostEqual(reopened.ledger.summary()["accounted_usd"], 200.9)
        self.assertEqual(self.rows()[:1], original)

    def test_old_unconfirmed_charge_blocks_construction_and_existing_client(self):
        client = self.client()
        self.seed(None, reserved=1, state="unconfirmed")
        before = self.rows()
        self.assertEqual(client.ledger.summary()["unconfirmed_requests"], 1)
        for action in (self.client, lambda: client.ledger.reserve(.01)):
            with self.assertRaisesRegex(gen.GenerationError, "Uncertain"):
                action()
        self.assertEqual(self.rows(), before)

    def test_all_months_validate_money_and_complete_utc_timestamps(self):
        invalid = (
            {"amount": 1, "created": "2027-01-31"},
            {"amount": 1, "created": "2027-01-31T23:59:59+01:00"},
            {"amount": 1, "created": "2027-02-30T00:00:00Z"},
            {"amount": 1, "created": "2027-03-01T00:00:00Z"},
            {"amount": 1, "reserved": 101},
            {"amount": 2, "reserved": 1},
            {"amount": -1, "reserved": 1},
        )
        for fixture in invalid:
            with self.subTest(fixture=fixture):
                with closing(sqlite3.connect(self.path)) as db, db:
                    db.execute("DELETE FROM spend")
                self.seed(**fixture)
                before = self.rows()
                with self.assertRaises(gen.GenerationError):
                    self.client()
                self.assertEqual(self.rows(), before)

    def test_prior_month_over100_cannot_be_hidden_by_rollover(self):
        self.seed(60)
        self.seed(41)
        before = self.rows()
        with self.assertRaisesRegex(gen.GenerationError, "monthly"):
            self.client()
        self.assertEqual(self.rows(), before)

    def test_transaction_current_month_cap_and_pending_halt_across_clients(self):
        self.seed(99.9, created="2027-02-01T00:00:00Z")
        first, second = self.client(), self.client()
        reservation = first.ledger.reserve(.1)
        with self.assertRaises(gen.GenerationError):
            second.ledger.reserve(.00001)
        first.ledger.settle(reservation, .1)
        before = self.rows()
        with self.assertRaisesRegex(gen.GenerationError, "spending limit"):
            second.ledger.reserve(.00001)
        self.assertEqual(self.rows(), before)
        self.assertEqual(second.ledger.summary()["period_accounted_usd"], 100)

    def test_bad_paid_cap_fails_before_any_sql(self):
        for value in (101, True, float("nan"), float("inf"), -1, "100"):
            with self.subTest(value=value), patch.object(gen.sqlite3, "connect", side_effect=AssertionError("No SQL")):
                with self.assertRaises(ValueError):
                    self.client(max_total_usd=value)

    def test_known_charge_above_reservation_is_retained_and_blocks_new_work(self):
        client = self.client()
        reservation = client.ledger.reserve(.1)
        client.ledger.settle(reservation, .2)
        before = self.rows()
        self.assertEqual(before[-1][1:4], (.1, .2, "settled"))
        for action in (self.client, lambda: client.ledger.reserve(.01), client.ledger.summary):
            with self.assertRaises(gen.GenerationError):
                action()
        self.assertEqual(self.rows(), before)

    def test_monthly100_key_uses_monthly_usage_instead_of_lifetime(self):
        client = self.client()
        key = {"limit": 100, "limit_remaining": 99, "limit_reset": "monthly", "usage": 1500, "usage_monthly": 1, "is_free_tier": False}
        with patch.object(client, "_json_request", side_effect=[self.catalog(), {"data": key}]):
            client.preflight()
        self.assertEqual(client._paid_remaining, 99)
        for modified in ({**key, "usage_monthly": None}, {**key, "limit_remaining": 100}, {**key, "limit_reset": "unexpected"}, {**key, "limit": 101}):
            with self.subTest(key=modified), patch.object(client, "_json_request", side_effect=[self.catalog(), {"data": modified}]):
                with self.assertRaises(gen.GenerationError):
                    client.preflight()

    def test_current_period_byok_is_included_only_when_requested(self):
        client = self.client()
        key = {"limit": 100, "limit_reset": "monthly", "usage": 2000, "usage_monthly": 2, "byok_usage_monthly": 3, "include_byok_in_limit": True, "limit_remaining": 95}
        with patch.object(client, "_json_request", side_effect=[self.catalog(), {"data": key}]):
            client.preflight()
        self.assertEqual(client._paid_remaining, 95)
        del key["byok_usage_monthly"]
        with patch.object(client, "_json_request", side_effect=[self.catalog(), {"data": key}]):
            with self.assertRaises(gen.GenerationError):
                client.preflight()

    def test_account_reset_fields_are_allowlisted_and_unknown_text_dropped(self):
        client = self.client()
        with patch.object(client, "_json_request", return_value={"data": {"limit_reset": "monthly", "usage": 900, "usage_monthly": 4, "label": "invented-secret-label"}}):
            result = client.account_limits()
        self.assertEqual(result["data"], {"limit_reset": "monthly", "usage": 900, "usage_monthly": 4})
        self.assertNotIn("invented-secret-label", json.dumps(result))

    def test_paid_reservation_covers_catalog_capacity_instead_of_byte_estimate(self):
        client = self.client()
        with self.assertRaisesRegex(gen.GenerationError, "capacity"):
            client.paid_reservation(b'{}')
        key = {"limit": 100, "usage": 0, "limit_remaining": 100}
        with patch.object(client, "_json_request", side_effect=[self.catalog(), {"data": key}]):
            client.preflight()
        self.assertEqual(client.paid_reservation(b'{}'), .159744)
        self.assertEqual(client.paid_reservation(b'x' * 50000), .159744)
        client._paid_remaining = .159743
        with self.assertRaisesRegex(gen.GenerationError, "balance"):
            client.paid_reservation(b'{}')

    def test_paid_preflight_rejects_missing_capacity_and_unaccounted_price(self):
        client = self.client()
        for capacity in (None, True, 0, 16_000_001):
            fixture = self.catalog()
            fixture["data"][0]["context_length"] = capacity
            with self.subTest(capacity=capacity), patch.object(client, "_json_request", return_value=fixture):
                with self.assertRaisesRegex(gen.GenerationError, "capacity"):
                    client.preflight()
        for category in ("request", "internal_reasoning", "unknown_fee", "input_cache_write"):
            fixture = self.catalog()
            fixture["data"][0]["pricing"][category] = "0.1"
            with self.subTest(category=category), patch.object(client, "_json_request", return_value=fixture):
                with self.assertRaises(gen.GenerationError):
                    client.preflight()
        self.assertEqual(self.rows(), [])

    def test_all_scheduled_prices_must_fit_without_assuming_a_discount(self):
        client = self.client()
        override = {"utc_start": 1500, "utc_end": 100, "prompt": ".000000075", "completion": ".0000003", "input_cache_read": ".0000000075"}
        fixture = self.catalog()
        fixture["data"][0]["pricing"]["overrides"] = [override]
        key = {"limit": 100, "usage": 0, "limit_remaining": 100}
        with patch.object(client, "_json_request", side_effect=[fixture, {"data": key}]):
            client.preflight()
        self.assertEqual(client.paid_reservation(b'{}'), .159744)
        for change in ({"completion": ".00000061"}, {"prompt": ".00000016"},
                       {"unknown_fee": ".00000001"}, {"utc_start": True},
                       {"utc_start": 1260}, {"overrides": [override]}):
            modified = self.catalog()
            modified["data"][0]["pricing"]["overrides"] = [{**override, **change}]
            with self.subTest(change=change), patch.object(client, "_json_request", return_value=modified):
                with self.assertRaises(gen.GenerationError):
                    client.preflight()
        self.assertEqual(self.rows(), [])


class PaidKeyPriorityTests(unittest.TestCase):
    def test_increased_key_has_priority_over_old_environment_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fictional.env"
            path.write_text("ai-model-key-increased-paid=new-file-key\nai-model-paid-key=old-file-key\n", encoding="utf-8")
            settings = cfg.load_openrouter_settings(env_file=path, environ={"ai-model-paid-key": "old-env-key"}, profile="auto")
            self.assertEqual((settings.api_key, settings.profile), ("new-file-key", "paid"))
            empty = cfg.load_openrouter_settings(env_file=path, environ={"ai-model-key-increased-paid": ""}, profile="paid")
            self.assertIsNone(empty.api_key)
            path.write_text("ai-model-key-increased-paid=first\nai-model-key-increased-paid=second\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                cfg.load_openrouter_settings(env_file=path, environ={}, profile="paid")


if __name__ == "__main__":
    unittest.main()
