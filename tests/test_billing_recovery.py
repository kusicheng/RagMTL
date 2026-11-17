from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from magra_rag import billing_coverage as coverage, generation
from magra_rag.resources import ResourceBudget


OLD = ["original-unknown", .159744, None, "unconfirmed", "2026-11-01T02:35:21+00:00"]
NEW = ["interrupted-unknown", .159744, None, "reserved", "2026-11-01T22:07:40+00:00"]
NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)
OBSERVATION = "lossless-v1-000098"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class InterruptedBillingRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for target in ("urlopen", "build_opener"):
            blocker = patch.object(generation.urllib.request, target, side_effect=AssertionError("Unexpected network"))
            blocker.start(); self.addCleanup(blocker.stop)

    def pin(self, path, value, raw=False):
        body = value if raw else canonical(value)
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return {"path": path, "bytes": len(body), "sha256": digest(body)}

    def fixture(self):
        provider = {"only": ["fireworks"], "require_parameters": True, "allow_fallbacks": False,
                    "max_price": {"prompt": .15, "completion": .60}}
        body = {"model": "minimax/minimax-m3.1-flash", "messages": [{"role": "user", "content": "PAYLOAD_CANARY"}],
                "temperature": 0, "max_tokens": 4096, "stream": False, "provider": provider,
                "response_format": {"type": "json_object"}, "usage": {"include": True}, "reasoning": {"effort": "low"}}
        request = canonical(body)
        response = b"OPAQUE_HTTP429_CANARY"
        intent = {"reservation_id": OLD[0], "POST_attempts": 1, "reserved_USD": ".159744", "model": body["model"],
                  "max_tokens": 4096, "no_retry": True, "provider_only": ["fireworks"], "body_sha256": digest(request)}
        receipt = {"HTTP_status": 429, "body_complete": True, "over_capture_bound": False, "raw_before_parse_and_billing": True,
                   "request_sha256": digest(request), "sha256": digest(response), "bytes": len(response)}
        endpoint = {"model_id": body["model"], "tag": "fireworks", "context_length": 1048576, "max_completion_tokens": 524288,
                    "pricing": {"prompt": ".00000015", "completion": ".0000006"}, "supported_parameters": ["max_tokens"]}
        old_values = {"request_intent": canonical(intent), "request_body": request, "response_body": response,
                      "response_receipt": canonical(receipt), "catalog": canonical({"selected": endpoint})}
        old_pins = {role: self.pin("legacy/" + role + ".bin", raw, raw=True) for role, raw in old_values.items()}
        old = {"schema": "magra.billing-coverage.v1", "authorized_at_utc": "2026-11-01T03:00:00+00:00", "monthly_ceiling_usd": 100,
               "confirmed_exhausted": False, "covered_request": {"row": list(OLD), "row_sha256": digest(canonical(OLD)),
               "actual_cost_usd": None, "upper_bound_kind": "catalog_context_plus_max_output", "catalog_endpoint_path": ["selected"], "evidence": old_pins}}
        base_pin = self.pin("legacy/coverage.json", old)
        model = {"id": body["model"], "context_length": 1048576,
                 "pricing": {"prompt": ".000000075", "completion": ".0000003", "input_cache_read": ".0000000075",
                 "overrides": [{"utc_start": 100, "utc_end": 1500, "prompt": ".00000015", "completion": ".0000006"}]},
                 "supported_parameters": ["max_tokens", "response_format", "reasoning"]}
        catalog_pin = self.pin("attempt/http/0001.bin", {"data": [model]})
        preflight = {"observation_id": OBSERVATION, "catalog": {"id": body["model"], "context_length": model["context_length"],
                     "pricing": model["pricing"], "catalog_sha256": digest(json.dumps(model, sort_keys=True).encode()),
                     "checked_at": "2026-11-01T22:07:39+00:00"},
                     "account": {"checked_at": "2026-11-01T22:07:39.500000+00:00", "data": {"limit": 100, "limit_reset": "monthly"}}}
        http = {"observation_id": OBSERVATION, "method": "GET", "endpoint": "/models", "HTTP_status": 200,
                "raw_file": "http/0001.bin", "raw_sha256": catalog_pin["sha256"], "raw_bytes": catalog_pin["bytes"]}
        pins = {"request_body": self.pin("attempt/requests/" + OBSERVATION + ".bin", request, raw=True),
                "sealed_request_body": self.pin("prepared/bodies/" + OBSERVATION + ".bin", request, raw=True),
                "preflights": self.pin("attempt/preflights.jsonl", canonical(preflight) + b"\n", raw=True),
                "catalog": catalog_pin, "http": self.pin("attempt/HTTP.jsonl", canonical(http) + b"\n", raw=True)}
        entry = {"row": list(NEW), "row_sha256": digest(canonical(NEW)), "observation_id": OBSERVATION, "actual_cost_usd": None,
                 "upper_bound_kind": "catalog_context_plus_max_output", "dispatch_status": "unknown", "no_retry": True, "evidence": pins}
        manifest = {"schema": "magra.interrupted-billing-coverage.v2", "authorized_at_utc": "2026-11-03T00:00:00+00:00",
                    "monthly_ceiling_usd": 100, "confirmed_exhausted": False, "base_coverage": base_pin, "interrupted_requests": [entry]}
        return manifest, body, model, preflight, http

    def load(self, manifest):
        pin = self.pin("interrupted-coverage.json", manifest)
        with patch.object(coverage, "datetime", wraps=datetime) as clock:
            clock.now.return_value = NOW
            return coverage.load_interrupted_billing_coverage(self.root, pin["path"], expected_sha256=pin["sha256"])

    def update_body(self, manifest, body):
        evidence = manifest["interrupted_requests"][0]["evidence"]
        for role in ("request_body", "sealed_request_body"):
            evidence[role] = self.pin(evidence[role]["path"], body)

    def update_catalog(self, manifest, model, preflight, http):
        evidence = manifest["interrupted_requests"][0]["evidence"]
        evidence["catalog"] = self.pin(evidence["catalog"]["path"], {"data": [model]})
        preflight["catalog"].update(pricing=model["pricing"], context_length=model["context_length"],
                                    catalog_sha256=digest(json.dumps(model, sort_keys=True).encode()))
        evidence["preflights"] = self.pin(evidence["preflights"]["path"], canonical(preflight) + b"\n", raw=True)
        http.update(raw_bytes=evidence["catalog"]["bytes"], raw_sha256=evidence["catalog"]["sha256"])
        evidence["http"] = self.pin(evidence["http"]["path"], canonical(http) + b"\n", raw=True)

    def test_exact_composition_keeps_both_real_unknowns_and_opaque_request_content(self):
        manifest, _, _, _, _ = self.fixture()
        original = coverage._strict
        def safe(raw):
            self.assertNotIn(b"PAYLOAD_CANARY", raw)
            self.assertNotIn(b"OPAQUE_HTTP429_CANARY", raw)
            return original(raw)
        with patch.object(coverage, "_strict", side_effect=safe):
            policy = self.load(manifest)
        report = policy.validate_rows_and_reserved_carry([OLD, NEW], NOW)
        self.assertEqual(policy.covered_row, tuple(OLD))
        self.assertEqual(policy.covered_entries, (tuple(OLD), tuple(NEW)))
        self.assertTrue(policy.protects_id(OLD[0])); self.assertTrue(policy.protects_id(NEW[0]))
        self.assertEqual((report.unconfirmed_requests, report.covered_unconfirmed_requests, report.uncovered_unconfirmed_requests), (2, 2, 0))
        self.assertEqual(report.full_unknown_reserved_usd, Decimal(".319488"))
        self.assertEqual(report.known_charged_usd, 0)
        self.assertTrue(report.admission_permitted)

    def test_missing_tampered_and_extra_unknown_rows_fail_closed(self):
        manifest, _, _, _, _ = self.fixture()
        policy = self.load(manifest)
        for index, value in ((0, "other-id"), (1, .159743), (2, 0), (3, "unconfirmed"), (4, "2026-11-01T22:07:41+00:00")):
            changed = list(NEW); changed[index] = value
            for mode in (True, False):
                with self.subTest(index=index, mode=mode), self.assertRaises(coverage.BillingCoverageError):
                    policy.validate_rows_and_reserved_carry([OLD, changed], NOW, for_admission=mode)
        for rows in ([OLD], [NEW], [OLD, NEW, NEW]):
            with self.assertRaises(coverage.BillingCoverageError):
                policy.validate_rows_and_reserved_carry(rows, NOW)
        third = ["third-new-unknown", .01, None, "reserved", "2026-11-03T00:00:00+00:00"]
        with self.assertRaises(coverage.BillingCoverageError) as failure:
            policy.validate_rows_and_reserved_carry([OLD, NEW, third], NOW)
        self.assertEqual(failure.exception.uncovered_unconfirmed_requests, 1)

    def test_rollover_carries_both_full_reservations_once_and_enforces_cap(self):
        manifest, _, _, _, _ = self.fixture()
        policy = self.load(manifest)
        for date in (datetime(2026, 12, 2, tzinfo=timezone.utc), datetime(2027, 1, 2, tzinfo=timezone.utc)):
            report = policy.validate_rows_and_reserved_carry([OLD, NEW], date)
            self.assertEqual(report.period_reserved_carry_usd, Decimal(".319488"))
            self.assertEqual(report.period_effective_accounted_usd, Decimal(".319488"))
            self.assertEqual(report.lifetime_accounted_usd, Decimal(".319488"))
        known = ["known", 100, 99.680513, "settled", "2026-12-01T00:00:00+00:00"]
        with self.assertRaises(coverage.BillingCoverageError):
            policy.validate_rows_and_reserved_carry([OLD, NEW, known], datetime(2026, 12, 2, tzinfo=timezone.utc))

    def test_manifest_shape_row_hash_duplicate_future_and_bad_paths_reject(self):
        for case in ("empty", "duplicate", "base-duplicate", "future", "future-row", "rowhash", "extra", "evidence-extra", "path", "actual-cost"):
            manifest, _, _, _, _ = self.fixture()
            entry = manifest["interrupted_requests"][0]
            if case == "empty":manifest["interrupted_requests"] = []
            elif case == "duplicate":manifest["interrupted_requests"].append(deepcopy(entry))
            elif case == "base-duplicate":entry["row"] = list(OLD); entry["row_sha256"] = digest(canonical(OLD))
            elif case == "future":manifest["authorized_at_utc"] = "2026-11-04T00:00:00+00:00"
            elif case == "future-row":entry["row"][4] = "2026-11-04T00:00:00+00:00"; entry["row_sha256"] = digest(canonical(entry["row"]))
            elif case == "rowhash":entry["row_sha256"] = "0" * 64
            elif case == "extra":manifest["extra"] = True
            elif case == "evidence-extra":entry["evidence"]["response"] = entry["evidence"]["catalog"]
            elif case == "path":entry["evidence"]["request_body"]["path"] = "../outside.bin"
            else:entry["actual_cost_usd"] = 0
            with self.subTest(case=case), self.assertRaises(coverage.BillingCoverageError):self.load(manifest)

    def test_request_model_provider_controls_and_reserved_upper_bound_reject(self):
        for case in ("model", "fallback", "provider", "max_tokens", "stream", "price", "underreserved", "nontext", "saved-different"):
            manifest, body, _, _, _ = self.fixture()
            if case == "model":body["model"] = "different/model"
            elif case == "fallback":body["provider"]["allow_fallbacks"] = True
            elif case == "provider":body["provider"]["only"] = ["other"]
            elif case == "max_tokens":body["max_tokens"] = 4095
            elif case == "stream":body["stream"] = True
            elif case == "price":body["provider"]["max_price"]["prompt"] = .14
            elif case == "underreserved":
                row = manifest["interrupted_requests"][0]["row"]; row[1] = .159743
                manifest["interrupted_requests"][0]["row_sha256"] = digest(canonical(row))
            elif case == "nontext":body["messages"][0]["content"] = [{"type": "image_url", "image_url": "example"}]
            else:
                evidence = manifest["interrupted_requests"][0]["evidence"]
                evidence["request_body"] = self.pin(evidence["request_body"]["path"], b"different", raw=True)
            if case != "saved-different":self.update_body(manifest, body)
            with self.subTest(case=case), self.assertRaises(coverage.BillingCoverageError):self.load(manifest)

    def test_catalog_windows_metadata_missing_changed_and_response_claim_reject(self):
        for case in ("expensive-window", "unknown-fee", "capacity", "preflight-time", "preflight-model", "catalog-hash", "POST", "missing", "mutated"):
            manifest, _, model, preflight, http = self.fixture()
            if case == "expensive-window":model["pricing"]["overrides"][0]["completion"] = ".00000061"
            elif case == "unknown-fee":model["pricing"]["request"] = ".01"
            elif case == "capacity":model["context_length"] = 1048577
            elif case == "preflight-time":preflight["catalog"]["checked_at"] = "2026-11-01T22:07:41+00:00"
            elif case == "preflight-model":preflight["catalog"]["id"] = "other/model"
            elif case == "catalog-hash":preflight["catalog"]["catalog_sha256"] = "0" * 64
            elif case == "POST":http["method"] = "POST"
            elif case in {"missing", "mutated"}:
                path = self.root / manifest["interrupted_requests"][0]["evidence"]["catalog"]["path"]
                if case == "missing":path.unlink()
                else:path.write_bytes(b"mutated")
            if case not in {"missing", "mutated"}:
                if case == "catalog-hash":
                    pin = manifest["interrupted_requests"][0]["evidence"]["preflights"]
                    manifest["interrupted_requests"][0]["evidence"]["preflights"] = self.pin(pin["path"], canonical(preflight) + b"\n", raw=True)
                else:self.update_catalog(manifest, model, preflight, http)
            with self.subTest(case=case), self.assertRaises(coverage.BillingCoverageError):self.load(manifest)

    def test_actual_temporary_spendledger_protects_both_ids_and_new_unknown_blocks(self):
        manifest, _, _, _, _ = self.fixture()
        policy = self.load(manifest)
        path = self.root / "runs/spend.sqlite"; path.parent.mkdir()
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE spend(id TEXT PRIMARY KEY,reserved REAL NOT NULL,charged REAL,state TEXT NOT NULL,created_at TEXT NOT NULL)")
            db.executemany("INSERT INTO spend VALUES(?,?,?,?,?)", [OLD, NEW])
        before = path.read_bytes()
        ledger = generation.SpendLedger(path, 100, budget=ResourceBudget(self.root), require_existing=True,
                                        halt_on_unconfirmed=True, monthly=True, billing_coverage=policy)
        ledger.validate_admission()
        for identity in (OLD[0], NEW[0]):
            for amount in (None, 0, .01):
                with self.assertRaises(generation.GenerationError):ledger.settle(identity, amount)
        self.assertEqual(path.read_bytes(), before)
        identity = ledger.reserve(.01)
        ledger.settle(identity, .005)
        with closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute("SELECT * FROM spend ORDER BY rowid LIMIT 2").fetchall(), [tuple(OLD), tuple(NEW)])
        self.assertEqual(ledger.summary()["billing_coverage"]["covered_unconfirmed_requests"], 2)
        pending = ledger.reserve(.01)
        ledger.settle(pending, None)
        summary = ledger.summary()["billing_coverage"]
        self.assertEqual(summary["uncovered_unconfirmed_requests"], 1)
        with self.assertRaises(generation.GenerationError):ledger.validate_admission()
        with self.assertRaises(generation.GenerationError):ledger.reserve(.01)


if __name__ == "__main__":
    unittest.main()
