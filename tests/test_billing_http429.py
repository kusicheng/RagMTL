from copy import deepcopy
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import unittest
import sqlite3
from unittest.mock import patch

from magra_rag import billing_coverage as coverage, generation
from magra_rag.resources import ResourceBudget
from test_billing_recovery import InterruptedBillingRecoveryTests, OLD, NEW, NOW, canonical, digest


THIRD = ['terminal429-unknown', .159744, None, 'reserved', '2026-11-03T00:00:00+00:00']
OBSERVATION = 'evidence-dependency-v1-000005'


class HTTP429BillingTests(unittest.TestCase):
    setUp = InterruptedBillingRecoveryTests.setUp
    pin = InterruptedBillingRecoveryTests.pin
    fixture = InterruptedBillingRecoveryTests.fixture

    def fixture429(self):
        old, body, model, preflight, _ = self.fixture()
        base_pin = self.pin('v2-base.json', old)
        catalog = self.pin('third/http/0001.bin', {'data': [model]})
        preflight = deepcopy(preflight)
        preflight['observation_id'] = OBSERVATION
        for name in ('catalog', 'account'):
            preflight[name]['checked_at'] = '2026-11-02T23:59:59+00:00'
        common = {'observation_id': OBSERVATION, 'headers_recorded': False}
        records = [{**common, 'method': method, 'endpoint': endpoint, 'HTTP_status': status,
                    'dispatch_started_monotonic': i + 1}
                   for i, (method, endpoint, status) in enumerate((('GET', '/models', 200), ('GET', '/key', 200), ('POST', '/chat/completions', 429)))]
        records[0].update(raw_file='http/0001.bin', raw_sha256=catalog['sha256'], raw_bytes=catalog['bytes'])
        pins = {'catalog': catalog,
                'request_body': self.pin('third/requests/' + OBSERVATION + '.bin', canonical(body), raw=True),
                'sealed_request_body': self.pin('sealed/' + OBSERVATION + '.bin', canonical(body), raw=True),
                'preflights': self.pin('third/preflights.jsonl', canonical(preflight) + b'\n', raw=True),
                'http': self.pin('third/HTTP.jsonl', b''.join(canonical(r) + b'\n' for r in records), raw=True)}
        entry = {'row': list(THIRD), 'row_sha256': digest(canonical(THIRD)), 'observation_id': OBSERVATION,
                 'actual_cost_usd': None, 'upper_bound_kind': 'catalog_context_plus_max_output',
                 'dispatch_status': 'http429_billing_unknown', 'no_retry': True, 'evidence': pins}
        manifest = {'schema': 'magra.http429-billing-coverage.v3', 'authorized_at_utc': '2026-11-03T00:00:00+00:00',
                    'monthly_ceiling_usd': 100, 'confirmed_exhausted': False, 'base_coverage': base_pin,
                    'http429_requests': [entry]}
        return manifest, body, records

    def load429(self, manifest):
        pin = self.pin('v3-coverage.json', manifest)
        with patch.object(coverage, 'datetime', wraps=datetime) as clock:
            clock.now.return_value = NOW
            return coverage.load_http429_billing_coverage(self.root, pin['path'], expected_sha256=pin['sha256'])

    def save_http(self, manifest, records):
        evidence = manifest['http429_requests'][0]['evidence']
        evidence['http'] = self.pin(evidence['http']['path'], b''.join(canonical(r) + b'\n' for r in records), raw=True)

    def test_three_full_unknowns_with_opaque_request_and_rollover(self):
        manifest, _, _ = self.fixture429()
        original = coverage._strict
        def guarded(raw):
            self.assertNotIn(b'PAYLOAD_CANARY', raw)
            self.assertNotIn(b'OPAQUE_HTTP429_CANARY', raw)
            return original(raw)
        with patch.object(coverage, '_strict', side_effect=guarded):
            policy = self.load429(manifest)
        self.assertEqual(policy.covered_entries, (tuple(OLD), tuple(NEW), tuple(THIRD)))
        for when in (NOW, datetime(2026, 12, 2, tzinfo=timezone.utc)):
            report = policy.validate_rows_and_reserved_carry([OLD, NEW, THIRD], when)
            self.assertEqual(report.full_unknown_reserved_usd, Decimal('.479232'))
            self.assertEqual(report.covered_unconfirmed_requests, 3)
            self.assertTrue(report.admission_permitted)
        over = ['known', 100, 99.520769, 'settled', '2026-11-03T00:00:00+00:00']
        with self.assertRaises(coverage.BillingCoverageError):
            policy.validate_rows_and_reserved_carry([OLD, NEW, THIRD, over], NOW)

    def test_default_and_v2_acceptance_not_widened(self):
        manifest, _, _ = self.fixture429()
        pin = self.pin('wrong-loader.json', manifest)
        for loader in (coverage.load_billing_coverage, coverage.load_interrupted_billing_coverage):
            with self.assertRaises(coverage.BillingCoverageError):
                loader(self.root, pin['path'], expected_sha256=pin['sha256'])
        with self.assertRaises(coverage.BillingCoverageError):
            coverage.validate_rows_and_reserved_carry([OLD, NEW, THIRD], NOW)
        old, _, _, _, record = self.fixture()
        evidence = old['interrupted_requests'][0]['evidence']
        evidence['http'] = self.pin(evidence['http']['path'], canonical(record) + b'\n' + canonical({**record, 'method': 'POST', 'endpoint': '/chat/completions', 'HTTP_status': 429}) + b'\n', raw=True)
        pin = self.pin('old-v2-post.json', old)
        with self.assertRaises(coverage.BillingCoverageError):
            coverage.load_interrupted_billing_coverage(self.root, pin['path'], expected_sha256=pin['sha256'])

    def test_terminal_trace_must_be_exact_failure_without_usage_or_success(self):
        for case in ('status200', 'missingPOST', 'doublePOST', 'missingGETkey', 'raw_file', 'usage', 'cost', 'headers', 'clock'):
            manifest, _, records = self.fixture429()
            if case == 'status200': records[-1]['HTTP_status'] = 200
            elif case == 'missingPOST': records.pop()
            elif case == 'doublePOST': records.append(deepcopy(records[-1]))
            elif case == 'missingGETkey': records.pop(1)
            elif case == 'headers': records[-1]['headers_recorded'] = True
            elif case == 'clock': records[-1]['dispatch_started_monotonic'] = -1
            else: records[-1][case] = {} if case == 'usage' else 0
            self.save_http(manifest, records)
            with self.subTest(case=case), self.assertRaises(coverage.BillingCoverageError): self.load429(manifest)

    def test_base_row_scope_and_unknown_state_reject(self):
        for case in ('baseV1', 'empty', 'tooMany', 'duplicate', 'settled', 'zeroCharge', 'underreserve', 'unknownDispatch', 'exhausted'):
            manifest, _, _ = self.fixture429()
            entry = manifest['http429_requests'][0]
            if case == 'baseV1': manifest['base_coverage'] = self.fixture()[0]['base_coverage']
            elif case == 'empty': manifest['http429_requests'] = []
            elif case == 'tooMany': manifest['http429_requests'] = [deepcopy(entry) for _ in range(7)]
            elif case == 'duplicate': manifest['http429_requests'].append(deepcopy(entry))
            elif case == 'settled': entry['row'][3] = 'settled'
            elif case == 'zeroCharge': entry['actual_cost_usd'] = 0
            elif case == 'underreserve': entry['row'][1] = .159743
            elif case == 'unknownDispatch': entry['dispatch_status'] = 'unknown'
            else: manifest['confirmed_exhausted'] = True
            entry['row_sha256'] = digest(canonical(entry['row']))
            with self.subTest(case=case), self.assertRaises(coverage.BillingCoverageError): self.load429(manifest)

    def test_saved_body_model_and_fallback_reject(self):
        for case in ('bodyMismatch', 'model', 'fallback'):
            manifest, body, _ = self.fixture429()
            evidence = manifest['http429_requests'][0]['evidence']
            if case == 'model': body['model'] = 'wrong/model'
            elif case == 'fallback': body['provider']['allow_fallbacks'] = True
            for role in ('request_body', 'sealed_request_body'):
                evidence[role] = self.pin(evidence[role]['path'], canonical(body), raw=True)
            if case == 'bodyMismatch': evidence['request_body'] = self.pin(evidence['request_body']['path'], b'changed', raw=True)
            with self.subTest(case=case), self.assertRaises(coverage.BillingCoverageError): self.load429(manifest)

    def test_three_rows_immutable_and_new_unknown_still_blocks(self):
        manifest, _, _ = self.fixture429()
        policy = self.load429(manifest)
        for column, value in ((0, 'other'), (1, 0), (2, 0), (3, 'unconfirmed')):
            changed = list(THIRD); changed[column] = value
            with self.subTest(column=column), self.assertRaises(coverage.BillingCoverageError):
                policy.validate_rows_and_reserved_carry([OLD, NEW, changed], NOW)
        fourth = ['new-uncovered', .159744, None, 'reserved', THIRD[4]]
        with self.assertRaises(coverage.BillingCoverageError):
            policy.validate_rows_and_reserved_carry([OLD, NEW, THIRD, fourth], NOW)

    def test_temporary_shared_ledger_protects_all_three_ids(self):
        manifest, _, _ = self.fixture429()
        policy = self.load429(manifest)
        path = self.root / 'spend.sqlite'
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE spend(id TEXT PRIMARY KEY,reserved REAL NOT NULL,charged REAL,state TEXT NOT NULL,created_at TEXT NOT NULL)')
            db.executemany('INSERT INTO spend VALUES(?,?,?,?,?)', [OLD, NEW, THIRD])
        before = path.read_bytes()
        ledger = generation.SpendLedger(path, 100, budget=ResourceBudget(self.root), require_existing=True, halt_on_unconfirmed=True, monthly=True, billing_coverage=policy)
        ledger.validate_admission()
        for row in (OLD, NEW, THIRD):
            for charged in (None, 0, .01):
                with self.assertRaises(generation.GenerationError): ledger.settle(row[0], charged)
        self.assertEqual(path.read_bytes(), before)
        pending = ledger.reserve(.01)
        ledger.settle(pending, None)
        with self.assertRaises(generation.GenerationError): ledger.reserve(.01)


if __name__ == '__main__':
    unittest.main()
