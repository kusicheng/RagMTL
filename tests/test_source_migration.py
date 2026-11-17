import tempfile
import unittest
from pathlib import Path

from magra_continuity.config import MySQLConfig
from magra_continuity.source_extract import analyze_source
from magra_continuity.source_store import _prepare
from magra_continuity.store import ContinuityStore


class SourceMigrationIntegrationTests(unittest.TestCase):
    def test_real_extractor_payload_passes_storage_validation_without_database(self):
        raw = b'"""Fictional module text."""\nfrom __future__ import annotations\n# Fictional explanation\nclass Failure(Exception):\n    """Fictional class text."""\n'
        with tempfile.TemporaryDirectory() as temporary:
            store = ContinuityStore(MySQLConfig(password="fictional"), Path(temporary))
            analysis = analyze_source(raw, "package/example.py")
            prepared = _prepare(store, [analysis])
            self.assertEqual([block["ordinal"] for block in prepared[0]["blocks"]], [0, 1, 2])
            self.assertEqual(prepared[0]["original_bytes"], raw)
            self.assertEqual(prepared[0]["semantic_sha256"], analysis["stripped_semantic_sha256"])
