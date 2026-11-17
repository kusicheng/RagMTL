                                                                               

import errno
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from magra_rag.resources import tree_bytes


class TreeBytesRaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.transient = self.root / "vanishing-cache-directory"
        self.transient.mkdir()
        self.stable = self.root / "stable"
        self.stable.mkdir()
        (self.root / ".hidden-log").write_bytes(b"abc")
        (self.stable / "retained.bin").write_bytes(b"retained-data")
        self.expected = 3 + len(b"retained-data")
        self.scandir = os.scandir

    def test_directory_removed_after_enqueue_preserves_other_file_accounting(self):
        scanned = []

        def remove_when_opened(directory):
            directory = Path(directory)
            scanned.append(directory)
            if directory == self.transient:
                                                                               
                                                                             
                                                                            
                self.assertIn(self.root, scanned)
                self.transient.rmdir()
            return self.scandir(directory)

        with patch("magra_rag.resources.os.scandir", side_effect=remove_when_opened):
            self.assertEqual(tree_bytes(self.root), self.expected)
        self.assertIn(self.transient, scanned)
        self.assertFalse(self.transient.exists())
        self.assertEqual(tree_bytes(self.root), self.expected)

    def test_permission_error_opening_queued_directory_propagates(self):
        error = PermissionError(errno.EACCES, "Denied queued directory", str(self.transient))

        def deny_when_opened(directory):
            if Path(directory) == self.transient:
                raise error
            return self.scandir(directory)

        with patch("magra_rag.resources.os.scandir", side_effect=deny_when_opened):
            with self.assertRaises(PermissionError) as caught:
                tree_bytes(self.root)
        self.assertIs(caught.exception, error)

    def test_other_os_error_opening_queued_directory_propagates(self):
        error = OSError(errno.EIO, "Queued directory I/O failure", str(self.transient))

        def fail_when_opened(directory):
            if Path(directory) == self.transient:
                raise error
            return self.scandir(directory)

        with patch("magra_rag.resources.os.scandir", side_effect=fail_when_opened):
            with self.assertRaises(OSError) as caught:
                tree_bytes(self.root)
        self.assertIs(caught.exception, error)

    def test_escaping_link_still_rejected_and_internal_file_link_counted(self):
        link_path = self.root / "virtual-link"

        class LinkEntry:
            path = str(link_path)

            def stat(self, *, follow_symlinks):
                return SimpleNamespace(st_size=0, st_file_attributes=0)

            def is_symlink(self):
                return True

        with self.scandir(self.root) as entries:
            real_entries = list(entries)

        def scan_with_link(directory):
            if Path(directory) == self.root:
                return nullcontext(iter([*real_entries, LinkEntry()]))
            return self.scandir(directory)

        resolve = Path.resolve

        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "outside.bin"
            outside.write_bytes(b"outside")
                                                                            
                                                                          
            def resolve_outside(path, *args, **kwargs):
                return outside if path == link_path else resolve(path, *args, **kwargs)

            def resolve_inside(path, *args, **kwargs):
                target = self.stable / "retained.bin" if path == link_path else path
                return resolve(target, *args, **kwargs)

            with patch("magra_rag.resources.os.scandir", side_effect=scan_with_link):
                with patch.object(Path, "resolve", autospec=True, side_effect=resolve_outside):
                    with self.assertRaisesRegex(ValueError, "Link escapes"):
                        tree_bytes(self.root)
                with patch.object(Path, "resolve", autospec=True, side_effect=resolve_inside):
                    self.assertEqual(tree_bytes(self.root), self.expected + len(b"retained-data"))


if __name__ == "__main__":
    unittest.main()
