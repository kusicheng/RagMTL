import ast
import codecs
import hashlib
import unittest

from magra_continuity.source_extract import SourceExtractionError, analyze_source, semantic_digest


class SourceExtractTests(unittest.TestCase):
    def analyze(self, text, encoding="utf-8"):
        raw = text.encode(encoding)
        result = analyze_source(raw, "example.py")
        self.assertEqual(result["original_bytes"], raw)
        self.assertEqual(result["original_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["stripped_sha256"], hashlib.sha256(result["stripped_bytes"]).hexdigest())
        self.assertEqual(result["semantic_sha256"], semantic_digest(result["stripped_bytes"]))
        self.assertEqual(result["semantic_sha256"], semantic_digest(raw))
        self.assertEqual(result["original_line_count"], result["stripped_line_count"])
        for block in result["blocks"]:
            self.assertEqual(raw[block["start_byte"]:block["end_byte"]], block["raw_bytes"])
            self.assertEqual(hashlib.sha256(block["raw_bytes"]).hexdigest(), block["sha256"])
        compile(result["stripped_bytes"], "example.py", "exec", dont_inherit=True)
        return result

    def test_nested_docstrings_comments_and_runtime_literals(self):
        text = (
            '"""Module documentation."""\n'
            'from __future__ import annotations\n'
            'SYSTEM = "Keep # literal and documentation-like runtime text."\n'
            'SQL = """SELECT \'# value\'"""\n'
            '# Module explanation\n'
            'class Outer:\n'
            '    """Class documentation."""\n'
            '    value = 3  # Attribute explanation\n'
            '    """Attribute documentation."""\n'
            '    def inner(self):\n'
            '        """Method documentation."""\n'
            '        # Method explanation\n'
            '        return SYSTEM + str(self.value)\n'
        )
        result = self.analyze(text)
        self.assertEqual(len(result["blocks"]), 7)
        self.assertEqual(
            [b["symbol"] for b in result["blocks"] if b["kind"] == "function_docstring"],
            ["Outer.inner"],
        )
        self.assertIn('SYSTEM = "Keep # literal', result["stripped_bytes"].decode())
        self.assertIn('SQL = """SELECT', result["stripped_bytes"].decode())
        self.assertIsNone(ast.get_docstring(ast.parse(result["stripped_bytes"])))
        self.assertEqual(result["runtime_doc_references"], [])

    def test_empty_suites_semicolons_and_module_future_import(self):
        text = (
            '"module"; from __future__ import annotations\n'
            'class Empty: "class"\n'
            'def empty(): ""\n'
            'async def asynchronous(): ("async"); return 1\n'
            'if True:\n'
            '    ("standalone")\n'
            '"later"; value = 4\n'
        )
        result = self.analyze(text)
        stripped = result["stripped_bytes"].decode()
        self.assertTrue(stripped.startswith("from __future__"))
        self.assertIn('def empty(): pass', stripped)
        self.assertIn('class Empty: pass', stripped)
        self.assertIn('pass   ; value = 4', stripped)
        self.assertEqual(len(result["blocks"]), 6)

    def test_multiline_parentheses_comments_and_indentation(self):
        text = (
            '(\n'
            ' # Explanation inside module documentation\n'
            ' "module "\n'
            ' "documentation"\n'
            '); from __future__ import annotations\n'
            'def f():\n'
            '    (\n'
            '        "function "\n'
            '        # Explanation inside function documentation\n'
            '        "documentation"\n'
            '    )\n'
        )
        result = self.analyze(text)
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual(result["blocks"][0]["text"], "module documentation")
        self.assertNotIn("Explanation", result["stripped_bytes"].decode())
        self.assertIn("from __future__", result["stripped_bytes"].decode())

    def test_unicode_columns_CRLF_and_BOM(self):
        text = (
            '"""模块说明 café e\u0301."""\r\n'
            'from __future__ import annotations\r\n'
            'π = "# 数据"  # 解释内容\r\n'
            'def 名称():\r\n'
            '    """解释 U+2028 \u2028 保持原始字符。"""\r\n'
            '    return π\r\n'
        )
        result = self.analyze(text, "utf-8-sig")
        self.assertTrue(result["stripped_bytes"].startswith(codecs.BOM_UTF8))
        self.assertEqual(result["stripped_bytes"].count(b"\r\n"), 6)
        comment = next(b for b in result["blocks"] if b["kind"] == "comment")
        self.assertEqual(comment["raw_bytes"], "# 解释内容".encode())
        self.assertEqual(comment["start_col"], text.splitlines()[2].index("# 解释"))
        self.assertIn('π = "# 数据"', result["stripped_bytes"].decode("utf-8-sig"))

    def test_non_utf8_cookie_and_bare_carriage_returns(self):
        text = '# coding: latin-1\r"""caf\xe9"""\rvalue = "caf\xe9" # prose\r'
        result = self.analyze(text, "latin-1")
        self.assertEqual(result["encoding"], "iso-8859-1")
        self.assertEqual(result["blocks"][0]["preserve_reason"], "encoding_cookie")
        self.assertIn('value = "caf\xe9"', result["stripped_bytes"].decode("latin-1"))
        self.assertEqual(result["stripped_bytes"].count(b"\r"), 3)
        self.assertNotIn(b"prose", result["stripped_bytes"])

    def test_directives_and_entire_legal_notice_remain(self):
        text = (
            '#!/usr/bin/env python\n'
            '# coding: utf-8\n'
            '# Copyright 2026 Example\n'
            '# All rights reserved.\n'
            '# Permission is hereby granted for testing.\n'
            '\n'
            'value = 3  # type: int\n'
            'other = value  # noqa: F821\n'
            '# fmt: off\n'
            '# pyright: strict\n'
            '# Plain removable explanation\n'
            '# fmt: on\n'
        )
        result = self.analyze(text)
        reasons = [b["preserve_reason"] for b in result["blocks"]]
        self.assertIn("legal_notice", reasons)
        self.assertEqual(reasons.count(None), 1)
        output = result["stripped_bytes"].decode()
        self.assertIn("# Copyright", output)
        self.assertIn("# All rights reserved.", output)
        self.assertIn("# noqa", output)
        self.assertIn("# type: int", output)
        self.assertIn("# fmt: off", output)
        self.assertNotIn("Plain removable explanation", output)

    def test_license_docstrings_and_embedded_directives_are_preserved(self):
        text = (
            '"""Copyright 2026 Example. Licensed under the MIT license."""\n'
            'def f():\n'
            '    (\n'
            '        "Documentation"\n'
            '        # fmt: skip\n'
            '    )\n'
        )
        result = self.analyze(text)
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual(
            [b["preserve_reason"] for b in result["blocks"]],
            ["legal_notice", "contains_preserved_comment"],
        )
        self.assertEqual(result["stripped_bytes"], result["original_bytes"])

    def test_fstrings_bytes_and_assigned_fixtures_are_functional(self):
        text = (
            'value = """Example documentation kept as a fixture."""\n'
            'count = 1\n'
            'f"{count}"\n'
            'b"binary literal"\n'
            'mapping = {"# heading": "source string"}\n'
        )
        result = self.analyze(text)
        self.assertEqual(result["blocks"], [])
        self.assertEqual(result["original_bytes"], result["stripped_bytes"])

    def test_doc_reference_reporting_and_top_level_only_document(self):
        result = self.analyze('"""Example."""\nprint(__doc__)\n')
        self.assertEqual(result["runtime_doc_references"], [{"line": 2, "symbol": "__module__"}])
        self.assertEqual(self.analyze('""')["blocks"][0]["text"], "")
        self.assertEqual(self.analyze(b"".decode())["blocks"], [])

    def test_invalid_source_bytes_or_syntax_reject(self):
        for raw in [b"\xff", b"# coding: no-such-codec\n", b"def bad(:\n", b"return 1\n", b"x = \x00\n"]:
            with self.subTest(raw=raw), self.assertRaises(SourceExtractionError):
                analyze_source(raw, "example.py")
        for value in ["text", bytearray(b"x = 1\n"), None]:
            with self.subTest(value=value), self.assertRaises(SourceExtractionError):
                analyze_source(value, "example.py")
        with self.assertRaises(SourceExtractionError):
            analyze_source(b"x = 1\n", "")


if __name__ == "__main__":
    unittest.main()
