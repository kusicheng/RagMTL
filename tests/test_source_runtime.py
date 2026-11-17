import unittest
from unittest.mock import patch

from magra_continuity.source_runtime import SourceDocumentedParser, module_description


class SourceRuntimeTests(unittest.TestCase):
    def test_regular_parsing_and_subcommands_do_not_connect_for_documentation(self):
        with patch("magra_continuity.source_runtime.module_description", side_effect=AssertionError("Unexpected connection")):
            parser = SourceDocumentedParser(source_path="magra_rag/cli.py")
            sub = parser.add_subparsers(dest="command", required=True)
            sub.add_parser("run").add_argument("question")
            args = parser.parse_args(["run", "fictional question"])
            self.assertEqual(args.question, "fictional question")
            self.assertIn("usage:", sub.choices["run"].format_help())

    def test_help_uses_database_text_without_caching_plaintext_in_source(self):
        with patch("magra_continuity.source_runtime.module_description", return_value="Fictional database description") as read:
            parser = SourceDocumentedParser(prog="example", source_path="magra_rag/cli.py")
            self.assertIn("Fictional database description", parser.format_help())
            read.assert_called_once_with("magra_rag/cli.py")
            self.assertIsNone(parser.description)

    def test_explicit_runtime_description_has_no_database_dependency(self):
        with patch("magra_continuity.source_runtime.module_description", side_effect=AssertionError("Unexpected connection")):
            parser = SourceDocumentedParser(source_path="magra_rag/cli.py", description="Functional supplied text")
            self.assertIn("Functional supplied text", parser.format_help())

    def test_missing_database_does_not_disable_option_help(self):
        with patch("magra_continuity.source_runtime.module_description", return_value=None):
            parser = SourceDocumentedParser(source_path="magra_rag/cli.py")
            self.assertIn("--help", parser.format_help())
        with patch("magra_continuity.config.load_mysql_config", side_effect=RuntimeError("Unavailable")):
            self.assertIsNone(module_description("magra_rag/cli.py"))
