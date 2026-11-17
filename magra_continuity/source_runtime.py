import argparse
from pathlib import Path


def module_description(path):
    try:
        from .config import load_mysql_config
        from .store import ContinuityStore
        from .source_store import source_doc
        workspace = Path(__file__).resolve().parent.parent
        with ContinuityStore(load_mysql_config(workspace / ".env"), workspace) as store:
            return source_doc(store, path, kind="module_docstring", symbol=None)
    except Exception:
        return None


class SourceDocumentedParser(argparse.ArgumentParser):
    def __init__(self, *args, source_path=None, **kwargs):
        self.source_path = source_path
        super().__init__(*args, **kwargs)

    def format_help(self):
        previous = self.description
        if previous is None and self.source_path:
            self.description = module_description(self.source_path)
        try:
            return super().format_help()
        finally:
            self.description = previous
