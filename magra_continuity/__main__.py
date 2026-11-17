                                                                                  
import argparse
import base64
import json
from pathlib import Path

from .config import ContinuityError, load_mysql_config
from .store import ContinuityStore
from .source_runtime import SourceDocumentedParser


def read_json(path):
    try:
        path = Path(path)
        if path.stat().st_size > 16_000_000:
            raise ContinuityError("JSON_input_too_large")
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        raise ContinuityError("JSON_input_unavailable_or_invalid") from None


def value_args(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", dest="json_value")
    group.add_argument("--json-file")


def parser():
    p = SourceDocumentedParser(source_path="magra_continuity/__main__.py")
    p.add_argument("--workspace", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--env", dest="env_file", help="Defaults to WORKSPACE/.env; mysql is a literal password")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap"); sub.add_parser("status")
    q = sub.add_parser("import-docs"); q.add_argument("paths", nargs="+")
    q = sub.add_parser("index-files"); q.add_argument("metadata_json"); q.add_argument("--active-content", action="append", default=[])
    q = sub.add_parser("latest"); q.add_argument("path"); q.add_argument("--bytes-base64", action="store_true")
    q = sub.add_parser("search"); q.add_argument("query"); q.add_argument("--files", action="store_true"); q.add_argument("--limit", type=int, default=20)
    q = sub.add_parser("path"); q.add_argument("path")
    q = sub.add_parser("list"); q.add_argument("parent", nargs="?", default="."); q.add_argument("--limit", type=int, default=100); q.add_argument("--after")
    q = sub.add_parser("state"); actions = q.add_subparsers(dest="action", required=True)
    r = actions.add_parser("get"); r.add_argument("key")
    r = actions.add_parser("list"); r.add_argument("--limit", type=int, default=100)
    r = actions.add_parser("set"); r.add_argument("key"); r.add_argument("--reason", default="explicit_state_update"); value_args(r)
    q = sub.add_parser("event"); actions = q.add_subparsers(dest="action", required=True)
    r = actions.add_parser("append"); r.add_argument("kind"); r.add_argument("--key"); value_args(r)
    r = actions.add_parser("list"); r.add_argument("--kind"); r.add_argument("--limit", type=int, default=20)
    q = sub.add_parser("docs"); actions = q.add_subparsers(dest="action", required=True)
    r = actions.add_parser("status")
    r = actions.add_parser("list"); r.add_argument("path", nargs="?"); r.add_argument("--limit", type=int, default=100); r.add_argument("--before-id", type=int)
    r = actions.add_parser("get"); r.add_argument("path"); r.add_argument("--source-id", type=int); r.add_argument("--kind"); r.add_argument("--symbol"); r.add_argument("--limit", type=int, default=200); r.add_argument("--after-ordinal", type=int)
    r = actions.add_parser("search"); r.add_argument("query"); r.add_argument("--limit", type=int, default=20)
    return p


def command(store, args):
    if args.command == "bootstrap": return store.bootstrap()
    if args.command == "status": return store.status()
    if args.command == "import-docs": return store.import_paths(args.paths)
    if args.command == "index-files":
        records = read_json(args.metadata_json)
        if not isinstance(records, list): raise ContinuityError("file_metadata_array_required")
        roles = {store.relative_path(record["path"]): record["role"] for record in records}
        content = {}
        for name in args.active_content:
            relative = store.content_path(name); path = store.workspace/relative
            if roles.get(relative) != "active": raise ContinuityError("explicit_active_content_only")
            if path.stat().st_size > 1_000_000: raise ContinuityError("active_content_too_large")
            content[relative] = path.read_bytes()
        return store.index_files(records, active_content=content)
    if args.command == "latest":
        result = store.latest(args.path, include_bytes=args.bytes_base64)
        if result is not None and args.bytes_base64:
            result["original_bytes_base64"] = base64.b64encode(result.pop("original_bytes")).decode("ascii")
        return result
    if args.command == "search":
        return (store.search_paths if args.files else store.search)(args.query, limit=args.limit)
    if args.command == "path": return store.path(args.path)
    if args.command == "list": return store.list_paths(args.parent, limit=args.limit, after=args.after)
    if args.command == "state" and args.action == "get": return store.get_state(args.key)
    if args.command == "state" and args.action == "list": return store.states(limit=args.limit)
    if args.command == "event" and args.action == "list": return store.events(kind=args.kind, limit=args.limit)
    if args.command == "docs":
        from . import source_store
        if args.action == "status": return source_store.source_counts(store)
        if args.action == "list": return source_store.source_file_versions(store, args.path, limit=args.limit, before_id=args.before_id)
        if args.action == "get": return source_store.source_documentation(store, args.path, source_id=args.source_id, kind=args.kind, symbol=args.symbol, limit=args.limit, after_ordinal=args.after_ordinal)
        if args.action == "search": return source_store.search_source_documentation(store, args.query, limit=args.limit)
    try:
        value = read_json(args.json_file) if args.json_file else json.loads(args.json_value)
    except ValueError:
        raise ContinuityError("value_must_be_JSON") from None
    if args.command == "state": return store.set_state(args.key, value, reason=args.reason)
    if args.command == "event": return store.event(args.kind, value, key=args.key)
    raise ContinuityError("unknown_continuity_command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        workspace = Path(args.workspace).resolve()
        config = load_mysql_config(args.env_file or workspace/".env")
        with ContinuityStore(config, workspace) as store:
            result = command(store, args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except ContinuityError as error:
        print(json.dumps({"status": "error", "code": str(error)}))
        return 1
    except Exception:
        print(json.dumps({"status": "error", "code": "continuity_command_failed"}))
        return 1


if __name__ == "__main__": raise SystemExit(main())
