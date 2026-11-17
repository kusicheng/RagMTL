                                                                          

                                                                             
                                                                               
                                                                              
                                             
   

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)\Z")
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_TOKEN = object()
_MILLION = Decimal(1_000_000)
_CEILING = Decimal(100)
_ROLES = {"request_intent", "request_body", "response_body", "response_receipt", "catalog"}


class BillingCoverageError(ValueError):
    pass                                                                        

    def __init__(self, code, *, unknown=None, covered=None):
        super().__init__(code)
        self.unconfirmed_requests = unknown
        self.covered_unconfirmed_requests = covered
        self.uncovered_unconfirmed_requests = None if unknown is None else unknown - covered


def _require(condition, code):
    if not condition:
        raise BillingCoverageError(code)


def _money(value):
    _require(type(value) in (int, float, str, Decimal), "invalid_monetary_amount")
    _require(len(str(value)) <= 128, "monetary_precision_bound")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise BillingCoverageError("invalid_monetary_amount") from None
    _require(result.is_finite() and result >= 0 and (result == 0 or -100 <= result.adjusted() <= 100), "invalid_monetary_amount")
    return result


def _timestamp(value, now=None):
    _require(isinstance(value, str) and _UTC.fullmatch(value), "complete_UTC_timestamp_required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BillingCoverageError("invalid_UTC_timestamp") from None
    _require(now is None or parsed <= now, "future_timestamp_not_allowed")
    return parsed


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_JSON_key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(BillingCoverageError("nonfinite_JSON")))
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise BillingCoverageError("invalid_JSON_evidence") from None


class _Scanner:
    pass                                                                         

    def __init__(self, raw):
        self.raw, self.pos = raw, 0

    def ws(self):
        while self.pos < len(self.raw) and self.raw[self.pos] in b" \t\r\n":
            self.pos += 1

    def take(self, byte):
        _require(self.pos < len(self.raw) and self.raw[self.pos] == byte, "invalid_JSON_structure")
        self.pos += 1

    def string(self):
        self.take(34)
        while self.pos < len(self.raw):
            byte = self.raw[self.pos]
            self.pos += 1
            if byte == 34:
                return
            _require(byte >= 32, "invalid_JSON_string")
            if byte == 92:
                _require(self.pos < len(self.raw), "invalid_JSON_escape")
                escaped = self.raw[self.pos]
                self.pos += 1
                if escaped == 117:
                    end = self.pos + 4
                    _require(end <= len(self.raw) and all(c in b"0123456789abcdefABCDEF" for c in self.raw[self.pos:end]), "invalid_JSON_escape")
                    self.pos = end
                else:
                    _require(escaped in b'"\\/bfnrt', "invalid_JSON_escape")
        raise BillingCoverageError("unterminated_JSON_string")

    def skip(self, depth=0):
        _require(depth <= 64, "JSON_depth_bound")
        self.ws()
        _require(self.pos < len(self.raw), "incomplete_JSON")
        byte = self.raw[self.pos]
        if byte == 34:
            self.string()
        elif byte in (123, 91):
            closing = 125 if byte == 123 else 93
            self.pos += 1
            self.ws()
            if self.pos < len(self.raw) and self.raw[self.pos] == closing:
                self.pos += 1
                return
            while True:
                if byte == 123:
                    self.string(); self.ws(); self.take(58)
                self.skip(depth + 1); self.ws()
                _require(self.pos < len(self.raw), "incomplete_JSON")
                if self.raw[self.pos] == closing:
                    self.pos += 1
                    return
                self.take(44); self.ws()
                _require(self.pos < len(self.raw) and self.raw[self.pos] != closing, "trailing_JSON_comma")
        else:
            for literal in (b"true", b"false", b"null"):
                if self.raw.startswith(literal, self.pos):
                    self.pos += len(literal)
                    return
            match = _NUMBER.match(self.raw, self.pos)
            _require(match is not None, "invalid_JSON_value")
            self.pos = match.end()


def _members(raw):
    scanner, result = _Scanner(raw), {}
    scanner.ws(); scanner.take(123); scanner.ws()
    if scanner.pos < len(raw) and raw[scanner.pos] == 125:
        scanner.pos += 1
    else:
        while True:
            start = scanner.pos
            scanner.string()
            _require(scanner.pos - start <= 256, "JSON_metadata_key_bound")
            key = _strict(raw[start:scanner.pos])
            _require(key not in result and len(result) < 4096, "duplicate_or_excess_JSON_keys")
            scanner.ws(); scanner.take(58); scanner.ws()
            start = scanner.pos
            scanner.skip(1)
            result[key] = raw[start:scanner.pos]
            scanner.ws()
            _require(scanner.pos < len(raw), "incomplete_JSON")
            if raw[scanner.pos] == 125:
                scanner.pos += 1
                break
            scanner.take(44); scanner.ws()
            _require(scanner.pos < len(raw) and raw[scanner.pos] != 125, "trailing_JSON_comma")
    scanner.ws()
    _require(scanner.pos == len(raw), "trailing_JSON_bytes")
    return result


def _project(raw, keys):
    members = _members(raw)
    _require(set(keys) <= set(members), "missing_financial_evidence_field")
    return {key: _strict(members[key]) for key in keys}


def _plain_messages(raw):
    scanner = _Scanner(raw)
    scanner.ws(); scanner.take(91); scanner.ws()
    count = 0
    while scanner.pos < len(raw) and raw[scanner.pos] != 93:
        start = scanner.pos
        scanner.skip(1)
        message = _members(raw[start:scanner.pos])
        role = _strict(message["role"]) if "role" in message else None
        _require(set(message) == {"role", "content"} and isinstance(role, str) and role in {"system", "user", "assistant"}
                 and message["content"].startswith(b'"'), "plain_text_messages_only")
        count += 1
        _require(count <= 1000, "message_count_bound")
        scanner.ws()
        if scanner.pos < len(raw) and raw[scanner.pos] == 93:
            break
        scanner.take(44); scanner.ws()
        _require(scanner.pos < len(raw) and raw[scanner.pos] != 93, "trailing_JSON_comma")
    scanner.take(93); scanner.ws()
    _require(count > 0 and scanner.pos == len(raw), "nonempty_plain_messages_required")


def _regular_project_file(root, relative):
    _require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative, "project_relative_evidence_path_required")
    parts = PurePosixPath(relative).parts
    _require(parts and not PurePosixPath(relative).is_absolute() and all(part not in {".", ".."} and not part.startswith(".") for part in parts)
             and PurePosixPath(relative).as_posix() == relative, "unsafe_evidence_path")
    path = root
    try:
        for part in parts:
            path /= part
            _require(not path.is_symlink() and not (hasattr(path, "is_junction") and path.is_junction()), "linked_evidence_not_allowed")
        _require(path.resolve(strict=True) == path.absolute() and path.is_relative_to(root)
                 and stat.S_ISREG(path.stat().st_mode), "regular_project_evidence_required")
    except OSError:
        raise BillingCoverageError("unavailable_evidence_file") from None
    _require(path.suffix.lower() not in {".sqlite", ".sqlite3", ".db", ".pem", ".key"}, "non_evidence_file_type")
    return path


def _read_pinned(root, pin, maximum):
    _require(isinstance(pin, dict) and set(pin) == {"path", "bytes", "sha256"}
             and type(pin["bytes"]) is int and 0 < pin["bytes"] <= maximum
             and isinstance(pin["sha256"], str) and _HEX.fullmatch(pin["sha256"]), "invalid_evidence_pin")
    path = _regular_project_file(root, pin["path"])
    try:
        before = path.stat()
        _require(before.st_size == pin["bytes"], "evidence_size_changed")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns), "evidence_file_replaced")
            raw = stream.read(maximum + 1)
        _regular_project_file(root, pin["path"])
        after = path.stat()
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                 and len(raw) == pin["bytes"] and _digest(raw) == pin["sha256"], "evidence_bytes_changed")
        return raw
    except OSError:
        raise BillingCoverageError("unavailable_evidence_file") from None


@dataclass(frozen=True, slots=True)
class CoverageReport:
    period: str
    limit_usd: Decimal
    lifetime_accounted_usd: Decimal
    known_charged_usd: Decimal
    period_recorded_usd: Decimal
    full_unknown_reserved_usd: Decimal
    covered_reserved_usd: Decimal
    period_reserved_carry_usd: Decimal
    period_effective_accounted_usd: Decimal
    period_remaining_usd: Decimal
    unconfirmed_requests: int
    covered_unconfirmed_requests: int
    uncovered_unconfirmed_requests: int
    manifest_sha256: str | None
    admission_permitted: bool

    def as_dict(self):
        pass                                                                   
        from dataclasses import fields
        return {field.name: str(value) if isinstance(value := getattr(self, field.name), Decimal) else value for field in fields(self)}


@dataclass(frozen=True, slots=True, init=False)
class BillingCoveragePolicy:
    manifest_sha256: str
    row_sha256: str
    covered_row: tuple
    authorized_at: datetime
    conservative_bound_usd: Decimal
    covered_entries: tuple
    _validated: object

    def __init__(self, *args, **kwargs):
        raise TypeError("Use load_billing_coverage with an explicit manifest hash")

    def validate_rows_and_reserved_carry(self, rows, now, *, monthly_limit_usd=100, for_admission=True):
        return validate_rows_and_reserved_carry(rows, now, coverage=self, monthly_limit_usd=monthly_limit_usd, for_admission=for_admission)

    def protects_id(self, identity):
        return any(row[0] == identity for row in self.covered_entries)


def load_billing_coverage(project_root, manifest_path, *, expected_sha256):
    pass                                                                           

                                                                             
                                                                             
                                                                             
       
    _require(isinstance(expected_sha256, str) and _HEX.fullmatch(expected_sha256), "explicit_manifest_SHA256_required")
    try:
        root = Path(project_root).resolve(strict=True)
        _require(root.is_dir(), "project_directory_required")
        candidate = Path(manifest_path)
        relative = candidate.relative_to(root).as_posix() if candidate.is_absolute() else candidate.as_posix()
        path = _regular_project_file(root, relative)
        raw = _read_pinned(root, {"path": relative, "bytes": path.stat().st_size, "sha256": expected_sha256}, 65_536)
    except (OSError, ValueError) as error:
        if isinstance(error, BillingCoverageError):
            raise
        raise BillingCoverageError("invalid_manifest_path") from None
    manifest = _strict(raw)
    _require(isinstance(manifest, dict) and set(manifest) == {"schema", "authorized_at_utc", "monthly_ceiling_usd", "confirmed_exhausted", "covered_request"}
             and manifest["schema"] == "magra.billing-coverage.v1" and type(manifest["monthly_ceiling_usd"]) is int and manifest["monthly_ceiling_usd"] == 100
             and manifest["confirmed_exhausted"] is False, "explicit_bounded_coverage_authorization_required")
    authorized_at = _timestamp(manifest["authorized_at_utc"])
    entry = manifest["covered_request"]
    _require(isinstance(entry, dict) and set(entry) == {"row", "row_sha256", "actual_cost_usd", "upper_bound_kind", "catalog_endpoint_path", "evidence"}
             and entry["actual_cost_usd"] is None and entry["upper_bound_kind"] == "catalog_context_plus_max_output", "one_explicit_unknown_row_proof_required")
    row = entry["row"]
    _require(isinstance(row, list) and len(row) == 5 and isinstance(row[0], str) and 0 < len(row[0]) <= 200
             and type(row[1]) in (int, float) and 0 < _money(row[1]) <= _CEILING and row[2] is None and row[3] == "unconfirmed"
             and _timestamp(row[4]) <= authorized_at and isinstance(entry["row_sha256"], str) and _HEX.fullmatch(entry["row_sha256"])
             and _digest(_canonical(row)) == entry["row_sha256"], "exact_preexisting_NULL_unconfirmed_row_required")
    evidence = entry["evidence"]
    _require(isinstance(evidence, dict) and set(evidence) == _ROLES, "complete_request_response_catalog_proof_required")
    data = {role: _read_pinned(root, evidence[role], 1_048_576 if role in {"request_body", "response_body", "catalog"} else 65_536) for role in sorted(_ROLES)}
    _require(len({pin["path"] for pin in evidence.values()}) == len(_ROLES), "distinct_evidence_files_required")
    intent = _project(data["request_intent"], ("reservation_id", "POST_attempts", "reserved_USD", "model", "max_tokens", "no_retry", "provider_only", "body_sha256"))
    receipt = _project(data["response_receipt"], ("HTTP_status", "body_complete", "over_capture_bound", "raw_before_parse_and_billing", "request_sha256", "sha256", "bytes"))
    _require(intent["reservation_id"] == row[0] and type(intent["POST_attempts"]) is int and intent["POST_attempts"] == 1 and intent["no_retry"] is True
             and _money(intent["reserved_USD"]) == _money(row[1]) and intent["body_sha256"] == evidence["request_body"]["sha256"], "one_original_request_and_full_reservation_binding")
    _require(type(receipt["HTTP_status"]) is int and receipt["HTTP_status"] == 429 and receipt["body_complete"] is True
             and receipt["over_capture_bound"] is False and receipt["raw_before_parse_and_billing"] is True
             and receipt["request_sha256"] == intent["body_sha256"] and receipt["sha256"] == evidence["response_body"]["sha256"]
             and type(receipt["bytes"]) is int and receipt["bytes"] == evidence["response_body"]["bytes"], "complete_immutable_HTTP429_capture_required")
    request_parts = _members(data["request_body"])
    _require({"model", "messages", "max_tokens", "stream", "provider"} <= set(request_parts)
             and set(request_parts) <= {"model", "messages", "temperature", "max_tokens", "stream", "reasoning", "response_format", "usage", "provider"}, "bounded_plain_chat_request_only")
    _plain_messages(request_parts["messages"])
    request = {key: _strict(request_parts[key]) for key in ("model", "max_tokens", "stream", "provider")}
    provider = request["provider"]
    _require(request["stream"] is False and type(request["max_tokens"]) is int and 0 < request["max_tokens"] <= 16_000_000
             and isinstance(intent["model"], str) and request["model"] == intent["model"]
             and type(intent["max_tokens"]) is int and request["max_tokens"] == intent["max_tokens"]
             and isinstance(provider, dict) and set(provider) == {"only", "require_parameters", "allow_fallbacks", "max_price"}
             and provider["only"] == intent["provider_only"] and isinstance(provider["only"], list) and len(provider["only"]) == 1
             and isinstance(provider["only"][0], str) and bool(provider["only"][0]) and provider["require_parameters"] is True and provider["allow_fallbacks"] is False
             and isinstance(provider["max_price"], dict) and set(provider["max_price"]) == {"prompt", "completion"}, "exact_bounded_request_controls_required")
    selected = data["catalog"]
    pointer = entry["catalog_endpoint_path"]
    _require(isinstance(pointer, list) and len(pointer) <= 8 and all(isinstance(key, str) and 0 < len(key) <= 128 for key in pointer), "catalog_object_path_required")
    for key in pointer:
        members = _members(selected)
        _require(key in members, "catalog_object_path_missing")
        selected = members[key]
    endpoint = _project(selected, ("model_id", "tag", "context_length", "max_completion_tokens", "pricing", "supported_parameters"))
    _require(endpoint["model_id"] == request["model"] and isinstance(request["model"], str) and bool(request["model"])
             and endpoint["tag"] == provider["only"][0] and type(endpoint["context_length"]) is int and 0 < endpoint["context_length"] <= 16_000_000
             and type(endpoint["max_completion_tokens"]) is int and 0 < request["max_tokens"] <= endpoint["max_completion_tokens"] <= 16_000_000
             and isinstance(endpoint["supported_parameters"], list) and "max_tokens" in endpoint["supported_parameters"]
             and isinstance(endpoint["pricing"], dict) and {"prompt", "completion"} <= set(endpoint["pricing"]), "same_catalog_model_provider_capacity_required")
    prompt, completion = (_money(provider["max_price"][key]) for key in ("prompt", "completion"))
    with localcontext() as context:
        context.prec, context.rounding = 256, ROUND_CEILING
        catalog_prompt = _money(endpoint["pricing"]["prompt"])
        for category, value in endpoint["pricing"].items():
            if category in {"prompt", "completion"}:
                continue
            amount = _money(value)
            if category in {"input_cache_read", "input_cache_write"}:
                _require(amount <= catalog_prompt, "cache_price_exceeds_prompt_bound")
            else:
                _require(amount == 0, "unaccounted_catalog_price_category")
        _require(catalog_prompt * _MILLION <= prompt and _money(endpoint["pricing"]["completion"]) * _MILLION <= completion,
                 "catalog_prices_exceed_explicit_request_ceilings")
        bound = ((endpoint["context_length"] * prompt + request["max_tokens"] * completion) / _MILLION).quantize(Decimal(".000000000000000001"), rounding=ROUND_CEILING)
    _require(0 < bound <= _money(row[1]), "full_catalog_financial_bound_not_reserved")
    policy = object.__new__(BillingCoveragePolicy)
    for name, value in {"manifest_sha256": expected_sha256, "row_sha256": entry["row_sha256"], "covered_row": tuple(row),
                        "authorized_at": authorized_at, "conservative_bound_usd": bound,
                        "covered_entries": (tuple(row),), "_validated": _TOKEN}.items():
        object.__setattr__(policy, name, value)
    return policy


def _interrupted_pricing(pricing, prompt, completion):
    _require(isinstance(pricing, dict) and {"prompt", "completion"} <= set(pricing), "interrupted_catalog_pricing_required")
    allowed = {"prompt", "completion", "input_cache_read", "input_cache_write", "request", "image", "web_search", "internal_reasoning"}
    overrides = pricing.get("overrides", [])
    _require(isinstance(overrides, list) and len(overrides) <= 16, "bounded_catalog_price_windows_required")
    windows = [pricing] + overrides
    for index, window in enumerate(windows):
        extra = {"overrides"} if index == 0 else {"utc_start", "utc_end"}
        _require(isinstance(window, dict) and {"prompt", "completion"} <= set(window) and set(window) <= allowed | extra,
                 "unsupported_interrupted_catalog_price_category")
        if index:
            _require({"utc_start", "utc_end"} <= set(window) and all(type(window[key]) is int and 0 <= window[key] <= 2359
                     and window[key] % 100 < 60 for key in ("utc_start", "utc_end")), "invalid_catalog_price_window")
        catalog_prompt = _money(window["prompt"])
        _require(catalog_prompt * _MILLION <= prompt and _money(window["completion"]) * _MILLION <= completion,
                 "interrupted_catalog_prices_exceed_request_ceilings")
        for key in set(window) - {"prompt", "completion", *extra}:
            amount = _money(window[key])
            _require(amount <= catalog_prompt if key in {"input_cache_read", "input_cache_write"} else amount == 0,
                     "interrupted_unaccounted_catalog_price")


def _interrupted_jsonl(raw, maximum):
    lines = raw.splitlines()
    _require(0 < len(lines) <= maximum and all(0 < len(line) <= 65536 for line in lines), "bounded_interrupted_JSONL_required")
    rows = [_strict(line) for line in lines]
    _require(all(isinstance(row, dict) for row in rows), "interrupted_JSONL_objects_required")
    return rows


def load_interrupted_billing_coverage(project_root, manifest_path, *, expected_sha256):
    _require(isinstance(expected_sha256, str) and _HEX.fullmatch(expected_sha256), "explicit_manifest_SHA256_required")
    try:
        root = Path(project_root).resolve(strict=True)
        _require(root.is_dir(), "project_directory_required")
        candidate = Path(manifest_path)
        relative = candidate.relative_to(root).as_posix() if candidate.is_absolute() else candidate.as_posix()
        path = _regular_project_file(root, relative)
        raw = _read_pinned(root, {"path": relative, "bytes": path.stat().st_size, "sha256": expected_sha256}, 65536)
    except (OSError, ValueError) as error:
        if isinstance(error, BillingCoverageError):
            raise
        raise BillingCoverageError("invalid_interrupted_manifest_path") from None
    manifest = _strict(raw)
    _require(isinstance(manifest, dict) and set(manifest) == {"schema", "authorized_at_utc", "monthly_ceiling_usd",
             "confirmed_exhausted", "base_coverage", "interrupted_requests"} and manifest["schema"] == "magra.interrupted-billing-coverage.v2"
             and type(manifest["monthly_ceiling_usd"]) is int and manifest["monthly_ceiling_usd"] == 100
             and manifest["confirmed_exhausted"] is False, "explicit_interrupted_coverage_authorization_required")
    authorized_at = _timestamp(manifest["authorized_at_utc"], datetime.now(timezone.utc))
    base_pin = manifest["base_coverage"]
    _read_pinned(root, base_pin, 65536)
    base = load_billing_coverage(root, base_pin["path"], expected_sha256=base_pin["sha256"])
    _require(base.authorized_at <= authorized_at, "base_coverage_postdates_interrupted_authorization")
    return _extend_interrupted_coverage(root, manifest["interrupted_requests"], authorized_at, base, expected_sha256)


def _extend_interrupted_coverage(root, entries, authorized_at, base, expected_sha256, *, http429=False):
    _require(isinstance(entries, list) and 1 <= len(entries) <= (6 if http429 else 8), "bounded_explicit_interrupted_entries_required")
    covered = list(base.covered_entries)
    identities = {row[0] for row in covered}
    observations = set()
    total_bound = base.conservative_bound_usd
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == {"row", "row_sha256", "observation_id", "actual_cost_usd",
                 "upper_bound_kind", "dispatch_status", "no_retry", "evidence"} and entry["actual_cost_usd"] is None
                 and entry["upper_bound_kind"] == "catalog_context_plus_max_output" and entry["dispatch_status"] == ("http429_billing_unknown" if http429 else "unknown")
                 and entry["no_retry"] is True, "explicit_dispatch_unknown_entry_required")
        row = entry["row"]
        _require(isinstance(row, list) and len(row) == 5 and isinstance(row[0], str) and 0 < len(row[0]) <= 200
                 and row[0] not in identities and type(row[1]) in (int, float) and 0 < _money(row[1]) <= _CEILING
                 and row[2] is None and row[3] in {"reserved", "unconfirmed"} and _timestamp(row[4], authorized_at)
                 and isinstance(entry["row_sha256"], str) and _HEX.fullmatch(entry["row_sha256"])
                 and _digest(_canonical(row)) == entry["row_sha256"], "exact_preexisting_interrupted_NULL_row_required")
        observation = entry["observation_id"]
        _require(isinstance(observation, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", observation)
                 and observation not in observations, "unique_bounded_interrupted_observation_required")
        evidence = entry["evidence"]
        roles = {"request_body", "sealed_request_body", "preflights", "catalog", "http"}
        _require(isinstance(evidence, dict) and set(evidence) == roles, "complete_interrupted_request_catalog_proof_required")
        maxima = {"request_body": 1048576, "sealed_request_body": 1048576, "preflights": 2097152, "catalog": 8000000, "http": 2097152}
        data = {role: _read_pinned(root, evidence[role], maxima[role]) for role in sorted(roles)}
        _require(len({pin["path"] for pin in evidence.values()}) == len(roles)
                 and all(PurePosixPath(evidence[role]["path"]).name == observation + ".bin" for role in ("request_body", "sealed_request_body"))
                 and data["request_body"] == data["sealed_request_body"], "saved_request_must_equal_distinct_sealed_body")
        parts = _members(data["request_body"])
        expected_parts = {"model", "messages", "temperature", "max_tokens", "stream", "provider", "response_format", "usage", "reasoning"}
        _require(set(parts) == expected_parts, "exact_interrupted_plain_chat_controls_required")
        _plain_messages(parts["messages"])
        request = {key: _strict(parts[key]) for key in expected_parts - {"messages"}}
        provider = request["provider"]
        _require(request["model"] == "minimax/minimax-m3.1-flash" and request["stream"] is False
                 and type(request["max_tokens"]) is int and request["max_tokens"] == 4096
                 and type(request["temperature"]) in (int, float) and request["temperature"] == 0
                 and request["response_format"] == {"type": "json_object"} and request["usage"] == {"include": True}
                 and request["reasoning"] == {"effort": "low"} and isinstance(provider, dict)
                 and set(provider) == {"only", "require_parameters", "allow_fallbacks", "max_price"}
                 and provider["only"] == ["fireworks"] and provider["require_parameters"] is True and provider["allow_fallbacks"] is False
                 and isinstance(provider["max_price"], dict) and set(provider["max_price"]) == {"prompt", "completion"},
                 "same_interrupted_model_provider_and_request_controls_required")
        prompt, completion = (_money(provider["max_price"][key]) for key in ("prompt", "completion"))
        _require(prompt == Decimal(".15") and completion == Decimal(".60"), "exact_interrupted_price_ceilings_required")
        preflights = _interrupted_jsonl(data["preflights"], 512)
        matching = [record for record in preflights if record.get("observation_id") == observation]
        _require(len(matching) == 1 and set(matching[0]) == {"observation_id", "catalog", "account"}, "one_original_interrupted_preflight_required")
        preflight = matching[0]
        pre_catalog, account = preflight["catalog"], preflight["account"]
        _require(isinstance(pre_catalog, dict) and isinstance(account, dict) and isinstance(account.get("data"), dict), "interrupted_preflight_metadata_required")
        _timestamp(pre_catalog.get("checked_at"), _timestamp(row[4]))
        _timestamp(account.get("checked_at"), _timestamp(row[4]))
        catalog = _strict(data["catalog"])
        _require(isinstance(catalog, dict) and isinstance(catalog.get("data"), list) and len(catalog["data"]) <= 10000,
                 "bounded_interrupted_models_catalog_required")
        matches = [model for model in catalog["data"] if isinstance(model, dict) and model.get("id") == request["model"]]
        _require(len(matches) == 1, "one_exact_interrupted_catalog_model_required")
        model = matches[0]
        catalog_hash = _digest(json.dumps(model, sort_keys=True).encode())
        capacity = model.get("context_length")
        _require(pre_catalog.get("id") == request["model"] and pre_catalog.get("catalog_sha256") == catalog_hash
                 and type(capacity) is int and 0 < capacity <= 16000000 and pre_catalog.get("context_length") == capacity
                 and pre_catalog.get("pricing") == model.get("pricing") and isinstance(model.get("supported_parameters"), list)
                 and "max_tokens" in model["supported_parameters"], "same_original_interrupted_catalog_hash_and_capacity_required")
        http = _interrupted_jsonl(data["http"], 2048)
        http_matches = [record for record in http if record.get("observation_id") == observation]
        catalogs = [record for record in http_matches if record.get("method") == "GET" and record.get("endpoint") == "/models"]
        _require(len(catalogs) == 1 and catalogs[0].get("HTTP_status") == 200
                 and catalogs[0].get("raw_sha256") == evidence["catalog"]["sha256"]
                 and catalogs[0].get("raw_bytes") == evidence["catalog"]["bytes"]
                 and isinstance(catalogs[0].get("raw_file"), str)
                 and (PurePosixPath(evidence["http"]["path"]).parent / catalogs[0]["raw_file"]).as_posix() == evidence["catalog"]["path"]
                 and (_terminal_http429_trace(http_matches) if http429 else not any(record.get("method") == "POST" for record in http_matches)), "original_catalog_capture_and_no_terminal_POST_required")
        with localcontext() as context:
            context.prec, context.rounding = 256, ROUND_CEILING
            _interrupted_pricing(model.get("pricing"), prompt, completion)
            bound = ((capacity * prompt + request["max_tokens"] * completion) / _MILLION).quantize(Decimal(".000000000000000001"), rounding=ROUND_CEILING)
            total_bound += bound
        _require(0 < bound <= _money(row[1]), "full_interrupted_catalog_financial_bound_not_reserved")
        identities.add(row[0]); observations.add(observation); covered.append(tuple(row))
    policy = object.__new__(BillingCoveragePolicy)
    values = {"manifest_sha256": expected_sha256, "row_sha256": base.row_sha256, "covered_row": base.covered_row,
              "authorized_at": authorized_at, "conservative_bound_usd": total_bound,
              "covered_entries": tuple(covered), "_validated": _TOKEN}
    for name, value in values.items():
        object.__setattr__(policy, name, value)
    return policy


def _terminal_http429_trace(records):
    fields = {"observation_id", "method", "endpoint", "dispatch_started_monotonic", "headers_recorded", "HTTP_status"}
    _require(len(records) == 3 and [(r.get("method"), r.get("endpoint"), r.get("HTTP_status")) for r in records] ==
             [("GET", "/models", 200), ("GET", "/key", 200), ("POST", "/chat/completions", 429)],
             "exact_original_GET_models_key_and_one_terminal_HTTP429_required")
    _require(set(records[0]) == fields | {"raw_file", "raw_sha256", "raw_bytes"}
             and set(records[1]) == fields and set(records[2]) == fields
             and all(r["headers_recorded"] is False for r in records),
             "terminal_HTTP429_must_have_no_raw_success_usage_or_settlement_claim")
    clocks = [r["dispatch_started_monotonic"] for r in records]
    _require(all(type(value) in (int, float) and value >= 0 and value < float("inf") for value in clocks)
             and clocks[0] <= clocks[1] <= clocks[2], "ordered_finite_HTTP429_dispatch_clocks_required")
    return True


def load_http429_billing_coverage(project_root, manifest_path, *, expected_sha256):
    _require(isinstance(expected_sha256, str) and _HEX.fullmatch(expected_sha256), "explicit_manifest_SHA256_required")
    try:
        root = Path(project_root).resolve(strict=True)
        _require(root.is_dir(), "project_directory_required")
        candidate = Path(manifest_path)
        relative = candidate.relative_to(root).as_posix() if candidate.is_absolute() else candidate.as_posix()
        path = _regular_project_file(root, relative)
        raw = _read_pinned(root, {"path": relative, "bytes": path.stat().st_size, "sha256": expected_sha256}, 65536)
    except (OSError, ValueError) as error:
        if isinstance(error, BillingCoverageError):
            raise
        raise BillingCoverageError("invalid_HTTP429_manifest_path") from None
    manifest = _strict(raw)
    _require(isinstance(manifest, dict) and set(manifest) == {"schema", "authorized_at_utc", "monthly_ceiling_usd",
             "confirmed_exhausted", "base_coverage", "http429_requests"}
             and manifest["schema"] == "magra.http429-billing-coverage.v3"
             and type(manifest["monthly_ceiling_usd"]) is int and manifest["monthly_ceiling_usd"] == 100
             and manifest["confirmed_exhausted"] is False, "explicit_HTTP429_coverage_authorization_required")
    authorized_at = _timestamp(manifest["authorized_at_utc"], datetime.now(timezone.utc))
    base_pin = manifest["base_coverage"]
    base_raw = _read_pinned(root, base_pin, 65536)
    _require(_strict(base_raw).get("schema") == "magra.interrupted-billing-coverage.v2", "HTTP429_base_must_be_original_v2_policy")
    base = load_interrupted_billing_coverage(root, base_pin["path"], expected_sha256=base_pin["sha256"])
    _require(len(base.covered_entries) == 2 and base.authorized_at <= authorized_at,
             "HTTP429_requires_two_preexisting_covered_rows_and_prior_authorization")
    return _extend_interrupted_coverage(root, manifest["http429_requests"], authorized_at, base, expected_sha256, http429=True)


def validate_rows_and_reserved_carry(rows, now, *, coverage=None, monthly_limit_usd=100, for_admission=True):
    pass                                                                       

                                                                            
                                                                              
                                                                           
                                                                             
                                                                               
                                                                        
       
    _require(type(for_admission) is bool, "explicit_admission_mode_required")
    with localcontext() as context:
        context.prec = 256
        return _validate_rows(rows, now, coverage, monthly_limit_usd, for_admission)


def _validate_rows(rows, now, coverage, monthly_limit_usd, for_admission):
    _require(isinstance(now, datetime) and now.tzinfo is not None and now.utcoffset() == timedelta(0), "UTC_now_required")
    limit = _money(monthly_limit_usd)
    _require(limit <= _CEILING and isinstance(rows, (list, tuple)) and len(rows) <= 100000, "bounded_monthly_history_required")
    _require(coverage is None or isinstance(coverage, BillingCoveragePolicy) and coverage._validated is _TOKEN, "validated_policy_required")
    if coverage is not None:
        _require(coverage.authorized_at <= now, "coverage_authorization_is_in_future")
    period, totals, lifetime, known = now.strftime("%Y-%m"), {}, Decimal(0), Decimal(0)
    unknown, covered, full_reserved, covered_reserved, carry = 0, 0, Decimal(0), Decimal(0), Decimal(0)
    identities = set()
    for row in rows:
        _require(isinstance(row, (list, tuple)) and len(row) == 5, "five_column_billing_rows_required")
        identity, reserved, charged, state, created_at = row
        _require(isinstance(identity, str) and identity and identity not in identities and state in {"reserved", "unconfirmed", "settled"}, "invalid_billing_row_identity_or_state")
        identities.add(identity)
        reserved = _money(reserved)
        _require(0 < reserved <= _CEILING, "invalid_billing_reservation")
        created = _timestamp(created_at, now); created_period = created.strftime("%Y-%m")
        expected = None if coverage is None else next((entry for entry in coverage.covered_entries if entry[0] == identity), None)
        listed = expected is not None
        if listed:
            _require(tuple(row) == expected and _digest(_canonical(list(row))) == _digest(_canonical(list(expected)))
                     and charged is None and state in {"reserved", "unconfirmed"}, "covered_row_changed_or_settled")
        if charged is None:
            _require(state in {"reserved", "unconfirmed"}, "NULL_charge_requires_unknown_state")
            unknown += 1; full_reserved += reserved; accounted = reserved
            if listed:
                covered += 1; covered_reserved += reserved
            if created_period != period:
                carry += reserved
        else:
            accounted = _money(charged)
            _require(state == "settled" and accounted <= reserved, "invalid_known_charge_or_state")
            known += accounted
        lifetime += accounted
        totals[created_period] = totals.get(created_period, Decimal(0)) + accounted
    _require(coverage is None or covered == len(coverage.covered_entries), "covered_preexisting_row_missing")
    if for_admission and unknown != covered:
        raise BillingCoverageError("uncovered_unknown_billing_blocks_admission", unknown=unknown, covered=covered)
    recorded = totals.get(period, Decimal(0)); effective = recorded + carry
    if for_admission:
        _require(all(total <= _CEILING for total in totals.values()) and effective <= limit, "monthly_cap_including_reserved_carry_exceeded")
    return CoverageReport(period, limit, lifetime, known, recorded, full_reserved, covered_reserved, carry, effective, max(Decimal(0), limit - effective),
                          unknown, covered, unknown - covered, None if coverage is None else coverage.manifest_sha256, for_admission)
