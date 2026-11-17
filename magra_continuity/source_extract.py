import ast
import bisect
import codecs
import hashlib
import io
import json
import re
import tokenize


class SourceExtractionError(ValueError):
    pass


_LEGAL = re.compile(
    r"\bcopyright\b|\bSPDX(?:-License-Identifier|-FileCopyrightText)?\b|"
    r"\ball rights reserved\b|\blicensed? under\b|\blicen[sc]e (?:notice|terms)\b|"
    r"\bpermission is hereby granted\b|\b(?:MIT|BSD|Apache|GPL|LGPL|MPL) licen[sc]e\b",
    re.IGNORECASE,
)
_DIRECTIVE = re.compile(
    r"^#\s*(?:type\s*:|noqa\b|nosec\b|nosemgrep\b|"
    r"fmt\s*:|ruff\s*:|isort\s*:|yapf\s*:|autopep8\s*:|"
    r"pylint\s*:|pyright\s*:|mypy\s*:|pyre[-:]|pytype\s*:|"
    r"pragma\s*:|coverage\s*:|cython\s*:|"
    r"region\b|endregion\b|(?:end\s*)?region\b|vim\s*:|vi\s*:)",
    re.IGNORECASE,
)
_CODING = re.compile(r"coding[:=]\s*[-\w.]+")


def _need(value, message):
    if not value:
        raise SourceExtractionError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source(raw):
    _need(type(raw) is bytes, "Source input must be exact bytes")
    try:
        encoding_lines = iter(part + b"\n" for part in re.split(br"\r\n|\r|\n", raw, maxsplit=2)[:2])
        encoding, _ = tokenize.detect_encoding(lambda: next(encoding_lines, b""))
        original = raw.decode(encoding)
        _need(original.encode(encoding) == raw, "Source encoding does not round trip exactly")
        encoder = codecs.getincrementalencoder(encoding)()
        prefix = encoder.encode("", final=False)
        offsets = [len(prefix)]
        encoded = [prefix]
        for character in original:
            piece = encoder.encode(character, final=False)
            encoded.append(piece)
            offsets.append(offsets[-1] + len(piece))
        encoded.append(encoder.encode("", final=True))
        _need(b"".join(encoded) == raw, "Source byte mapping does not round trip")
    except (SyntaxError, UnicodeError, LookupError) as error:
        raise SourceExtractionError("Invalid Python source encoding") from error
    normalized = []
    positions = [0]
    index = 0
    while index < len(original):
        if original[index] == "\r":
            step = 2 if original[index:index + 2] == "\r\n" else 1
            normalized.append("\n")
        else:
            step = 1
            normalized.append(original[index])
        index += step
        positions.append(index)
    normalized = "".join(normalized)
    starts = [0]
    starts.extend(match.end() for match in re.finditer("\n", normalized))
    return encoding, original, normalized, positions, offsets, starts


def _parse(text, path):
    try:
        tree = ast.parse(text, filename=path, type_comments=True)
        compile(tree, path, "exec", dont_inherit=True)
        return tree
    except (SyntaxError, ValueError, TypeError) as error:
        raise SourceExtractionError("Invalid Python source syntax") from error


class _SemanticTree(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Pass(self, node):
        return None


def _semantic(tree):
    normalized = _SemanticTree().visit(tree)
    value = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return _sha(value.encode("utf-8"))


def semantic_digest(raw):
    _, _, normalized, _, _, _ = _source(raw)
    return _semantic(_parse(normalized, "<source>"))


def _directive_reason(token):
    if token.start[0] == 1 and token.string.startswith("#!"):
        return "shebang"
    if token.start[0] <= 2 and _CODING.search(token.string):
        return "encoding_cookie"
    if token.start[0] <= 2 and token.string.startswith("# -*-"):
        return "editor_directive"
    if _DIRECTIVE.search(token.string):
        return "functional_directive"
    return None


def _blank(text):
    return "".join(character if character in "\r\n" else " " for character in text)


def _pass(text):
    match = re.search(r"[\r\n]", text)
    end = match.start() if match else len(text)
    return "pass" + " " * max(0, end - 4) + _blank(text[end:])


def analyze_source(raw, path):
    _need(type(path) is str and bool(path) and "\x00" not in path, "Source path must be a nonempty string")
    encoding, original, normalized, positions, offsets, starts = _source(raw)
    tree = _parse(normalized, path)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(normalized).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        raise SourceExtractionError("Invalid Python source tokens") from error

    def character_position(line, column, utf8=False):
        _need(type(line) is int and 1 <= line <= len(starts), "Invalid source line")
        beginning = starts[line - 1]
        ending = starts[line] if line < len(starts) else len(normalized)
        text = normalized[beginning:ending]
        if utf8:
            try:
                column = len(text.encode("utf-8")[:column].decode("utf-8"))
            except UnicodeError as error:
                raise SourceExtractionError("AST column is not a Unicode boundary") from error
        _need(0 <= column <= len(text), "Invalid source column")
        return beginning + column

    def location(position):
        line = bisect.bisect_right(starts, position)
        return line, position - starts[line - 1]

    parents = {}
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for index, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        parents[id(child)] = (parent, field, index)
            elif isinstance(value, ast.AST):
                parents[id(value)] = (parent, field, None)

    def symbol_for(node):
        names = []
        current = node
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(current.name)
            entry = parents.get(id(current))
            current = entry[0] if entry else None
        return ".".join(reversed(names)) or "__module__"

    scopes = [
        (node.lineno, node.end_lineno, node.col_offset, symbol_for(node))
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    def comment_symbol(line):
        candidates = [scope for scope in scopes if scope[0] <= line <= scope[1]]
        return min(candidates, key=lambda scope: (scope[1] - scope[0], -scope[2]))[3] if candidates else "__module__"

    comments = []
    for token in tokens:
        if token.type == tokenize.COMMENT:
            start = character_position(*token.start)
            end = character_position(*token.end)
            comments.append({"token": token, "start": start, "end": end, "reason": _directive_reason(token)})
    group = []
    for comment in comments + [None]:
        if comment is not None and (not group or normalized[group[-1]["end"]:comment["start"]].isspace()):
            group.append(comment)
            continue
        if group and any(_LEGAL.search(item["token"].string) for item in group):
            for item in group:
                item["reason"] = "legal_notice"
        group = [] if comment is None else [comment]

    strings = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        parent, field, index = parents[id(node)]
        kind = "standalone_string"
        if field == "body" and index == 0:
            if isinstance(parent, ast.Module):
                kind = "module_docstring"
            elif isinstance(parent, ast.ClassDef):
                kind = "class_docstring"
            elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function_docstring"
        start = character_position(node.lineno, node.col_offset, utf8=True)
        end = character_position(node.end_lineno, node.end_col_offset, utf8=True)
        reason = "legal_notice" if _LEGAL.search(node.value.value) else None
        inside = [comment for comment in comments if start <= comment["start"] and comment["end"] <= end]
        if not reason and any(comment["reason"] for comment in inside):
            reason = "contains_preserved_comment"
        if kind == "module_docstring" and reason is None:
            suffix = end
            while suffix < len(normalized) and normalized[suffix] in " \t\f":
                suffix += 1
            if suffix < len(normalized) and normalized[suffix] == ";":
                suffix += 1
                while suffix < len(normalized) and normalized[suffix] in " \t\f":
                    suffix += 1
                end = suffix
        strings.append({
            "start": start, "end": end, "kind": kind, "symbol": symbol_for(node),
            "text": node.value.value, "reason": reason,
        })

    entries = list(strings)
    for comment in comments:
        if any(item["start"] <= comment["start"] and comment["end"] <= item["end"] for item in strings):
            continue
        entries.append({
            "start": comment["start"], "end": comment["end"], "kind": "comment",
            "symbol": comment_symbol(comment["token"].start[0]),
            "text": comment["token"].string, "reason": comment["reason"],
        })
    entries.sort(key=lambda item: (item["start"], item["end"]))
    blocks = []
    edits = []
    last_end = -1
    for ordinal, entry in enumerate(entries):
        start, end = entry["start"], entry["end"]
        _need(start >= last_end, "Documentation spans overlap")
        last_end = end
        original_start, original_end = positions[start], positions[end]
        byte_start, byte_end = offsets[original_start], offsets[original_end]
        value = raw[byte_start:byte_end]
        start_line, start_column = location(start)
        end_line, end_column = location(end)
        blocks.append({
            "ordinal": ordinal, "kind": entry["kind"], "symbol": entry["symbol"],
            "start_line": start_line, "start_col": start_column,
            "end_line": end_line, "end_col": end_column,
            "start_byte": byte_start, "end_byte": byte_end,
            "raw_bytes": value, "sha256": _sha(value), "text": entry["text"],
            "preserve_reason": entry["reason"],
        })
        if entry["reason"] is None:
            segment = original[original_start:original_end]
            if entry["kind"] == "comment":
                replacement = _blank(segment)
            elif entry["kind"] == "module_docstring":
                replacement = _blank(segment)
                if normalized[start:end].rstrip().endswith(";"):
                    head = max(replacement.rfind("\n"), replacement.rfind("\r")) + 1
                    replacement = replacement[:head]
            else:
                replacement = _pass(segment)
            edits.append((original_start, original_end, replacement))
    stripped = original
    for start, end, replacement in reversed(edits):
        stripped = stripped[:start] + replacement + stripped[end:]
    try:
        stripped_bytes = stripped.encode(encoding)
    except (UnicodeError, LookupError) as error:
        raise SourceExtractionError("Stripped source cannot use the original encoding") from error
    _need(re.findall(r"\r\n|\r|\n", original) == re.findall(r"\r\n|\r|\n", stripped), "Line endings changed")
    _, _, stripped_normalized, _, _, _ = _source(stripped_bytes)
    original_semantic = _semantic(_parse(normalized, path))
    stripped_semantic = _semantic(_parse(stripped_normalized, path))
    _need(original_semantic == stripped_semantic, "Executable source AST changed")
    references = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id == "__doc__") or (
                isinstance(node, ast.Attribute) and node.attr == "__doc__"):
            references.append({"line": node.lineno, "symbol": symbol_for(node)})
    return {
        "path": path, "encoding": encoding, "original_bytes": raw,
        "original_sha256": _sha(raw), "stripped_bytes": stripped_bytes,
        "stripped_sha256": _sha(stripped_bytes), "semantic_sha256": original_semantic,
        "stripped_semantic_sha256": stripped_semantic,
        "original_line_count": normalized.count("\n") + 1,
        "stripped_line_count": stripped_normalized.count("\n") + 1,
        "runtime_doc_references": references, "blocks": blocks,
    }
