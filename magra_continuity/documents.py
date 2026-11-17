                                                                             
import hashlib
import re

from .config import ContinuityError


def document_parts(raw):
    if not isinstance(raw, bytes) or len(raw) > 16_000_000:
        raise ContinuityError("bounded_original_document_bytes_required")
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise ContinuityError("document_must_be_utf8") from None
    headings = []
    byte_offset = char_offset = 0
    fence = None
    for line in text.splitlines(keepends=True):
        plain = line.rstrip("\r\n")
        visible = plain.lstrip("\ufeff") if char_offset == 0 else plain
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", visible)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1] and not marker[2].strip():
                fence = None
        elif marker:
            fence = (marker[1][0], len(marker[1]))
        else:
            heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$", visible)
            if heading:
                label = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
                headings.append((byte_offset, char_offset, len(heading[1]), label))
        byte_offset += len(line.encode("utf-8"))
        char_offset += len(line)
    starts = headings if headings and headings[0][0] == 0 else [(0, 0, 0, "")] + headings
    sections = []
    for ordinal, (start_byte, start_character, level, heading) in enumerate(starts):
        end_byte, end_character = starts[ordinal+1][:2] if ordinal+1 < len(starts) else (len(raw), len(text))
        sections.append(dict(ordinal=ordinal, heading=heading, level=level, start_byte=start_byte,
                             end_byte=end_byte, start_character=start_character, end_character=end_character))
    return text, hashlib.sha256(raw).hexdigest(), sections
