"""Apply one FileChange's hunks to the original text, checking every context line."""

from __future__ import annotations

from patchparse import FileChange


class PatchError(Exception):
    pass


def apply_hunks(old_text: str, fc: FileChange) -> str:
    return apply_hunks_with_origin(old_text, fc)[0]


def apply_hunks_with_origin(old_text: str, fc: FileChange) -> tuple[str, list[int | None]]:
    """Patched text, and for each new line (0-based) the 1-based old line it came from
    (None for an added line)."""
    old = old_text.split("\n")
    out: list[str] = []
    origin: list[int | None] = []
    pos = 0  # next old line index (0-based) not yet copied
    for h in fc.hunks:
        start = h.old_start - 1 if h.old_len > 0 else h.old_start  # 0-based index of first old line
        if start < pos or start > len(old):
            raise PatchError(f"hunk at {h.old_start} out of order or range")
        out.extend(old[pos:start])
        origin.extend(range(pos + 1, start + 1))
        pos = start
        for line in h.lines:
            tag, body = (line[:1], line[1:]) if line else (" ", "")
            if tag == " ":
                if pos >= len(old) or old[pos] != body:
                    raise PatchError(f"context mismatch at old line {pos + 1}")
                out.append(body)
                origin.append(pos + 1)
                pos += 1
            elif tag == "-":
                if pos >= len(old) or old[pos] != body:
                    raise PatchError(f"removed line mismatch at old line {pos + 1}")
                pos += 1
            elif tag == "+":
                out.append(body)
                origin.append(None)
    out.extend(old[pos:])
    origin.extend(range(pos + 1, len(old) + 1))
    return "\n".join(out), origin
