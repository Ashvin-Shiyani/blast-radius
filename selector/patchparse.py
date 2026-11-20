"""Minimal unified-diff parser for sqlglot patches.

parse_patch(text) -> list of FileChange with:
  old_path, new_path (None for /dev/null), is_new, is_deleted, is_rename, is_binary
  hunks: list of Hunk(old_start, old_len, new_start, new_len, lines)
  removed: old-file line numbers of '-' lines
  added_after: old-file line number after which each '+' line lands (0 = before line 1)
  added_text / removed_text: the raw text of the + and - lines
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileChange:
    old_path: str | None = None
    new_path: str | None = None
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False
    is_binary: bool = False
    mode_change: bool = False
    hunks: list[Hunk] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    added_after: list[int] = field(default_factory=list)
    removed_text: list[str] = field(default_factory=list)
    added_text: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def paths(self) -> set[str]:
        return {p for p in (self.old_path, self.new_path) if p}


def _strip_prefix(p: str) -> str | None:
    p = p.strip()
    if "\t" in p:
        p = p.split("\t", 1)[0]
    if p == "/dev/null":
        return None
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    return p


def parse_patch(text: str) -> list[FileChange]:
    files: list[FileChange] = []
    cur: FileChange | None = None
    hunk: Hunk | None = None
    old_no = new_no = 0
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FileChange()
            files.append(cur)
            hunk = None
            m = re.match(r"^diff --git (\S+) (\S+)$", line)
            if m:
                cur.old_path = _strip_prefix(m.group(1))
                cur.new_path = _strip_prefix(m.group(2))
            i += 1
            continue
        if hunk is None or not (line[:1] in (" ", "+", "-", "\\") or line == ""):
            # header territory
            if cur is None and (line.startswith("--- ") or line.startswith("+++ ")):
                cur = FileChange()
                files.append(cur)
            if cur is not None:
                if line.startswith("--- "):
                    cur.old_path = _strip_prefix(line[4:])
                    if cur.old_path is None:
                        cur.is_new = True
                    hunk = None
                elif line.startswith("+++ "):
                    cur.new_path = _strip_prefix(line[4:])
                    if cur.new_path is None:
                        cur.is_deleted = True
                    hunk = None
                elif line.startswith("new file mode"):
                    cur.is_new = True
                elif line.startswith("deleted file mode"):
                    cur.is_deleted = True
                elif line.startswith("rename from ") or line.startswith("rename to ") or line.startswith("similarity index"):
                    cur.is_rename = True
                    if line.startswith("rename from "):
                        cur.old_path = line[len("rename from "):].strip()
                    if line.startswith("rename to "):
                        cur.new_path = line[len("rename to "):].strip()
                elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
                    cur.is_binary = True
                elif line.startswith("old mode") or line.startswith("new mode"):
                    cur.mode_change = True
                m = _HUNK.match(line)
                if m:
                    hunk = Hunk(int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1))
                    cur.hunks.append(hunk)
                    old_no, new_no = hunk.old_start, hunk.new_start
            i += 1
            continue
        # inside a hunk
        m = _HUNK.match(line)
        if m:
            hunk = Hunk(int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1))
            cur.hunks.append(hunk)
            old_no, new_no = hunk.old_start, hunk.new_start
            i += 1
            continue
        if line.startswith("diff --git "):
            continue
        if line.startswith("\\"):
            i += 1
            continue
        hunk.lines.append(line)
        if line.startswith("-"):
            cur.removed.append(old_no)
            cur.removed_text.append(line[1:])
            old_no += 1
        elif line.startswith("+"):
            cur.added_after.append(old_no - 1)
            cur.added_text.append(line[1:])
            new_no += 1
        elif line.startswith(" ") or line == "":
            old_no += 1
            new_no += 1
        if old_no >= hunk.old_start + hunk.old_len and new_no >= hunk.new_start + hunk.new_len:
            hunk = None
        i += 1
    return files
