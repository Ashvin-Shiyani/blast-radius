"""Static candidate selection: which tests can a patch possibly affect?

Recall comes first: a candidate set must contain every test that can fail. Dynamic
verification (runner.py) narrows candidates down to the ones that actually fail.
"""

from __future__ import annotations

import ast
import json
import os
import zlib
from dataclasses import dataclass, field

from applypatch import PatchError, apply_hunks_with_origin
from patchparse import FileChange, parse_patch
from scopes import FileScopes, significant_tokens

# Files that change how pytest itself collects or runs tests.
PYTEST_CONFIG = {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"}


class Map:
    def __init__(self, path: str, tests: list[dict]):
        with open(path, "rb") as fh:
            doc = json.loads(zlib.decompress(fh.read()))
        num = {t["id"]: t["number"] for t in tests}
        self.to_num = [num.get(i) for i in doc["ids"]]
        self._sets = doc["sets"]
        self._cache: dict[int, frozenset[int]] = {}
        self.rt: dict[str, dict[str, int]] = doc["rt"]
        self.imp: dict[str, dict[str, int]] = doc["imp"]
        self.opens: dict[str, int] = doc["opens"]
        self.all_tests = frozenset(t["number"] for t in tests)
        mapped = {n for n in self.to_num if n is not None}
        # tests in tests.json the map has never seen: we know nothing about them
        self.unmapped = frozenset(self.all_tests - mapped)

    def tset(self, k: int) -> frozenset[int]:
        s = self._cache.get(k)
        if s is None:
            acc, out = 0, set()
            for d in self._sets[k]:
                acc += d
                n = self.to_num[acc]
                if n is not None:
                    out.add(n)
            s = self._cache[k] = frozenset(out)
        return s

    def _lines(self, index: dict, f: str, lo: int, hi: int) -> set[int]:
        out: set[int] = set()
        d = index.get(f)
        if not d:
            return out
        for ln in range(lo, hi + 1):
            k = d.get(str(ln))
            if k is not None:
                out |= self.tset(k)
        return out

    def rt_lines(self, f: str, lo: int, hi: int) -> set[int]:
        return self._lines(self.rt, f, lo, hi)

    def imp_lines(self, f: str, lo: int, hi: int) -> set[int]:
        return self._lines(self.imp, f, lo, hi)

    def file_users(self, f: str) -> set[int]:
        out: set[int] = set()
        for index in (self.rt, self.imp):
            for k in index.get(f, {}).values():
                out |= self.tset(k)
        return out

    def openers(self, f: str) -> set[int]:
        k = self.opens.get(f)
        return set(self.tset(k)) if k is not None else set()

    def dir_openers(self, d: str) -> set[int]:
        prefix = d.rstrip("/") + "/"
        out: set[int] = set()
        for f, k in self.opens.items():
            if f.startswith(prefix) and "/" not in f[len(prefix):]:
                out |= self.tset(k)
        return out


@dataclass
class Analysis:
    cands: set[int] = field(default_factory=set)
    broad: bool = False  # nothing reliable is known: consider every test
    import_break: set[int] = field(default_factory=set)  # tests whose imports fail outright
    notes: list[str] = field(default_factory=list)
    new_sources: dict[str, str] = field(default_factory=dict)  # path -> patched text (.py)
    files: list[FileChange] = field(default_factory=list)


def _changed_span(old: str, new: str) -> tuple[int, int]:
    """Column span [a, b) of `old` that differs from `new` (common prefix/suffix removed)."""
    a = 0
    while a < len(old) and a < len(new) and old[a] == new[a]:
        a += 1
    b_old, b_new = len(old), len(new)
    while b_old > a and b_new > a and old[b_old - 1] == new[b_new - 1]:
        b_old -= 1
        b_new -= 1
    return a, max(b_old, a + 1)


def _lambda_body_contains(tree: ast.AST, line: int, a: int, b: int) -> tuple[int, int] | None:
    """If [a, b) on `line` lies inside some lambda's body, return that body's line range."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            body = node.body
            if body.lineno == line == body.end_lineno and body.col_offset <= a and b <= body.end_col_offset:
                span = (body.lineno, body.end_lineno)
                if best is None or (body.end_col_offset - body.col_offset) < best[2]:
                    best = (span[0], span[1], body.end_col_offset - body.col_offset)
    return (best[0], best[1]) if best else None


def _replacement_pairs(fc: FileChange) -> dict[int, str]:
    """Old line number -> new text, for hunks that replace exactly one line with one line."""
    pairs: dict[int, str] = {}
    for h in fc.hunks:
        old_no = h.old_start
        i = 0
        lines = h.lines
        while i < len(lines):
            line = lines[i]
            if line.startswith("-"):
                j = i
                while j < len(lines) and lines[j].startswith("-"):
                    j += 1
                k = j
                while k < len(lines) and lines[k].startswith("+"):
                    k += 1
                if j - i == 1 and k - j == 1:
                    pairs[old_no] = lines[j][1:]
                old_no += j - i
                i = k
            elif line.startswith("+"):
                i += 1
            else:
                old_no += 1
                i += 1
    return pairs


def _is_code_line(text: str) -> bool:
    t = text.strip()
    return bool(t) and not t.startswith("#")


def _rule_for_old_line(path: str, ln: int, span, sc: FileScopes, M: Map) -> tuple[set[int], str]:
    """Tests that can observe a change on line `ln` of the ORIGINAL file."""
    hit = M.rt_lines(path, ln, ln) | M.imp_lines(path, ln, ln)
    if span is not None:
        lam = sc.lambda_body_at(ln, *span)
        if lam is not None:
            return hit | M.rt_lines(path, lam[0], lam[1]), "lambda"
    f = sc.func_of(ln)
    if f is not None and f[2] <= ln <= f[1]:
        lo, hi = sc.stmt_within(ln, f[2], f[1])
        return hit | M.rt_lines(path, lo, hi), "body"
    if f is not None:
        # decorators, def line, signature defaults: observed by everyone who calls it
        return hit | M.rt_lines(path, f[2], f[1]), "signature"
    st = sc.stmt_of(ln) or (ln, ln)
    return hit | M.imp_lines(path, *st) | M.rt_lines(path, *st), "import"


def _insert_new_lines(fc: FileChange) -> set[int]:
    """New-file line numbers of '+' lines that are not a 1:1 stand-in for a '-' line:
    pure insertions, and the surplus of a block that adds more lines than it removes."""
    out: set[int] = set()
    for h in fc.hunks:
        new_no = h.new_start
        lines = h.lines
        i = 0
        while i < len(lines):
            if lines[i].startswith("-") or lines[i].startswith("+"):
                j = i
                while j < len(lines) and lines[j].startswith("-"):
                    j += 1
                k = j
                while k < len(lines) and lines[k].startswith("+"):
                    k += 1
                removed, added = j - i, k - j
                for t in range(added):
                    if t >= removed:
                        out.add(new_no + t)
                new_no += added
                i = k
            else:
                new_no += 1
                i += 1
    return out


def analyze_python(fc: FileChange, repo: str, M: Map, A: Analysis) -> None:
    path = fc.old_path or fc.path
    old_file = os.path.join(repo, path)
    base = os.path.basename(fc.path)
    if base in PYTEST_CONFIG:
        A.broad = True
        A.notes.append(f"{fc.path}: pytest config")
        return
    if fc.is_new or not os.path.isfile(old_file):
        A.notes.append(f"{fc.path}: new python file, imported by nothing at the base commit")
        return
    users = M.file_users(path)
    if fc.is_deleted or fc.is_rename or (fc.new_path and fc.old_path and fc.new_path != fc.old_path):
        A.cands |= users
        A.notes.append(f"{fc.path}: deleted/renamed module -> all {len(users)} users")
        return
    old_src = open(old_file, encoding="utf-8").read()
    try:
        new_src, origin = apply_hunks_with_origin(old_src, fc)
    except PatchError as exc:
        A.broad = True
        A.notes.append(f"{fc.path}: patch does not apply ({exc})")
        return
    A.new_sources[path] = new_src
    old_tok = significant_tokens(old_src)
    if old_tok is not None and old_tok == significant_tokens(new_src):
        A.notes.append(f"{fc.path}: comments/blank lines only")
        return
    try:
        compile(new_src, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        A.cands |= users
        A.import_break |= users
        A.notes.append(f"{fc.path}: patched file does not compile -> {len(users)} users")
        return
    sc = FileScopes(old_src)
    old_lines = old_src.split("\n")
    pairs = _replacement_pairs(fc)
    # 1. modified and removed lines, judged in the original file
    for ln in sorted(set(fc.removed)):
        span = _changed_span(old_lines[ln - 1], pairs[ln]) if ln in pairs else None
        hit, kind = _rule_for_old_line(path, ln, span, sc, M)
        A.cands |= hit
        A.notes.append(f"{path}:{ln} {kind} -> {len(hit)}")
    # 2. inserted lines, judged in the patched file and mapped back to the original
    new_lines = new_src.split("\n")
    surplus = _insert_new_lines(fc)
    inserted = [
        i for i, o in enumerate(origin)
        if o is None and (i + 1) in surplus and _is_code_line(new_lines[i])
    ]
    if inserted:
        nsc = FileScopes(new_src)
        for i in inserted:
            nln = i + 1
            f = nsc.func_of(nln)
            if f is not None and f[2] <= nln <= f[1] and origin[f[0] - 1] is not None:
                # inside an existing def body: the new statement runs where its
                # neighbours run; use the nearest original code lines around it
                prev_old = next_old = None
                for j in range(i - 1, f[2] - 2, -1):
                    if origin[j] is not None and _is_code_line(new_lines[j]):
                        prev_old = origin[j]
                        break
                for j in range(i + 1, f[1]):
                    if origin[j] is not None and _is_code_line(new_lines[j]):
                        next_old = origin[j]
                        break
                hit: set[int] = set()
                for o in (prev_old, next_old):
                    if o is not None:
                        h, _ = _rule_for_old_line(path, o, None, sc, M)
                        hit |= h
                if prev_old is None:
                    # first statement of the body: everyone who enters the function
                    of = sc.func_of(origin[f[0] - 1])
                    if of is not None:
                        hit |= M.rt_lines(path, of[2], of[1])
                A.cands |= hit
                A.notes.append(f"{path}:+{nln} body insert -> {len(hit)}")
            else:
                # a new def, a class attribute or a module statement: runs at import and can
                # override anything inherited, so everyone who imports the enclosing class
                cls = None
                for c in nsc.classes:
                    if c[0] <= nln <= c[1] and (cls is None or c[0] >= cls[0]):
                        cls = c
                hit = set()
                if cls is not None and origin[cls[0] - 1] is not None:
                    o0 = origin[cls[0] - 1]
                    oc = None
                    for c in sc.classes:
                        if c[0] == o0:
                            oc = c
                    if oc is not None:
                        hit = M.imp_lines(path, oc[0], oc[1]) | M.rt_lines(path, oc[0], oc[1])
                if not hit:
                    hit = set(users)
                A.cands |= hit
                A.notes.append(f"{path}:+{nln} import-time insert -> {len(hit)}")


def analyze_data(fc: FileChange, repo: str, M: Map, A: Analysis) -> None:
    base = os.path.basename(fc.path)
    if base in PYTEST_CONFIG:
        A.broad = True
        A.notes.append(f"{fc.path}: pytest config")
        return
    hit: set[int] = set()
    for p in fc.paths:
        hit |= M.openers(p)
        if fc.is_new or not os.path.exists(os.path.join(repo, p)):
            # a new data file is only read by code that lists its directory
            hit |= M.dir_openers(os.path.dirname(p))
    A.cands |= hit
    A.notes.append(f"{fc.path}: data file opened by {len(hit)}")


def analyze_patch(text: str, repo: str, M: Map) -> Analysis:
    A = Analysis()
    try:
        files = parse_patch(text)
    except Exception as exc:  # noqa: BLE001
        A.broad = True
        A.notes.append(f"unparseable patch: {exc!r}")
        return A
    if not files and text.strip():
        A.broad = True
        A.notes.append("patch with no file sections")
        return A
    A.files = files
    for fc in files:
        if fc.is_binary:
            A.broad = True
            A.notes.append(f"{fc.path}: binary change")
            continue
        if fc.path.endswith(".py") or (fc.old_path or "").endswith(".py"):
            analyze_python(fc, repo, M, A)
        else:
            analyze_data(fc, repo, M, A)
    return A
