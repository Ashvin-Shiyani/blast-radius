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
        self.inst: dict[str, int] = doc.get("inst", {})
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

    def inst_tests(self, qualname: str) -> set[int] | None:
        """Tests that instantiated the class (or a subclass); None if the map has no class data."""
        if not self.inst:
            return None
        k = self.inst.get(qualname)
        return set(self.tset(k)) if k is not None else set()

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
    # (path, kind, lo, hi): every judged change; kind "body"/"lambda" carry the runtime line
    # range whose executors were selected; any other kind (import, signature, data, ...) may
    # act anywhere its module is loaded
    changes: list[tuple[str, str, int, int]] = field(default_factory=list)
    # True when candidates were added by anything other than a judged line of an existing
    # python file (new/deleted/renamed/non-compiling modules, data files)
    opaque: bool = False
    # the same, for python files only (a changed data file does not set it)
    code_opaque: bool = False
    data_cands: set[int] = field(default_factory=set)  # readers of changed data files


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


_REFS = None
COMMON_LIMIT = 400


def _refs():
    global _REFS
    if _REFS is None:
        path = os.environ.get("BR_REFS") or os.path.join(os.path.dirname(os.environ.get("BR_MAP", "/app/map/blast.map")), "refs.json")
        try:
            from refindex import RefIndex

            _REFS = RefIndex(path)
        except Exception:  # noqa: BLE001
            _REFS = False
    return _REFS or None


def _module_of(path: str) -> str:
    mod = path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")
    return mod[: -len(".__init__")] if mod.endswith(".__init__") else mod


def _import_time_users(f: str, ln: int, M: "Map", R) -> set[int]:
    """Tests affected through an import-time line of file f: users of the class whose body
    holds it (subclass users come with the MRO records), else everyone who imports it."""
    q = R.class_at(f, ln)
    if q is not None and not f.startswith("tests/"):
        users = M.inst_tests(f"{_module_of(f)}:{q}")
        if users:
            return users | M.rt_lines(f, ln, ln)
    return M.imp_lines(f, ln, ln) | M.rt_lines(f, ln, ln)


def _users_of_name(name: str, M: "Map", R, attr_only_import: bool, skip: tuple[str, int, int]) -> set[int] | None:
    """Tests that run code reading attributes through `name` (a class): runtime lines with
    `name.` and import-time lines with `name.`, the latter credited to the users of the
    class whose body holds them. None if the name is too common."""
    if R.count(name) > COMMON_LIMIT:
        return None
    users: set[int] = set()
    sf, s0, s1 = skip
    for f, ln, flags in R.entries(name):
        if f == sf and s0 <= ln <= s1:
            continue
        if not (flags & 1) or (flags & 4):
            continue  # a mention (dict key, base list, isinstance, a word in a string) reads nothing
        if M.rt_lines(f, ln, ln):
            users |= M.rt_lines(f, ln, ln)
        if M.imp_lines(f, ln, ln):
            users |= _import_time_users(f, ln, M, R)
    return users


def _readers(names: set[str], M: "Map", R, skip: tuple[str, int, int]) -> set[int] | None:
    """Tests that run code reading any of `names` (assignments excluded); None if unsafe.
    Import-time reads inside another class's body are credited to that class's users."""
    users: set[int] = set()
    sf, s0, s1 = skip
    for n in names:
        if (n.startswith("__") and n.endswith("__")) or R.count(n) > COMMON_LIMIT:
            return None
        for f, ln, flags in R.entries(n):
            if (f == sf and s0 <= ln <= s1) or (flags & 2):
                continue
            users |= M.rt_lines(f, ln, ln)
            if M.imp_lines(f, ln, ln):
                users |= _import_time_users(f, ln, M, R)
    return users


def _assigned_names(node) -> set[str]:
    names: set[str] = set()

    def walk(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, ast.Attribute):
            names.add(t.attr)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                walk(e)
        elif isinstance(t, ast.Starred):
            walk(t.value)

    if isinstance(node, ast.Assign):
        for t in node.targets:
            walk(t)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        walk(node.target)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            names.add(a.asname or a.name.split(".")[0])
    return names


def _refine_import_time(path: str, ln: int, st: tuple[int, int], sc: FileScopes, M: "Map",
                        importers: set[int]) -> tuple[set[int], str]:
    """Narrow 'everyone who imports this' to the tests that can read what the statement
    defines: users of the enclosing class, or readers of the module-level names."""
    R = _refs()
    if R is None or not importers:
        return importers, "import"
    chain = sc.class_chain(ln)
    if chain:
        if path.startswith("tests/"):
            # test classes are built by unittest (stdlib __init__): no instantiation records
            return importers, "import/test-class"
        users = M.inst_tests(f"{_module_of(path)}:{'.'.join(c[2] for c in chain)}")
        if not users:
            # never seen instantiated (stdlib-built: Enum, NamedTuple, ...): cannot narrow by users
            return importers, "import/no-instances"
        outer = chain[0]
        refs = _users_of_name(outer[2], M, R, True, (path, outer[0], outer[1]))
        if refs is None:
            return importers, "import"
        users |= refs
        refined = importers & users
        names = _assigned_names(sc.stmt_node_of(ln))
        if names:
            readers = _readers(names, M, R, (path, st[0], st[1]))
            if readers is not None:
                refined &= readers
                return refined, f"import/class {'.'.join(c[2] for c in chain)}/attr {sorted(names)[:2]}"
        return refined, f"import/class {'.'.join(c[2] for c in chain)}"
    node = sc.stmt_node_of(ln)
    names = _assigned_names(node)
    if not names or any(n.startswith("__") and n.endswith("__") for n in names):
        return importers, "import"
    users = _readers(names, M, R, (path, st[0], st[1]))
    if users is None:
        return importers, "import"
    return importers & users, f"import/names {sorted(names)[:3]}"


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def docstring_only(old_tok, new_tok, old_src: str, new_src: str) -> bool:
    """True if the only differences are inside docstrings (no runtime reader of docstring
    text exists in sqlglot or its tests)."""
    import tokenize as _tk
    if old_tok is None or new_tok is None or len(old_tok) != len(new_tok):
        return False
    if any(a != b and not (a[0] == b[0] == _tk.STRING) for a, b in zip(old_tok, new_tok)):
        return False
    try:
        a = ast.dump(_strip_docstrings(ast.parse(old_src)))
        b = ast.dump(_strip_docstrings(ast.parse(new_src)))
    except SyntaxError:
        return False
    return a == b


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
    importers = hit | M.imp_lines(path, *st) | M.rt_lines(path, *st)
    refined, kind = _refine_import_time(path, ln, st, sc, M, importers)
    # code inside the statement that runs later (lambdas in a class-level dict) stays in
    return refined | M.rt_lines(path, *st), kind


def replacement_targets(fc: FileChange) -> list[tuple[int, int, str, str]]:
    """(old line, new line, old text, new text) for each 1:1 line replacement in the patch."""
    out = []
    for h in fc.hunks:
        old_no, new_no = h.old_start, h.new_start
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
                if j - i == 1 and k - j == 1:
                    out.append((old_no, new_no, lines[i][1:], lines[j][1:]))
                old_no += j - i
                new_no += k - j
                i = k
            else:
                old_no += 1
                new_no += 1
                i += 1
    return out


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


def _runtime_range(path: str, ln: int, span, sc: FileScopes, kind: str) -> tuple[int, int]:
    """Line range whose runtime executors can observe a body/lambda change at `ln`."""
    if kind == "lambda" and span is not None:
        lam = sc.lambda_body_at(ln, *span)
        if lam is not None:
            return lam
    if kind == "body":
        f = sc.func_of(ln)
        if f is not None:
            return sc.stmt_within(ln, f[2], f[1])
    return (ln, ln)


def analyze_python(fc: FileChange, repo: str, M: Map, A: Analysis) -> None:
    path = fc.old_path or fc.path
    old_file = os.path.join(repo, path)
    base = os.path.basename(fc.path)
    if base in PYTEST_CONFIG:
        A.broad = True
        A.notes.append(f"{fc.path}: pytest config")
        return
    if fc.is_new or not os.path.isfile(old_file):
        hit: set[int] = set()
        R = _refs()
        if R is not None:
            for n in ("iter_modules", "walk_packages", "listdir", "scandir", "glob", "iglob", "rglob"):
                for f, ln in R.lines(n):
                    hit |= M.rt_lines(f, ln, ln) | M.imp_lines(f, ln, ln)
        A.cands |= hit
        A.opaque = True
        A.notes.append(f"{fc.path}: new python file: {len(hit)} tests that list packages/directories")
        return
    users = M.file_users(path)
    if fc.is_deleted or fc.is_rename or (fc.new_path and fc.old_path and fc.new_path != fc.old_path):
        A.cands |= users
        A.opaque = True
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
    new_tok = significant_tokens(new_src)
    head_changed = any(ln <= 2 for ln in fc.removed) or any(a < 2 for a in fc.added_after)
    if head_changed:
        try:  # a coding cookie on line 1-2 is honoured when the file is imported
            compile(new_src.encode("utf-8", "surrogateescape"), path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, LookupError, UnicodeDecodeError):
            A.cands |= users
            A.import_break |= users
            A.opaque = True
            A.notes.append(f"{fc.path}: file no longer decodes/compiles -> {len(users)} users")
            return
    if old_tok is not None and old_tok == new_tok:
        A.notes.append(f"{fc.path}: comments/blank lines only")
        return
    if docstring_only(old_tok, new_tok, old_src, new_src):
        A.notes.append(f"{fc.path}: comments/blank lines only (docstrings)")
        return
    try:
        compile(new_src, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        A.cands |= users
        A.import_break |= users
        A.opaque = True
        A.notes.append(f"{fc.path}: patched file does not compile -> {len(users)} users")
        return
    if len(fc.removed) + len(fc.added_text) > 200:
        # a sweeping edit: line-level precision is not worth the time; everyone who loads it
        A.cands |= users
        A.notes.append(f"{fc.path}: {len(fc.removed) + len(fc.added_text)} changed lines -> all {len(users)} users")
        A.changes.append((path, "import", 1, 1))
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
        A.changes.append((path, kind.split("/")[0], *_runtime_range(path, ln, span, sc, kind)))
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
                of2 = sc.func_of(origin[f[0] - 1])
                A.changes.append((path, "body", of2[2], of2[1]) if of2 else (path, "import", 0, 0))
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
                        hit, _ = _refine_import_time(path, oc[0], (oc[0], oc[1]), sc, M, hit)
                if not hit:
                    hit = set(users)
                A.cands |= hit
                A.notes.append(f"{path}:+{nln} import-time insert -> {len(hit)}")
                A.changes.append((path, "import", 0, 0))


def _touches_pytest_section(fc: FileChange, repo: str) -> bool:
    """True if the change edits pytest configuration inside an ini/toml/cfg file."""
    base = os.path.basename(fc.path)
    if base in ("pytest.ini", "conftest.py") or fc.is_new or fc.is_deleted:
        return True
    try:
        old = open(os.path.join(repo, fc.old_path or fc.path), encoding="utf-8").read().split("\n")
    except OSError:
        return True
    headers = [(i + 1, l.strip()) for i, l in enumerate(old) if l.strip().startswith("[")]
    def section(line: int) -> str:
        cur = ""
        for ln, h in headers:
            if ln <= line:
                cur = h
        return cur
    lines = set(fc.removed) | set(fc.added_after) | {a + 1 for a in fc.added_after}
    if any("pytest" in t for t in fc.added_text + fc.removed_text):
        return True
    return any("pytest" in section(ln) for ln in lines if ln >= 1)


def _config_parses(fc: FileChange, repo: str) -> bool:
    """pytest reads these files at start-up: one that no longer parses stops every test."""
    try:
        old = open(os.path.join(repo, fc.old_path or fc.path), encoding="utf-8").read()
        new = apply_hunks_with_origin(old, fc)[0]
        if fc.path.endswith(".toml"):
            import tomllib

            tomllib.loads(new)
        else:
            import configparser

            configparser.ConfigParser(interpolation=None).read_string(new)
        return True
    except Exception:  # noqa: BLE001
        return False


def analyze_data(fc: FileChange, repo: str, M: Map, A: Analysis) -> None:
    base = os.path.basename(fc.path)
    if base in PYTEST_CONFIG and (fc.path.count("/") == 0 or base in ("conftest.py", "pytest.ini")):
        if _touches_pytest_section(fc, repo) or not _config_parses(fc, repo):
            A.broad = True
            A.notes.append(f"{fc.path}: pytest config (or no longer parses)")
            return
        A.notes.append(f"{fc.path}: config outside pytest sections")
    hit: set[int] = set()
    for p in fc.paths:
        hit |= M.openers(p)
        if fc.is_new or not os.path.exists(os.path.join(repo, p)):
            # a new data file is only read by code that lists its directory
            hit |= M.dir_openers(os.path.dirname(p))
    A.cands |= hit
    A.data_cands |= hit
    A.opaque = True
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
            before, A.opaque = A.opaque, False
            analyze_python(fc, repo, M, A)
            A.code_opaque = A.code_opaque or A.opaque
            A.opaque = A.opaque or before
        else:
            analyze_data(fc, repo, M, A)
    return A



def import_check_modules(A: "Analysis", repo: str) -> list[str]:
    """Modules whose import could now raise: those with import-time or signature changes,
    plus sqlglot modules that read the changed names or class at import (subclass bodies).
    Each file is parsed once and at most 60 changed statements are examined (huge patches
    still get their own modules checked)."""
    mods: list[str] = []
    R = _refs()
    scopes: dict[str, FileScopes | None] = {}
    names: set[str] = set()
    examined = 0
    for path, kind, lo, hi in A.changes:
        if kind in ("body", "lambda"):
            continue
        if path.endswith(".py"):
            m = _module_of(path)
            if m not in mods:
                mods.append(m)
        if R is None or lo <= 0 or examined >= 60:
            continue
        examined += 1
        if path not in scopes:
            try:
                scopes[path] = FileScopes(open(os.path.join(repo, path), encoding="utf-8").read())
            except Exception:  # noqa: BLE001
                scopes[path] = None
        sc = scopes[path]
        if sc is None:
            continue
        node = sc.stmt_node_of(lo)
        names |= _assigned_names(node) if node is not None else set()
        chain = sc.class_chain(lo)
        if chain:
            names.add(chain[0][2])
    for n in sorted(names):
        if R.count(n) > COMMON_LIMIT:
            continue
        for f, ln, flags in R.entries(n):
            if f.startswith("sqlglot/") and not (flags & 4):
                m = _module_of(f)
                if m not in mods:
                    mods.append(m)
    return mods[:80]
