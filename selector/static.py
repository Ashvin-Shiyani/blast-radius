"""Static candidate selection: which tests can a patch possibly affect?

Recall comes first: a candidate set must contain every test that can fail. Dynamic
verification (runner.py) narrows candidates down to the ones that actually fail.
"""

from __future__ import annotations

import ast
import json
import os
import re
import symtable
import zlib
from dataclasses import dataclass, field

from applypatch import PatchError, apply_hunks_with_origin
from patchparse import FileChange, parse_patch
from scopes import FileScopes, significant_tokens

# Files that change how pytest itself collects or runs tests.
PYTEST_CONFIG = {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
                 "pytest.toml", ".pytest.toml"}  # (v8: the last two are read by pytest 9)


class Map:
    def __init__(self, path: str, tests: list[dict]):
        with open(path, "rb") as fh:
            doc = json.loads(zlib.decompress(fh.read()))
        num = {t["id"]: t["number"] for t in tests}
        self.num = num
        self.to_num = [num.get(i) for i in doc["ids"]]
        self._sets = doc["sets"]
        self._cache: dict[int, frozenset[int]] = {}
        self.rt: dict[str, dict[str, int]] = doc["rt"]
        self.imp: dict[str, dict[str, int]] = doc["imp"]
        self.opens: dict[str, int] = doc["opens"]
        self.inst: dict[str, int] = doc.get("inst", {})
        self.all_tests = frozenset(t["number"] for t in tests)
        self.test_ids = frozenset(t["id"] for t in tests)
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


QUIET_LIMIT = 12  # more quiet added lines than this are not examined one by one


def _without(lines: list[str], drop: set[int]) -> str:
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)


def _live_quiet_lines(new_src: str, new_lines: list[str], new_tok, origin: list) -> set[int]:
    """v8. Indices (0-based) of added lines that LOOK like a comment or a blank line but are
    part of the program: a line inside a multi-line string, after a backslash continuation,
    or anywhere else it changes what the parser reads.

    Decided on the program, not on the look of the line: a set of such lines is harmless only
    if the patched file without them has the very same token stream (comments and
    non-logical newlines aside), hence the same syntax tree. Lines that cannot be taken out
    are returned, unless all they change is the text of docstrings and of bare string
    statements (see _strip_bare_strings: without those the two syntax trees are identical)."""
    quiet = [i for i, o in enumerate(origin) if o is None and not _is_code_line(new_lines[i])]
    if not quiet:
        return set()
    if new_tok is None:
        return set(quiet)

    def same_without(drop: set[int]) -> bool:
        tok = significant_tokens(_without(new_lines, drop))
        return tok is not None and tok == new_tok

    if same_without(set(quiet)):
        return set()
    dropped: set[int] = set()
    live: set[int] = set()
    if len(quiet) > QUIET_LIMIT:
        live = set(quiet)
    else:
        for i in quiet:
            if same_without(dropped | {i}):
                dropped.add(i)
            else:
                live.add(i)
    try:
        a = ast.dump(_strip_bare_strings(ast.parse(new_src)))
        b = ast.dump(_strip_bare_strings(ast.parse(_without(new_lines, dropped | live))))
        if a == b:
            return set()  # docstring / bare string text only
    except (SyntaxError, ValueError):
        pass
    return live


def _strip_bare_strings(tree: ast.AST) -> ast.AST:
    """Remove every statement that is nothing but a string constant: a docstring (the
    docstring rule: no test reads docstring text) and a bare string anywhere else in a body
    (an "attribute docstring", a string used as a comment: the compiler emits no code for it)."""
    for node in ast.walk(tree):
        for fld in ("body", "orelse", "finalbody"):
            body = getattr(node, fld, None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                kept = [st for st in body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
                                                  and isinstance(st.value.value, str))]
                if len(kept) != len(body):
                    setattr(node, fld, kept or [ast.Pass()])
    return tree


def _same_but_stmts(old_src: str, new_src: str, old_nodes: list, new_nodes: list) -> bool:
    """True if the two files have the same syntax tree once the given statements (the old
    file's in the old tree, the new file's in the new tree) are emptied: every difference
    between the files lies inside those statements."""

    def key(n) -> tuple:
        return (type(n).__name__, n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)

    try:
        dumps = []
        for src, nodes in ((old_src, old_nodes), (new_src, new_nodes)):
            want = {key(n) for n in nodes}
            tree = ast.parse(src)
            found = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.stmt) and key(node) in want:
                    for f in node._fields:
                        setattr(node, f, [] if isinstance(getattr(node, f, None), list) else None)
                    found += 1
            if found != len(want):
                return False
            dumps.append(ast.dump(tree))
        return dumps[0] == dumps[1]
    except (SyntaxError, ValueError):
        return False


def _same_but_bare_strings(old_src: str, new_src: str) -> bool:
    try:
        return ast.dump(_strip_bare_strings(ast.parse(old_src))) == ast.dump(_strip_bare_strings(ast.parse(new_src)))
    except (SyntaxError, ValueError):
        return False


def _unhosted_new_lines(fc: FileChange, live: set[int]) -> set[int]:
    """v8. New-file line numbers of added lines that no removed CODE line stands for. A
    removed line that looks like a comment or a blank line hosts nothing (judged in the
    original file it selects nobody when it lies between statements), so in each block the
    added code lines (and live quiet lines) beyond the number of removed code lines are
    insertions and are judged as such."""
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
                hosts = sum(1 for t in range(i, j) if _is_code_line(lines[t][1:]))
                rank = 0
                for t in range(j, k):
                    nln = new_no + (t - j)
                    if _is_code_line(lines[t][1:]) or (nln - 1) in live:
                        if rank >= hosts:
                            out.add(nln)
                        rank += 1
                new_no += k - j
                i = k
            else:
                new_no += 1
                i += 1
    return out


def _same_tree_as_bytes(old_src: str, new_src: str) -> bool:
    """The two files parse to the same syntax tree when read as the interpreter reads them
    (from bytes, honouring a coding cookie)."""
    try:
        a = ast.dump(ast.parse(old_src.encode("utf-8", "surrogateescape")))
        b = ast.dump(ast.parse(new_src.encode("utf-8", "surrogateescape")))
    except (SyntaxError, ValueError, LookupError, UnicodeDecodeError):
        return False
    return a == b


_MENTIONS: dict[str, bool] = {}
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", "build", "dist"}


def _tree_mentions(repo: str, word: str) -> bool:
    """True if `word` occurs as a whole word in any python file of the pinned tree: an import
    (also inside try/except), an importlib call, a lazily looked up attribute or dialect name."""
    got = _MENTIONS.get(word)
    if got is None:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])")
        got = False
        for root, dirs, files in os.walk(repo):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(root, name), encoding="utf-8", errors="replace") as fh:
                        if pat.search(fh.read()):
                            got = True
                            break
                except OSError:
                    got = True  # unreadable: assume it does
                    break
            if got:
                break
        _MENTIONS[word] = got
    return got


def _new_module_reach(p: str, repo: str) -> str | None:
    """v8. Why a NEW python file can change what every test imports, or None: it shadows an
    existing module or package, it turns an existing directory into a package, or its name is
    one the pinned tree already asks for (an optional import in try/except, importlib, a lazy
    attribute or dialect lookup, a third-party or stdlib module of that name)."""
    d, b = os.path.split(p)
    if b == "__init__.py":
        if os.path.isfile(os.path.join(repo, d + ".py")):
            return f"shadows module {d}.py"
        if os.path.isdir(os.path.join(repo, d)):
            return f"turns the existing directory {d or '.'} into a package"
        name = os.path.basename(d)
    else:
        stem = p[:-3] if p.endswith(".py") else p
        if os.path.isdir(os.path.join(repo, stem)):
            return f"stands beside the directory {stem}"
        name = os.path.basename(stem)
    if name in ("sitecustomize", "usercustomize"):
        return "is imported at interpreter start"
    if name and _tree_mentions(repo, name):
        return f"has a name the pinned tree mentions ({name}): optional or dynamic import, or shadowing"
    return None


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
        # v9 (F1): a class-creation hook of an enclosing class reads the nested class at import
        # and keeps what it derives on the enclosing class (_Dialect.__new__: QUOTE_START,
        # HEX_START, SUPPORTS_COLUMN_JOIN_MARKS ... from the nested Tokenizer), where code reads
        # it through an instance of the enclosing class without ever instantiating the nested
        # one. Every enclosing class's instantiators (subclasses come with the MRO records) use it.
        for k in range(1, len(chain)):
            users |= M.inst_tests(f"{_module_of(path)}:{'.'.join(c[2] for c in chain[:k])}") or set()
        outer = chain[0]
        refs = _users_of_name(outer[2], M, R, True, (path, outer[0], outer[1]))
        if refs is None:
            return importers, "import"
        users |= refs
        # v9 (F1): ... and so does every class that keeps this one under an attribute or is handed
        # it by a class-creation hook (see _holders)
        held = _holders(outer[2], M, R, (path, outer[0], outer[1]))
        if held is None:
            return importers, "import"
        users |= held
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


_REPO = None
_SRC_LINES: dict[str, list[str]] = {}


def _pinned_line(f: str, ln: int) -> str | None:
    lines = _SRC_LINES.get(f)
    if lines is None:
        try:
            with open(os.path.join(_REPO or os.environ.get("BR_REPO", "/app/sqlglot"), f), encoding="utf-8") as fh:
                lines = fh.read().split("\n")
        except OSError:
            lines = []
        _SRC_LINES[f] = lines
    return lines[ln - 1] if 0 < ln <= len(lines) else None


# defs the interpreter calls while it creates a class (the pinned tree has one metaclass,
# _Dialect, with __new__ only, and two __init_subclass__)
CLASS_HOOKS = ("__new__", "__init_subclass__", "__set_name__", "__prepare__", "__class_getitem__")


def _in_class_hook(f: str, ln: int) -> bool:
    """True if a def above line `ln` that may enclose it (read off the indentation) is a
    class-creation hook; an unreadable file counts as yes."""
    text = _pinned_line(f, ln)
    if text is None:
        return True
    cur = len(text) - len(text.lstrip())
    for j in range(ln - 1, 0, -1):
        t = _SRC_LINES[f][j - 1].lstrip()
        ind = len(_SRC_LINES[f][j - 1]) - len(t)
        if ind < cur and t.startswith(("def ", "async def ")):
            if t.split("(")[0].split()[-1] in CLASS_HOOKS:
                return True
            cur = ind
            if cur == 0:
                break
    return False


def _holders(name: str, M: "Map", R, skip: tuple[str, int, int]) -> set[int] | None:
    """v9 (F1). Tests that can read attributes of the class `name` through another class that
    keeps it, without instantiating it and without spelling its name:
    - a class body that stores it under an attribute (`Parser = BigQueryParser`: the dialect
      reads `parser_class.FUNCTIONS`, `generator_class.TYPE_MAPPING` ... through itself): the
      users of that class;
    - a class-creation hook that is handed it as an object (_Dialect.__new__ takes
      tokens.Tokenizer, Parser, Generator, JSONPathTokenizer as every dialect's defaults and
      derives QUOTE_START, HEX_START ... from them): everyone who runs that line.
    None if a file cannot be read."""
    out: set[int] = set()
    sf, s0, s1 = skip
    pat = re.compile(r"^\s*[A-Za-z_]\w*\s*(?::[^=]*)?=\s*(?:[\w.]+\.)?" + re.escape(name) + r"\s*(?:#.*)?$")
    for f, ln, flags in R.entries(name):
        if flags & 7 or (f == sf and s0 <= ln <= s1):
            continue
        text = _pinned_line(f, ln)
        if text is None:
            return None
        if pat.match(text):
            if R.class_at(f, ln) is not None:
                out |= _import_time_users(f, ln, M, R) | M.rt_lines(f, ln, ln)
        elif M.rt_lines(f, ln, ln) and _in_class_hook(f, ln):
            out |= M.rt_lines(f, ln, ln)
    return out


def _is_nocode(st) -> bool:
    """A statement that compiles to no instruction of its own, so that no coverage map can say
    who ran it: global / nonlocal, a bare constant (docstring), an annotation without a value."""
    return (isinstance(st, (ast.Global, ast.Nonlocal))
            or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))
            or (isinstance(st, ast.AnnAssign) and st.value is None))


def _own_yield(node) -> bool:
    """Does the def (or lambda) yield in its own scope, i.e. is it a generator function?"""
    todo = [node.body] if isinstance(node, ast.Lambda) else list(node.body)
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        todo.extend(ast.iter_child_nodes(n))
    return False


def _nested_kinds(fn, line_of) -> dict:
    out: dict = {}
    for n in ast.walk(fn):
        if n is not fn and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            base = (line_of(n.lineno), type(n).__name__, getattr(n, "name", ""))
            k = 0
            while base + (k,) in out:
                k += 1
            out[base + (k,)] = _own_yield(n)
    return out


def _shape(node, holes: set[int]) -> str:
    """ast.dump of a def with the statements in `holes` taken out of their blocks."""
    saved = []
    for n in ast.walk(node):
        for fld in ("body", "orelse", "finalbody"):
            lst = getattr(n, fld, None)
            if isinstance(lst, list) and lst and any(id(x) in holes for x in lst):
                saved.append((n, fld, lst))
                setattr(n, fld, [x for x in lst if id(x) not in holes])
    try:
        return ast.dump(node)
    finally:
        for n, fld, lst in saved:
            setattr(n, fld, lst)


def _seg_text(lines: list[str], first: int, last: int) -> str:
    """Lines first..last of a file as a module of their own that keeps the file's line numbers
    (an indented def is put under `if 1:`, which opens no scope)."""
    seg = "\n".join(lines[first - 1:last]) + "\n"
    if lines[first - 1][:1] in (" ", "\t"):
        return "\n" * (first - 2) + "if 1:\n" + seg
    return "\n" * (first - 1) + seg


def _scope_tables(text: str, name: str, path: str) -> list:
    """(line, type, name, {symbol: (local, parameter, declared global, nonlocal, free)}) for the
    def `name` at the top of the segment and every scope nested in it, from the compiler's own
    symbol tables."""
    top = symtable.symtable(text, path, "exec")
    out: list = []

    def walk(tab):
        syms = {y.get_name(): (y.is_local(), y.is_parameter(), y.is_declared_global(), y.is_nonlocal(), y.is_free())
                for y in tab.get_symbols()}
        out.append((tab.get_lineno(), str(tab.get_type()).rsplit(".", 1)[-1].lower(), tab.get_name(), syms))
        for ch in tab.get_children():
            walk(ch)

    for ch in top.get_children():
        if ch.get_name() == name:
            walk(ch)
            break
    return out


_PLAIN = (False, False, False, False, False)  # a name only read here and found in the module or the builtins
# What does NOT need the whole def:
# - a name this scope only reads (from the module, the builtins or an enclosing def) comes, goes
#   or is found one level further out: the scope whose binding changed is judged on its own;
# - a local name LOSES its last binding (the statement was deleted or became `pass`) and is read
#   from further out afterwards. Whoever read it before had run one of its bindings (or got an
#   UnboundLocalError / NameError in the pinned tree), and every binding that went is a judged
#   statement, so its executors are already candidates.
# Everything else does: a name that becomes local, a parameter, a global or nonlocal declaration.
_LOCAL = (True, False, False, False, False)
_FREE = (False, False, False, False, True)


def _scopes_differ(told: list, tnew: list, line_of, in_old, in_new) -> str | None:
    """Why a name of code that did not change may now mean another variable, or None."""
    def keyed(tabs, f):
        d: dict = {}
        for ln, ty, nm, syms in tabs:
            base = (f(ln), ty, nm)
            k = 0
            while base + (k,) in d:
                k += 1
            d[base + (k,)] = (ln, syms)
        return d

    do, dn = keyed(told, lambda x: x), keyed(tnew, line_of)
    for key, (ln, so) in do.items():
        if key not in dn:
            if not in_old(ln):
                return f"scope {key[2]} at {ln} has no counterpart"
            continue
        sn = dn[key][1]
        for nm in set(so) | set(sn):
            a, b = so.get(nm, _PLAIN), sn.get(nm, _PLAIN)
            if a != b and not (b in (_PLAIN, _FREE) and a in (_PLAIN, _FREE, _LOCAL)):
                return f"name {nm} is bound differently in {key[2]}"
    for key, (ln, sn) in dn.items():
        if key not in do and not in_new(ln):
            return f"new scope {key[2]} at +{ln} has no counterpart"
    return None


_BLOCK_WORDS = ("if", "elif", "else", "for", "while", "try", "except", "finally", "with", "class", "def", "async", "match", "case")


def _block_word(text: str) -> str:
    """The keyword of a line that opens a block (`else:`, `if x:`, `try:` ...), else ''."""
    t = text.strip()
    m = re.match(r"[a-z]+", t)
    if m and m.group(0) in _BLOCK_WORDS and t.split("#")[0].rstrip().endswith(":"):
        return m.group(0)
    return ""


def _line_binds(text: str) -> frozenset:
    """The names one line binds when it is a statement of its own (assignment targets, imported
    names, def and class names); for the first line of a longer statement, the text left of its
    `=`. Used to see whether a replaced line at class or module level defines other names."""
    t = text.strip()
    try:
        names: set[str] = set()
        for n in ast.walk(ast.parse(t)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.stmt):
                names |= _assigned_names(n)
        return frozenset(names)
    except (SyntaxError, ValueError, RecursionError):
        m = re.match(r"([^=(\[{]*?)(?<![=!<>+\-*/%&|^:@])=(?!=)", t)
        return frozenset({"?" + m.group(1).strip()}) if m else frozenset()


def _inner_stmt(stmts: list, line: int):
    best = None
    for s0, s1, node in stmts:
        if s0 <= line <= s1 and (best is None or s1 - s0 < best[1] - best[0]):
            best = (s0, s1, node)
    return best[2] if best else None


def _scope_rule(path: str, fc: FileChange, new_src: str, origin: list, sc: FileScopes, live: set[int],
                inserted: set[int], M: Map, A: Analysis, users: set[int]) -> None:
    """v9. What a changed line of a def does to the REST of the def.

    The rules above judge a change by its own statement: who ran that statement can see it.
    That holds only if the patched def is the old def with those statements exchanged and
    nothing else: the same blocks around the same other statements, every name of the other
    statements bound to the same variable, the def still (not) a generator. Here this is
    proven for each outermost def the patch touches, and where the proof fails everyone who
    runs the def is a candidate:
      1. shape: the old and the patched def are the same syntax tree once the judged
         statements (the innermost statement of each changed line, and its counterpart at the
         same place on the other side) are taken out of their blocks; an `else:`, `try:`,
         `if ...:` line or a changed indentation that regroups other statements fails this,
         unless it stays inside one judged compound statement, whose executors are selected;
      2. names: the compiler's symbol tables of both defs give every name of every surviving
         scope the same binding (local / parameter / global / nonlocal / free); a name that is
         only read may come or go;
      3. kind: no def or lambda in it gains or loses its last `yield` (if the outermost def
         does, callers that never iterate it see the change without running a line of it:
         every user of the file);
      4. no judged old statement is one without bytecode (global, nonlocal, docstring, bare
         annotation): the map cannot know who "ran" it.
    A statement inserted whole runs for whoever finished the statement before it in its block
    (or entered the block): the executors of that whole statement are added."""
    old_lines, new_lines = sc.source.split("\n"), new_src.split("\n")
    new_of = {o: i + 1 for i, o in enumerate(origin) if o is not None}
    removed = sorted(set(fc.removed))
    removed_set = set(removed)
    added = [i + 1 for i, o in enumerate(origin) if o is None and (_is_code_line(new_lines[i]) or i in live)]

    def outer_def(line: int):
        best = None
        for f in sc.funcs:
            if f[0] <= line <= f[1] and (best is None or f[1] - f[0] > best[1] - best[0]):
                best = f
        return best

    units: dict = {}
    for ln in removed:
        f = outer_def(ln)
        if f is not None:
            units[f] = True
    for n in added:
        a = next((origin[j] for j in range(n - 2, -1, -1) if origin[j] is not None), None)
        f = outer_def(a) if a is not None else None
        if f is not None:
            units[f] = True

    # Class and module level are not proven statement by statement (the rules above judge such
    # a line by the names its own statement defines). A line there that opens or closes a
    # block, or that changes its indentation, regroups the statements around it: nothing is
    # known about which names that touches, so every user of the file.
    top: list[str] = []
    for old_no, new_no, old_text, new_text in replacement_targets(fc):
        if outer_def(old_no) is None and (_is_code_line(old_text) or _is_code_line(new_text)):
            io, inw = len(old_text) - len(old_text.lstrip()), len(new_text) - len(new_text.lstrip())
            if _block_word(old_text) != _block_word(new_text) or (io != inw and _is_code_line(old_text) and _is_code_line(new_text)):
                top.append(f"{old_no}")
            elif _line_binds(new_text) - _line_binds(old_text):
                # the line now defines a name it did not define before (a target renamed to an
                # attribute the class inherits): the rules above know the readers of the OLD names
                # only. A name that is merely no longer defined is theirs.
                top.append(f"{old_no} binds other names")
    paired_old = {t[0] for t in replacement_targets(fc)}
    paired_new = {t[1] for t in replacement_targets(fc)}
    for ln in removed:
        if ln not in paired_old and outer_def(ln) is None and _block_word(old_lines[ln - 1]):
            top.append(f"-{ln}")
    for n in added:
        if n not in paired_new and _block_word(new_lines[n - 1]):
            a = next((origin[j] for j in range(n - 2, -1, -1) if origin[j] is not None), None)
            b = next((origin[j] for j in range(n, len(origin)) if origin[j] is not None), None)
            if (a is None or outer_def(a) is None) and (b is None or outer_def(b) is None):
                top.append(f"+{n}")
    if top:
        A.cands |= users
        A.opaque = True
        A.changes.append((path, "import", 0, 0))
        A.notes.append(f"{path}:{','.join(top[:4])} a line at class or module level regroups statements or binds other names -> all {len(users)} users")

    def wide(f, why: str, everyone: bool = False) -> None:
        more = set(users) if everyone else (M.rt_lines(path, f[0], f[1]) | M.imp_lines(path, f[2], f[1]))
        A.cands |= more
        if everyone:
            A.opaque = True
            A.changes.append((path, "import", 0, 0))
        else:
            A.changes.append((path, "body", f[2], f[1]))
        A.notes.append(f"{path}:{f[0]} def: {why} -> {'all ' if everyone else ''}{len(more)}")

    for f in units:
        s, e, b0 = f
        fo = next((nd for s0, s1, nd in sc.stmt_nodes if s1 == e and isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and min([d.lineno for d in nd.decorator_list] + [nd.lineno]) == s), None)
        if fo is None or s in removed_set or s not in new_of or fo.lineno in removed_set:
            wide(f, "its first line changed")
            continue
        ns = new_of[s]
        nxt = next((new_of[o] for o in range(e + 1, len(old_lines) + 1) if o in new_of), None)
        ne = nxt - 1 if nxt is not None else len(new_lines)
        try:
            new_text = _seg_text(new_lines, ns, ne)
            tree = ast.parse(new_text)
            body = tree.body[0].body if new_lines[ns - 1][:1] in (" ", "\t") else tree.body
            fn = body[0]
        except (SyntaxError, ValueError, IndexError, AttributeError):
            wide(f, "the patched def does not stand alone", True)
            continue
        if type(fn) is not type(fo) or fn.name != fo.name or min([d.lineno for d in fn.decorator_list] + [fn.lineno]) != ns:
            wide(f, "the patched def is another def", True)
            continue
        nb0, nend = fn.body[0].lineno, fn.end_lineno

        def line_of(n: int):
            return origin[n - 1] if 0 < n <= len(origin) else None

        old_stmts = [(s0, s1, nd) for s0, s1, nd in sc.stmt_nodes if b0 <= s0 and s1 <= e]
        new_stmts = [(nd.lineno, nd.end_lineno, nd) for nd in ast.walk(fn) if isinstance(nd, ast.stmt) and nd is not fn]
        j_old: dict = {}
        j_new: dict = {}
        direct: set[int] = set()
        for ln in removed:
            if b0 <= ln <= e:
                st = _inner_stmt(old_stmts, ln)
                if st is not None:
                    j_old[id(st)] = st
                    direct.add(id(st))
        for n in added:
            if nb0 <= n <= nend:
                st = _inner_stmt(new_stmts, n)
                if st is not None:
                    j_new[id(st)] = st
        for st in list(j_old.values()):
            n = new_of.get(st.lineno) if st.lineno not in removed_set else None
            if n is not None:
                for s0, s1, nd in new_stmts:
                    if s0 == n and type(nd) is type(st) and nd.col_offset == st.col_offset:
                        j_new[id(nd)] = nd
                        break
        whole_new: list = []
        for st in list(j_new.values()):
            o = line_of(st.lineno)
            hit = None
            if o is not None:
                hit = next((nd for s0, s1, nd in old_stmts if s0 == o and type(nd) is type(st) and nd.col_offset == st.col_offset), None)
            if hit is not None:
                j_old[id(hit)] = hit
            elif o is None and all((x - 1) in inserted or not _is_code_line(new_lines[x - 1]) for x in range(st.lineno, st.end_lineno + 1)):
                # every code line of it is an insertion (no removed line stands for it)
                whole_new.append(st)
        if _shape(fo, set(j_old)) != _shape(fn, set(j_new)):
            wide(f, "the statements around the change are grouped differently")
            continue
        if _own_yield(fo) != _own_yield(fn):
            wide(f, "it gains or loses its last yield", True)
            continue
        why = None
        if any(isinstance(st, (ast.Global, ast.Nonlocal)) for st in list(j_old.values()) + list(j_new.values())):
            why = "a global/nonlocal declaration changed"
        elif any(_is_nocode(st) for st in j_old.values()):
            why = "a statement without bytecode was replaced"
        else:
            ko, kn = _nested_kinds(fo, lambda x: x), _nested_kinds(fn, line_of)
            if any(k in kn and kn[k] != v for k, v in ko.items()):
                why = "a nested def gains or loses its last yield"
        if why is None:
            in_old = lambda ln: any(st.lineno <= ln <= st.end_lineno for st in j_old.values())  # noqa: E731
            in_new = lambda ln: any(st.lineno <= ln <= st.end_lineno for st in j_new.values())  # noqa: E731
            try:
                why = _scopes_differ(_scope_tables(_seg_text(old_lines, s, e), fo.name, path),
                                     _scope_tables(new_text, fn.name, path), line_of, in_old, in_new)
            except (SyntaxError, ValueError, RecursionError) as exc:
                why = f"no symbol table ({type(exc).__name__})"
        if why is not None:
            wide(f, why)
            continue
        # proven local: every judged old statement is covered by its executors (the rules above
        # did this for the statement of a removed line; a counterpart found through an added
        # line, e.g. the `if` that got an `else:`, is covered here)
        for st in j_old.values():
            more = M.rt_lines(path, st.lineno, st.end_lineno) | M.imp_lines(path, st.lineno, st.end_lineno)
            if id(st) not in direct or not more <= A.cands:
                A.notes.append(f"{path}:{st.lineno} {type(st).__name__} as a whole -> {len(more)}")
                A.changes.append((path, "body", st.lineno, st.end_lineno))
            A.cands |= more
        if whole_new:
            where: dict = {}
            for nd in ast.walk(fn):
                for fld in ("body", "orelse", "finalbody"):
                    lst = getattr(nd, fld, None)
                    if isinstance(lst, list):
                        for i, x in enumerate(lst):
                            where[id(x)] = (lst, i)
            for st in whole_new:
                lst, i = where[id(st)]
                anchor = None
                for x in list(reversed(lst[:i])) + lst[i + 1:]:
                    o = line_of(x.lineno)
                    if o is None or _is_nocode(x):
                        continue
                    anchor = next((nd for s0, s1, nd in old_stmts if s0 == o and type(nd) is type(x) and nd.col_offset == x.col_offset), None)
                    if anchor is not None:
                        break
                if anchor is None:
                    wide(f, f"a statement inserted at +{st.lineno} with no unchanged statement beside it")
                    break
                more = M.rt_lines(path, anchor.lineno, anchor.end_lineno) | M.imp_lines(path, anchor.lineno, anchor.end_lineno)
                if not more <= A.cands:
                    A.notes.append(f"{path}:+{st.lineno} inserted beside the {type(anchor).__name__} at {anchor.lineno} -> {len(more)}")
                    A.changes.append((path, "body", anchor.lineno, anchor.end_lineno))
                A.cands |= more


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


def _lambda_confined(old_src: str, new_src: str, sc: FileScopes, ln: int, span) -> bool:
    """True if the original and the patched module differ only inside the body of one lambda:
    the innermost one whose body holds the changed columns of line `ln`. Proven on the syntax
    trees: with that body blanked in both, the two modules are identical. Importing the module
    then builds the same objects as before (only that function's code differs), so the change
    can be observed only by running the lambda's body."""
    if span is None:
        return False
    best = None
    for (l0, c0, l1, c1) in sc.lambdas:
        if (l0, c0) <= (ln, span[0]) and (ln, span[1]) <= (l1, c1):
            if best is None or (l1 - l0, c1 - c0) < (best[2] - best[0], best[3] - best[1]):
                best = (l0, c0, l1, c1)
    if best is None:
        return False
    try:
        trees = [ast.parse(old_src), ast.parse(new_src)]
    except (SyntaxError, ValueError):
        return False
    for tree in trees:
        found = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda) and (node.body.lineno, node.body.col_offset) == (best[0], best[1]):
                node.body = ast.Constant(value=None)
                found += 1
        if found != 1:
            return False
    return ast.dump(trees[0]) == ast.dump(trees[1])


def _rule_for_old_line(path: str, ln: int, span, sc: FileScopes, M: Map,
                       lambda_only: bool = False) -> tuple[set[int], str]:
    """Tests that can observe a change on line `ln` of the ORIGINAL file."""
    hit = M.rt_lines(path, ln, ln) | M.imp_lines(path, ln, ln)
    if span is not None:
        lam = sc.lambda_body_at(ln, *span)
        if lam is not None:
            if lambda_only:
                # (see _lambda_confined) tests that only execute the enclosing statement at
                # import time, e.g. building a class-level dict of lambdas, cannot tell: the
                # candidates are the tests that run function code on the lambda's lines
                return M.rt_lines(path, lam[0], lam[1]), "lambda"
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


# ---------------------------------------------------------------------------------------
# v10: an identity operator swapped with an equality operator (`is` <-> `==`, `is not` <-> `!=`)
#
# `a is b` and `a == b` give the same result whenever the classes of a and b make equality the
# same thing as identity. Which classes meet at each comparison of the pinned package is recorded
# in map/cmp.json (classes per comparison site, nothing else). Here, at selection time, the
# patch is checked to be exactly one such swap and nothing else; then every recorded class pair of
# that site is judged from Python's comparison protocol (the left `__eq__`, then the reflected
# one, then identity; `!=` through `__ne__`). If every pair is identity-equivalent the patched
# program computes, at every execution of that comparison, the value the pinned program
# computes, with no side effect, so no test can tell them apart and the change selects nothing.
# Anything not proven goes to the rules below, unchanged. BR_CMP=0 turns the rule off.
CMP = os.environ.get("BR_CMP", "1") == "1"
_CMP_DOC = None  # None: not loaded yet; False: unavailable
_CMP_FRESH = None
_CMP_SWAPS = {(ast.Is, ast.Eq), (ast.Eq, ast.Is), (ast.IsNot, ast.NotEq), (ast.NotEq, ast.IsNot)}
_CMP_TEXT = {("is", "=="), ("==", "is"), ("is not", "!="), ("!=", "is not")}
_CMP_OBJECT = "builtins:object"
# builtins whose __eq__/__ne__ return NotImplemented for an operand of another family
_CMP_BUILTIN = {"builtins:int": "num", "builtins:float": "num", "builtins:complex": "num", "builtins:str": "str",
                "builtins:bytes": "bytes", "builtins:tuple": "tuple", "builtins:list": "list", "builtins:dict": "dict",
                "builtins:set": "set", "builtins:frozenset": "set", "builtins:range": "range", "builtins:slice": "slice",
                "builtins:builtin_function_or_method": "bfn"}
# families of which two different ones can never be equal (case f)
_CMP_DISJOINT = {"num", "str", "tuple", "list", "dict", "set"}
# the one audited `__eq__` written in the package: accepted only while its source is this text
_CMP_EXPR = "sqlglot.expressions.core:Expression"
_CMP_EXPR_FILE = "sqlglot/expressions/core.py"
_CMP_EXPR_EQ = ("def __eq__(self, other: object) -> bool:\n"
                "        return self is other or (type(self) is type(other) and hash(self) == hash(other))")
_CMP_EXPR_NE = ("def __ne__(self, other: object) -> bool:\n"
                "        return not self.__eq__(other)")


def _cmp_doc():
    """The comparison-class map, or False if it is missing, unreadable or of another Python."""
    global _CMP_DOC
    if _CMP_DOC is None:
        _CMP_DOC = False
        try:
            import sys

            p = os.environ.get("BR_CMPMAP") or os.path.join(
                os.path.dirname(os.environ.get("BR_MAP", "/app/map/blast.map")), "cmp.json")
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
            if doc.get("v") == 1 and doc.get("python") == list(sys.version_info[:2]):
                _CMP_DOC = doc
        except Exception:  # noqa: BLE001
            _CMP_DOC = False
    return _CMP_DOC


def _cmp_tree_digest(root: str) -> str:
    """Content digest of sqlglot/ and tests/ (the digest build/fetch_sqlglot.py verifies)."""
    import hashlib

    h = hashlib.sha256()
    files = []
    for part in ("sqlglot/", "tests/"):
        for dp, dns, fns in os.walk(os.path.join(root, part)):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for fn in fns:
                if not fn.endswith(".pyc"):
                    files.append(os.path.relpath(os.path.join(dp, fn), root))
    for rel in sorted(files):
        h.update(rel.encode() + b"\0")
        with open(os.path.join(root, rel), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def _cmp_fresh(doc: dict, repo: str, M: "Map") -> bool:
    """The map was recorded on exactly this checkout and every test of tests.json ran in it."""
    global _CMP_FRESH
    if _CMP_FRESH is None:
        _CMP_FRESH = False
        try:
            _CMP_FRESH = (_cmp_tree_digest(repo) == doc["digest"]
                          and M.test_ids <= frozenset(doc["tests"]))
        except Exception:  # noqa: BLE001
            _CMP_FRESH = False
    return _CMP_FRESH


def _cmp_swap_site(old_src: str, new_src: str, ln: int) -> tuple[int, int, int, int] | None:
    """Position (lineno, col_offset, end_lineno, end_col_offset) of the swapped Compare in the old file."""
    hit = _cmp_swap_nodes(old_src, new_src, ln)
    if hit is None:
        return None
    n = hit[0]
    return (n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)


_CMP_DYNAMIC = (ast.Name, ast.Attribute, ast.Call, ast.Subscript, ast.Await, ast.Yield, ast.YieldFrom,
                ast.NamedExpr, ast.Starred)


def _cmp_literal(e: ast.expr) -> bool:
    """True if the compiler may see the operand as a literal other than None, True, False or
    `...`: it warns about `is` / `is not` with one, and a warning is something a test can see.
    Anything built without a name, an attribute, a call or a subscript counts (a superset)."""
    if isinstance(e, ast.Constant) and (e.value is None or e.value is True or e.value is False or e.value is Ellipsis):
        return False
    return not any(isinstance(n, _CMP_DYNAMIC) for n in ast.walk(e))


def _cmp_swap_nodes(old_src: str, new_src: str, ln: int):
    """The one Compare of the old file, and its counterpart in the new file, whose operator swap
    (is <-> ==, is not <-> !=, one operator, operands unchanged) is the whole difference between
    the two syntax trees; None if the trees differ in any other way.

    The files differ in line `ln` only (one line replaced by one line), so a statement that does
    not touch that line has the same text at the same place in both files; it is compared by type
    and extent. Everything on the path to the changed line is compared node by node."""
    ol, nl = old_src.split("\n"), new_src.split("\n")
    if len(ol) != len(nl) or ol[:ln - 1] != nl[:ln - 1] or ol[ln:] != nl[ln:]:
        return None
    try:
        old_tree, new_tree = ast.parse(old_src), ast.parse(new_src)
    except (SyntaxError, ValueError):
        return None
    out = []
    stack = [(old_tree, new_tree)]
    while stack:
        x, y = stack.pop()
        if type(x) is not type(y):
            return None
        if isinstance(x, ast.AST):
            if (isinstance(x, ast.Compare) and len(x.ops) == 1 and len(y.ops) == 1
                    and len(x.comparators) == 1 and len(y.comparators) == 1
                    and type(x.ops[0]) is not type(y.ops[0])):
                if (type(x.ops[0]), type(y.ops[0])) not in _CMP_SWAPS or out:
                    return None
                out.append((x, y))
                stack.append((x.left, y.left))
                stack.append((x.comparators[0], y.comparators[0]))
                continue
            for f in x._fields:
                stack.append((getattr(x, f, None), getattr(y, f, None)))
        elif isinstance(x, list):
            if len(x) != len(y):
                return None
            if x and isinstance(x[0], ast.stmt):
                for p, q in zip(x, y):
                    if type(p) is not type(q):
                        return None
                    lo = min([d.lineno for d in getattr(p, "decorator_list", [])] + [p.lineno])
                    if lo <= ln <= p.end_lineno:
                        stack.append((p, q))
                    elif ((p.lineno, p.col_offset, p.end_lineno, p.end_col_offset)
                          != (q.lineno, q.col_offset, q.end_lineno, q.end_col_offset)):
                        return None
            else:
                stack.extend(zip(x, y))
        elif x != y or repr(x) != repr(y):
            return None
    if len(out) != 1:
        return None
    return out[0]


def _cmp_expr_audit(repo: str, path: str, new_src: str) -> bool:
    """True if, in the patched tree, class Expression defines `__eq__` and `__ne__` once each,
    undecorated, with exactly the audited text, and binds those names in no other way."""
    try:
        if path == _CMP_EXPR_FILE:
            src = new_src
        else:
            with open(os.path.join(repo, _CMP_EXPR_FILE), encoding="utf-8") as fh:
                src = fh.read()
        hits = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Expression"]
        if len(hits) != 1:
            return False
        found = {"__eq__": [], "__ne__": []}
        for st in ast.walk(hits[0]):
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name in found:
                if st not in hits[0].body or st.decorator_list:
                    return False
                found[st.name].append(ast.get_source_segment(src, st))
            elif isinstance(st, ast.Name) and st.id in found and not isinstance(st.ctx, ast.Load):
                return False
            elif isinstance(st, ast.Constant) and st.value in ("__eq__", "__ne__"):
                return False
        return found["__eq__"] == [_CMP_EXPR_EQ] and found["__ne__"] == [_CMP_EXPR_NE]
    except Exception:  # noqa: BLE001
        return False


def _cmp_family(c: list, expr_ok) -> str:
    """How instances of a recorded class compare: c = [class, class defining __eq__, class
    defining __ne__, is an Enum, is a Flag, Enum members compare by identity, same facts everywhere]."""
    key, eq, ne, _en, fl, _ok, uniq = c
    if not uniq or fl:
        return "other"
    if key == "builtins:NoneType":
        return "none"
    if key == "builtins:bool":
        return "bool"
    if eq == _CMP_OBJECT and ne == _CMP_OBJECT:
        return "default"  # object's own __eq__/__ne__: identity
    if eq == _CMP_EXPR and ne == _CMP_EXPR:
        return "expr" if expr_ok() else "other"
    if eq == ne:
        fam = _CMP_BUILTIN.get(eq)
        if fam == "bfn":
            return "bfn" if key == eq else "other"
        if fam:
            return fam
    return "other"  # its own __eq__/__ne__, a metaclass's, a C type not listed: never equivalent


def _cmp_pair(a: list, b: list, expr_ok) -> bool:
    """True only if `x is y` == (`x == y`) and `x is not y` == (`x != y`) for every x of exact
    class a and y of exact class b, with no side effect. Conservative: unlisted means False."""
    fa, fb = _cmp_family(a, expr_ok), _cmp_family(b, expr_ok)
    if fa == "other" or fb == "other":
        return False
    if fa == "none" or fb == "none":
        return True  # a, b: nothing listed ever equals None
    if a[3] or b[3]:  # an Enum class
        if a[3] and b[3]:
            if a[0] == b[0]:  # c: two members of one Enum; with a str/int mix-in only if measured
                return fa == "default" or a[5] is True
            return fa == "default" and fb == "default"
        fe, fo = (fa, fb) if a[3] else (fb, fa)
        return fe == "default" and fo in ("default", "expr")  # never an Enum member with a str or an int
    if fa == "bool" and fb == "bool":
        return True  # a: True and False are singletons
    if fa == "default" or fb == "default":
        return True  # d, e: identity on one side, NotImplemented or identity on the other
    if fa == "expr" and fb == "expr":
        return a[0] != b[0]  # g: two different exact Expression classes; the same class compares by hash
    if fa == "expr" or fb == "expr":
        return True  # g: Expression.__eq__ is False for another type, the builtin returns NotImplemented
    ga, gb = ("num" if fa == "bool" else fa), ("num" if fb == "bool" else fb)
    return ga != gb and ga in _CMP_DISJOINT and gb in _CMP_DISJOINT  # f; the same family is never equivalent


def _cmp_rule(path: str, fc: FileChange, repo: str, old_src: str, new_src: str, M: "Map") -> str | None:
    """A note about the patch as an identity/equality swap. The change is proven unobservable
    exactly when the note ends with "all identity-equivalent"; None if it is not such a swap."""
    try:
        ln = fc.removed[0]
        new_line = _replacement_pairs(fc).get(ln)
        if new_line is None or fc.mode_change:
            return None
        old_line = old_src.split("\n")[ln - 1]
        a = 0
        while a < len(old_line) and a < len(new_line) and old_line[a] == new_line[a]:
            a += 1
        bo, bn = len(old_line), len(new_line)
        while bo > a and bn > a and old_line[bo - 1] == new_line[bn - 1]:
            bo -= 1
            bn -= 1
        if (old_line[a:bo].strip(), new_line[a:bn].strip()) not in _CMP_TEXT:
            return None
        hit = _cmp_swap_nodes(old_src, new_src, ln)
        if hit is None:
            return None
        old_node, new_node = hit
        site = (old_node.lineno, old_node.col_offset, old_node.end_lineno, old_node.end_col_offset)
        if isinstance(new_node.ops[0], (ast.Is, ast.IsNot)) and (
                _cmp_literal(new_node.left) or _cmp_literal(new_node.comparators[0])):
            return "cmp: `is` with a literal operand: the compiler warns about it"
        doc = _cmp_doc()
        if not doc:
            return "cmp: no class map for this Python"
        P = doc["sites"].get(path, {}).get("%d|%d|%d|%d" % site)
        if not P:
            return None  # no test of the profile ran this comparison: the rules below decide
        if isinstance(P, dict):
            return "cmp: the classes at this comparison differ between profile runs"
        audit = []

        def expr_ok() -> bool:
            if not audit:
                audit.append(_cmp_expr_audit(repo, path, new_src))
            return audit[0]

        classes = doc["classes"]
        for ia, ib in P:
            if not _cmp_pair(classes[ia], classes[ib], expr_ok):
                return f"cmp: {len(P)} class pairs, not all identity-equivalent ({classes[ia][0]} with {classes[ib][0]})"
        if not _cmp_fresh(doc, repo, M):
            return "cmp: class map is stale for this checkout or test list"
        return f"cmp: {len(P)} class pairs, all identity-equivalent"
    except Exception as exc:  # noqa: BLE001
        return f"cmp: rule failed ({exc!r})"


# ---- v11: readers of one entry of a class-level table ------------------------------------
TAB = os.environ.get("BR_TAB", "1") == "1"
_TAB_DOC = None  # None: not loaded yet; False: unavailable
_TAB_FRESH = None
_TAB_LINES = None


def _tab_doc():
    """The table-entry reader map, or False if it is missing, unreadable or of another Python."""
    global _TAB_DOC
    if _TAB_DOC is None:
        _TAB_DOC = False
        try:
            import sys

            p = os.environ.get("BR_TABMAP") or os.path.join(
                os.path.dirname(os.environ.get("BR_MAP", "/app/map/blast.map")), "tab.json")
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
            if doc.get("v") == 1 and doc.get("python") == list(sys.version_info[:2]):
                _TAB_DOC = doc
        except Exception:  # noqa: BLE001
            _TAB_DOC = False
    return _TAB_DOC


def _tab_lines(doc: dict) -> dict:
    """file -> the lines on which the key of a listed entry starts."""
    global _TAB_LINES
    if _TAB_LINES is None:
        out: dict = {}
        for site, ent in doc["sites"].items():
            lines = out.setdefault(site.split("|")[0], set())
            for pos in ent:
                lines.add(int(pos.split(":")[0]))
        _TAB_LINES = out
    return _TAB_LINES


def _tab_fresh(doc: dict, repo: str, M: "Map") -> bool:
    """The map was recorded on exactly this checkout and every test of tests.json ran in it."""
    global _TAB_FRESH
    if _TAB_FRESH is None:
        _TAB_FRESH = False
        try:
            _TAB_FRESH = (_cmp_tree_digest(repo) == doc["digest"]
                          and M.test_ids <= frozenset(doc["tests"]))
        except Exception:  # noqa: BLE001
            _TAB_FRESH = False
    return _TAB_FRESH


def _tab_literal(e: ast.expr) -> bool:
    """A display of constants: evaluating it cannot fail, call anything or bind anything."""
    if isinstance(e, ast.Constant):
        return e.value is None or type(e.value) in (str, bytes, int, float, bool)
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
        return isinstance(e.operand, ast.Constant) and type(e.operand.value) in (int, float)
    if isinstance(e, (ast.Tuple, ast.List)):
        return all(_tab_literal(x) for x in e.elts)
    if isinstance(e, ast.Set):
        return all(isinstance(x, ast.Constant) and type(x.value) in (str, bytes, int, float, bool) for x in e.elts)
    if isinstance(e, ast.Dict):
        return (all(isinstance(k, ast.Constant) and type(k.value) in (str, bytes, int, float, bool) for k in e.keys)
                and all(_tab_literal(v) for v in e.values))
    return False


def _tab_stmt(tree: ast.Module, ln: int):
    """(index path, statement, name) of the class-level `NAME = {...}` holding line `ln`: a direct
    statement of a class body whose enclosing blocks, up to the module, are class bodies only."""
    body, trail = tree.body, []
    while True:
        for i, st in enumerate(body):
            if st.lineno <= ln <= (st.end_lineno or st.lineno):
                break
        else:
            return None
        trail.append(i)
        if isinstance(st, ast.ClassDef):
            if ln < st.body[0].lineno:
                return None  # decorators, bases, keywords
            body = st.body
            continue
        if len(trail) < 2:
            return None  # module level
        name = None
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            name = st.targets[0].id
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            name = st.target.id
        if name is None or not isinstance(st.value, ast.Dict):
            return None
        return trail, st, name


def _tab_rule(path: str, fc: FileChange, repo: str, old_src: str, new_src: str, M: "Map"):
    """v11. (tests, note) when the whole patch replaces, on one line, a literal value of ONE entry of
    a class-level dict display by another literal, and the entry map lists the entry; else None.

    Why only the listed tests can tell: the two files then build the same classes, functions and
    tables except for the object stored under that one key (same keys, same order, no call and no
    name involved in the old or the new value, so nothing else runs differently at import). The
    map was recorded with that very table replaced by a dict that notes every lookup of a key and
    every read of all its values, wherever the table object travels (subclasses, instances, other
    classes that keep it, the tokenizer core). An entry is listed only if its key and the whole
    table were never read while a module was being executed or outside a test (so nothing derived
    from the value is kept anywhere at import), and the table was never changed after it was built."""
    try:
        if not TAB or fc.mode_change or not path.startswith("sqlglot/"):
            return None
        ln = fc.removed[0]
        if ln <= 2 or _replacement_pairs(fc).get(ln) is None:
            return None
        doc = _tab_doc()
        if not doc or ln not in _tab_lines(doc).get(path, ()):
            return None  # no listed entry starts on this line: nothing to parse
        told, tnew = ast.parse(old_src), ast.parse(new_src)
        hit = _tab_stmt(told, ln)
        if hit is None:
            return None
        trail, st, name = hit
        node = tnew
        for i in trail:
            body = node.body
            if i >= len(body):
                return None
            node = body[i]
        if type(node) is not type(st) or not isinstance(node.value, ast.Dict):
            return None
        do, dn = st.value, node.value
        if len(do.keys) != len(dn.keys) or len(do.values) != len(dn.values):
            return None
        diff = [i for i in range(len(do.values)) if ast.dump(do.values[i]) != ast.dump(dn.values[i])]
        if len(diff) != 1 or do.keys[diff[0]] is None:
            return None
        i = diff[0]
        if not (_tab_literal(do.values[i]) and _tab_literal(dn.values[i])):
            return None
        key = do.keys[i]
        do.values[i] = dn.values[i] = ast.Constant(value=None)
        if ast.dump(told) != ast.dump(tnew):
            return None
        ent = doc["sites"].get("%s|%d|%s" % (path, st.lineno, name))
        if ent is None:
            return None
        k = ent.get("%d:%d" % (key.lineno, key.col_offset))
        if k is None or not _tab_fresh(doc, repo, M):
            return None
        tests, acc, ids = set(), 0, doc["tests"]
        for d in doc["sets"][k]:
            acc += d
            n = M.num.get(ids[acc])
            if n is not None:
                tests.add(n)
        tests |= M.openers(path) | set(M.unmapped)
        return tests, f"table entry {name}[{ast.unparse(key)}]: {len(tests)} readers"
    except Exception:  # noqa: BLE001
        return None


def analyze_python(fc: FileChange, repo: str, M: Map, A: Analysis) -> None:
    path = fc.old_path or fc.path
    old_file = os.path.join(repo, path)
    base = os.path.basename(fc.path)
    if base in PYTEST_CONFIG:
        A.broad = True
        A.notes.append(f"{fc.path}: pytest config")
        return
    if fc.is_new or not os.path.isfile(old_file):
        why_all = _new_module_reach(fc.new_path or fc.path, repo)
        if why_all is not None:
            A.broad = True
            A.notes.append(f"{fc.path}: new python file {why_all}")
            return
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
        if head_changed and not _same_tree_as_bytes(old_src, new_src):
            # v8: a coding cookie that changes how the file's bytes are read
            A.cands |= users
            A.opaque = True
            A.changes.append((path, "import", 0, 0))
            A.notes.append(f"{fc.path}: same tokens, another tree when read from bytes -> all {len(users)} users")
            return
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
    if CMP and len(A.files) == 1 and len(fc.removed) == 1 and len(fc.added_text) == 1:
        # v10: the whole patch is one line; if it is one identity/equality swap proven unobservable
        # from the classes that meet there, it selects nothing
        note = _cmp_rule(path, fc, repo, old_src, new_src, M)
        if note:
            A.notes.append(f"{path}:{fc.removed[0]} {note}")
            if note.endswith("all identity-equivalent"):
                return
        # v11: one literal value of one entry of a class-level table: only the tests that looked
        # that key up, or read all values of the table, can tell (see _tab_rule)
        tab = _tab_rule(path, fc, repo, old_src, new_src, M)
        if tab is not None:
            A.notes.append(f"{path}:{fc.removed[0]} {tab[1]}")
            if tab[0]:
                A.cands |= tab[0]
                A.changes.append((path, "import", fc.removed[0], fc.removed[0]))
            return
    sc = FileScopes(old_src)
    old_lines = old_src.split("\n")
    pairs = _replacement_pairs(fc)
    n_changes0 = len(A.changes)
    stmt_starts = {st[0] for st in sc.stmts}
    new_src_lines = new_src.split("\n")
    # 1. modified and removed lines, judged in the original file
    for ln in sorted(set(fc.removed)):
        span = _changed_span(old_lines[ln - 1], pairs[ln]) if ln in pairs else None
        lambda_only = (len(fc.removed) == 1 and len(fc.added_text) == 1
                       and _lambda_confined(old_src, new_src, sc, ln, span))
        hit, kind = _rule_for_old_line(path, ln, span, sc, M, lambda_only)
        A.cands |= hit
        A.notes.append(f"{path}:{ln} {kind} -> {len(hit)}")
        A.changes.append((path, kind.split("/")[0], *_runtime_range(path, ln, span, sc, kind)))
    # 1b. v8: a statement of a function body that now starts further left has left its block (an
    # `if`, a loop, a nested def): it also runs for tests that never ran the old line. Judged
    # per block of the patch: if a line that starts a statement in the patched file sits left
    # of every removed statement of that block, everyone who runs the outermost enclosing def
    # is a candidate; if it left that def too (class body or module level), everyone who
    # loads the file.
    moved_sc = None
    for h in fc.hunks:
        old_no, new_no = h.old_start, h.new_start
        hl = h.lines
        i = 0
        while i < len(hl):
            if not (hl[i].startswith("-") or hl[i].startswith("+")):
                old_no += 1
                new_no += 1
                i += 1
                continue
            j = i
            while j < len(hl) and hl[j].startswith("-"):
                j += 1
            k = j
            while k < len(hl) and hl[k].startswith("+"):
                k += 1
            gone = []
            for t in range(i, j):
                ln = old_no + (t - i)
                fn = sc.func_of(ln)
                if ln in stmt_starts and _is_code_line(hl[t][1:]) and fn is not None and fn[2] <= ln <= fn[1]:
                    gone.append(ln)
            if gone and k > j:
                ind_old = min(len(old_lines[ln - 1]) - len(old_lines[ln - 1].lstrip()) for ln in gone)
                left = [new_no + (t - j) for t in range(j, k)
                        if _is_code_line(hl[t][1:]) and len(hl[t][1:]) - len(hl[t][1:].lstrip()) < ind_old]
                if left:
                    if moved_sc is None:
                        moved_sc = FileScopes(new_src)
                    starts_new = {st[0] for st in moved_sc.stmts}
                    left = [n for n in left if n in starts_new]
                if left:
                    ln = gone[0]
                    ind_new = min(len(new_src_lines[n - 1]) - len(new_src_lines[n - 1].lstrip()) for n in left)
                    outer = None
                    for fn in sc.funcs:
                        if fn[0] <= ln <= fn[1] and (outer is None or fn[1] - fn[0] > outer[1] - outer[0]):
                            outer = fn
                    head = old_lines[outer[0] - 1]
                    if ind_new <= len(head) - len(head.lstrip()):
                        more = set(users)
                        A.changes.append((path, "import", ln, ln))
                    else:
                        more = M.rt_lines(path, outer[0], outer[1])
                        A.changes.append((path, "body", outer[2], outer[1]))
                    A.cands |= more
                    A.notes.append(f"{path}:{ln} moved left out of its block -> {len(more)}")
            old_no += j - i
            new_no += k - j
            i = k
    # 2. inserted lines, judged in the patched file and mapped back to the original
    new_lines = new_src.split("\n")
    surplus = _insert_new_lines(fc)
    # v8: whether an added line is "only a comment or a blank line" is a fact about the program
    # (see _live_quiet_lines), and a removed comment or blank line stands for no added code
    live = _live_quiet_lines(new_src, new_lines, new_tok, origin)
    unhosted = _unhosted_new_lines(fc, set())
    inserted = [
        i for i, o in enumerate(origin)
        if o is None and _is_code_line(new_lines[i]) and ((i + 1) in surplus or (i + 1) in unhosted)
    ]
    if live:
        # Each live quiet line belongs to the statement that contains it: the one its nearest
        # original lines above and below are both part of (the string, or the continued line,
        # runs from one to the other). Proven on the syntax trees: with those statements
        # emptied in both files, the files are identical. Then the statement is judged by the
        # rule for a changed line of it (import time at module or class level, executors in a
        # function body). Anything else: nothing is known, every user of the file.
        host: dict[int, tuple] = {}
        try:
            lsc = FileScopes(new_src)
            for i in sorted(live):
                jp = next((j for j in range(i - 1, -1, -1) if origin[j] is not None), None)
                jn = next((j for j in range(i + 1, len(origin)) if origin[j] is not None), None)
                so = sc.stmt_node_of(origin[jp]) if jp is not None and jn is not None else None
                sn = lsc.stmt_node_of(jp + 1) if so is not None else None
                if so is None or sn is None or so is not sc.stmt_node_of(origin[jn]) or sn is not lsc.stmt_node_of(jn + 1):
                    host = {}
                    break
                host[origin[jp]] = (so, sn)
            if host and not _same_but_stmts(old_src, new_src, [v[0] for v in host.values()],
                                            [v[1] for v in host.values()]):
                host = {}
        except (SyntaxError, ValueError):
            host = {}
        if host:
            done: set[int] = set()
            for o in sorted(host):
                if id(host[o][0]) in done:
                    continue
                done.add(id(host[o][0]))
                hit, kind = _rule_for_old_line(path, o, None, sc, M)
                A.cands |= hit
                A.notes.append(f"{path}:{o} {kind} (line inside a string or continuation) -> {len(hit)}")
                A.changes.append((path, kind.split("/")[0], *_runtime_range(path, o, None, sc, kind)))
        else:
            A.cands |= users
            A.opaque = True
            A.changes.append((path, "import", 0, 0))
            A.notes.append(f"{path}: a line that looks like a comment changes the tree beyond one statement"
                           f" -> all {len(users)} users")
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
    # 3. v9: what a changed line does to the rest of its def (see _scope_rule)
    try:
        _scope_rule(path, fc, new_src, origin, sc, live, set(inserted), M, A, users)
    except Exception as exc:  # noqa: BLE001  (a line-local judgement that could not be proven is not trusted)
        A.cands |= users
        A.opaque = True
        A.changes.append((path, "import", 0, 0))
        A.notes.append(f"{path}: no proof that the change stays in its statement ({type(exc).__name__}: {exc}) -> all {len(users)} users")
    if len(A.changes) == n_changes0:
        if _same_but_bare_strings(old_src, new_src):
            # v8: proven on the syntax trees: only docstrings / bare string statements differ
            A.notes.append(f"{fc.path}: comments/blank lines only (bare strings)")
            return
        # v8: the syntax trees differ and no rule above judged anything: nothing is known
        # about where the difference acts
        A.cands |= users
        A.opaque = True
        A.changes.append((path, "import", 0, 0))
        A.notes.append(f"{path}: a difference no rule judged -> all {len(users)} users")


def _touches_pytest_section(fc: FileChange, repo: str) -> bool:
    """True if the change edits pytest configuration inside an ini/toml/cfg file."""
    base = os.path.basename(fc.path)
    if base in ("pytest.ini", "conftest.py", "pytest.toml", ".pytest.toml") or fc.is_new or fc.is_deleted:
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
    # (v8: also below tests/, where pytest looks first when it is given test ids)
    if base in PYTEST_CONFIG and (fc.path.count("/") == 0 or fc.path.startswith("tests/")
                                  or base in ("conftest.py", "pytest.ini", "pytest.toml", ".pytest.toml")):
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
    global _REPO
    _REPO = repo
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
