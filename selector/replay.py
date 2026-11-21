"""Replay of the executor's TPC tests without their external engine (duckdb).

Three of sqlglot's tests spend nearly all their time outside sqlglot:

  test_execute_tpcds / test_execute_tpch   run every TPC query through sqlglot's python
      executor and compare the rows with what duckdb returns for the same query
      (transpiled to duckdb SQL by sqlglot);
  test_optimized_tpch   runs each TPC-H query and its optimized form in duckdb and compares.

duckdb is not part of the change under test: what it returns depends only on the SQL text it
is handed and on fixture data. So each test's verdict is a function of a few values sqlglot
computes: the duckdb SQL strings, and (for the execute tests) the rows and columns of the
python executor's result. `baseline()` records hashes of those values at the pinned commit
(a map built from the pinned commit, made at image build time). `check()` recomputes them on
a patched tree by calling the same functions the test calls, in the same order, and reports
"same" only if every value is identical — then the test passes exactly as it does at the
pinned commit. Anything else (a different value, an exception, a query whose baseline is
not reproducible) is "diff", and the caller keeps the test selected.

A second, optional shortcut skips the python executor for a query: when no changed line is
ever executed after `optimize()` returns (EXEC_LINES, traced at build time in processes that
never ran the parse/optimize phase, so lazily built state is included), an optimized tree
identical to the pinned one gives identical rows. Trees are compared by a full structural
dump (classes, args in order, types, comments, meta, parent links).

Only valid when the patch does not touch tests/ or fixture data: the caller checks that.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys

T_TPCDS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
T_TPCH = "tests/test_executor.py::TestExecutor::test_execute_tpch"
T_OPT_TPCH = "tests/test_executor.py::TestExecutor::test_optimized_tpch"
REPLAYABLE = (T_TPCDS, T_TPCH, T_OPT_TPCH)


def _h(*parts: str) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(p.encode("utf-8", "surrogatepass"))
        m.update(b"\0")
    return m.hexdigest()[:16]


def _atom(v) -> str:
    t = type(v)
    if t in (str, int, float, bool, type(None), bytes):
        return f"{t.__name__}:{v!r}"
    if t in (list, tuple):
        return f"{t.__name__}[" + ",".join(_atom(x) for x in v) + "]"
    if t is dict:
        return "dict{" + ",".join(f"{_atom(k)}={_atom(x)}" for k, x in v.items()) + "}"
    if t in (set, frozenset):
        return f"{t.__name__}{{" + ",".join(sorted(_atom(x) for x in v)) + "}"
    if _is_expr(v):
        return canon(v)
    return f"{t.__module__}.{t.__qualname__}:{v!r}"


class _Lit:
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def _is_expr(v) -> bool:
    return hasattr(v, "parent") and isinstance(getattr(v, "args", None), dict)


def canon(root) -> str:
    """Deterministic dump of an expression tree: everything later code can read from it."""
    out: list[str] = []
    stack: list = [(root, None, None, None)]
    while stack:
        node, parent, key, index = stack.pop()
        if type(node) is _Lit:
            out.append(node.text)
            continue
        if not _is_expr(node):
            out.append(_atom(node))
            continue
        cls = type(node)
        out.append(f"({cls.__module__}.{cls.__qualname__}")
        if parent is not None:
            # the links optimizer rules maintain by hand
            if node.parent is not parent:
                out.append("!parent")
            if getattr(node, "arg_key", None) != key:
                out.append(f"!key={getattr(node, 'arg_key', None)!r}")
            if getattr(node, "index", None) != index:
                out.append(f"!index={getattr(node, 'index', None)!r}")
        ty = getattr(node, "_type", None)
        if ty is not None and ty is not node:
            out.append("type=" + (canon(ty) if _is_expr(ty) else _atom(ty)))
        comments = getattr(node, "comments", None)
        if comments:
            out.append("comments=" + _atom(comments))
        meta = getattr(node, "_meta", None)
        if meta:
            out.append("meta=" + _atom(meta))
        stack.append((_Lit(")"), None, None, None))
        for k, v in reversed(list(node.args.items())):
            if type(v) is list:
                stack.append((_Lit("]"), None, None, None))
                for i in range(len(v) - 1, -1, -1):
                    stack.append((v[i], node, k, i))
                stack.append((_Lit(f"{k}=["), None, None, None))
            else:
                stack.append((v, node, k, None))
                stack.append((_Lit(f"{k}="), None, None, None))
    return " ".join(out)


class _Captured(BaseException):
    pass


class _Slow(BaseException):
    pass


STEP_LIMIT_S = 4.0  # one query's step takes well under a second at the pinned commit
# v7: a per-query CPU limit beside the wall limit: a few times what the query took alone at
# the pinned commit when the image was built, plus a constant. A change that makes one query
# explode is given up after that, not after STEP_LIMIT_S.
CPU_LIMIT_FACTOR = 4.0
CPU_LIMIT_SLACK_S = 0.5
_tick_on = False


def _tick(seconds: float = STEP_LIMIT_S, cpu: float | None = None) -> None:
    """Re-arm the watchdog: a step that runs longer raises _Slow (a hang is a failure)."""
    if _tick_on:
        import signal

        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.setitimer(signal.ITIMER_PROF, cpu or 0.0)


def _arm() -> None:
    global _tick_on
    import signal

    def _late(signum, frame):
        raise _Slow()

    signal.signal(signal.SIGALRM, _late)
    signal.signal(signal.SIGPROF, _late)
    _tick_on = True


def _disarm() -> None:
    global _tick_on
    if _tick_on:
        import signal

        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
    _tick_on = False


def _capture_plan(expression, *a, **k):
    raise _Captured(expression)


def _rows_fp(table) -> str:
    cols = table.columns
    rows = table.rows
    return _h(_atom(list(cols)), *(_atom(r) for r in rows))


_LOADED: set = set()  # suites whose tables this process has loaded (once, as in the test run)


class _NoDuck:
    """Stands in for the duckdb module while the test class loads its tables. The replay never
    queries duckdb, and defining its lazy views is half of the loader's time."""

    @staticmethod
    def connect(*a, **k):
        return _NoDuck()

    def execute(self, *a, **k):
        return self

    def close(self):
        pass


def _load_tables(te, tpch: bool) -> None:
    """The test class's own loader (setUpClass), unchanged, for the python tables: duckdb is
    stubbed, and for TPC-DS the TPC-H tables (which that test never reads) are skipped."""
    real_duck, real_schema = te.duckdb, te.TPCH_SCHEMA
    te.duckdb = _NoDuck
    if not tpch:
        te.TPCH_SCHEMA = {}
    try:
        te.TestExecutor.setUpClass()
    finally:
        te.duckdb, te.TPCH_SCHEMA = real_duck, real_schema
    _LOADED.add("tpcds")
    if tpch:
        _LOADED.add("tpch")


def _exec_test(tpch: bool, base: list | None, skip_exec_ok: bool, log=None, only=None, window=None,
               stop=None, limits=None, strict_trees: bool = False):
    """Mirror of TestExecutor._mp_execute + subtestHelper for one TPC suite.

    Returns the per-query records when `base` is None (building), else True/False (same?).
    `only`: process just these queries (reach-limited replay). `window(i)` / `window(None)` is
    called when the work for query i starts / ends (build-time tracing). `stop()` is asked
    before each query; a true answer ends the run as "not the same". `limits[i]`: CPU seconds
    allowed to one step of query i (None: the wall limit alone). `strict_trees`: a tree that
    differs from the pinned one ends the run as "not the same" (the executor is not tried)."""
    import tests.test_executor as te
    import sqlglot.executor as ex
    from sqlglot import parse_one, transpile

    cls = te.TestExecutor
    rel = "optimizer/tpc-h/tpc-h.sql" if tpch else "optimizer/tpc-ds/tpc-ds.sql"
    schema = te.TPCH_SCHEMA if tpch else te.TPCDS_SCHEMA
    sqls = list(te.load_sql_fixture_pairs(rel))
    if base is not None and len(base) != len(sqls):
        return False
    real_plan = ex.Plan
    state = {"tables": None}
    fail = object()

    def lim(i):
        return limits[i] if limits is not None else None

    def real_tables():
        if state["tables"] is None:
            if ("tpch" if tpch else "tpcds") not in _LOADED:
                _tick(30.0)
                _load_tables(te, tpch)
            state["tables"] = cls.tpch_tables if tpch else cls.tpcds_tables
        return state["tables"]

    def enter(i):
        if window is not None:
            window(i)

    def one(i, meta, sql, blob):
        expression = pickle.loads(blob)
        if not meta.get("execute"):
            te.initializer(schema, {})
            return fail if te.mp_execute(expression, meta) is not None else None
        b = base[i] if base is not None else None
        if base is not None and (b is None or b[0] is None or b[1] is None or b[2] is None):
            return fail  # no reproducible baseline for this query
        duck = _h(transpile(sql, write="duckdb")[0])
        if b is not None and duck != b[1]:
            if log:
                log(f"replay: query {i} duckdb sql differs")
            return fail
        fp_opt = None
        if base is None or skip_exec_ok:
            # the optimized tree handed to the planner, captured by stopping there
            from sqlglot.executor.table import Table

            te.initializer(schema, {name: Table(columns=tuple(cols)) for name, cols in schema.items()})
            ex.Plan = _capture_plan
            try:
                te.mp_execute(pickle.loads(blob), meta)
                return fail  # the planner was never reached: not the pinned pipeline
            except _Captured as cap:
                fp_opt = _h(canon(cap.args[0]))
            finally:
                ex.Plan = real_plan
            if b is not None and fp_opt == b[0]:
                return (fp_opt, duck, b[2])
            if strict_trees or (stop is not None and stop()):
                return fail
        te.initializer(schema, real_tables())
        _tick(STEP_LIMIT_S, lim(i))
        table = te.mp_execute(expression, meta)
        if table is None:
            return fail
        table = pickle.loads(pickle.dumps(table))
        res = _rows_fp(table)
        if b is not None and res != b[2]:
            if log:
                log(f"replay: query {i} rows differ")
            return fail
        return (fp_opt, duck, res)

    records: list = []
    # the parent parses and pickles every query before any result is looked at
    blobs: list = []
    for i, (meta, sql, _) in enumerate(sqls):
        if only is not None and i not in only:
            blobs.append(None)
            continue
        _tick(STEP_LIMIT_S, lim(i))
        enter(i)
        try:
            blobs.append(pickle.dumps(parse_one(sql)))
        finally:
            enter(None)
    for i, (meta, sql, _) in enumerate(sqls):
        if blobs[i] is None:
            records.append(None)
            continue
        if stop is not None and stop():
            return False
        _tick(STEP_LIMIT_S, lim(i))
        enter(i)
        try:
            rec = one(i, meta, sql, blobs[i])
        finally:
            enter(None)
        if rec is fail:
            return False
        records.append(rec)
    return records if base is None else True


def _opt_tpch(base: list | None, log=None, only=None, window=None, stop=None):
    import tests.test_executor as te
    from sqlglot import transpile

    sqls = list(te.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
    if base is not None and len(base) != len(sqls):
        return False
    records = []
    for i, (_, sql, optimized) in enumerate(sqls):
        if only is not None and i not in only:
            records.append(None)
            continue
        if stop is not None and stop():
            return False
        _tick()
        if window is not None:
            window(i)
        try:
            fp = _h(transpile(sql, write="duckdb")[0], transpile(optimized, write="duckdb")[0])
        finally:
            if window is not None:
                window(None)
        if base is not None and (base[i] is None or fp != base[i]):
            if log:
                log(f"replay: optimized tpch query {i} differs")
            return False
        records.append(fp)
    return records if base is None else True


def run_one(test_id: str, base, skip_exec_ok: bool = False, log=None, only=None, window=None, stop=None,
            limits=None, strict_trees: bool = False):
    if test_id == T_OPT_TPCH:
        return _opt_tpch(base, log, only, window, stop)
    return _exec_test(test_id == T_TPCH, base, skip_exec_ok, log, only, window, stop, limits, strict_trees)


def check_limited(test_ids: list[str], base: dict, skip_exec_ok: bool, only: dict, pstate: dict,
                  log=None) -> bool:
    """Reach-limited replay of `test_ids` on a patched tree whose changed expression is probed.

    `only[test id]` lists the queries whose processing runs the changed expression's lines at
    the pinned commit (reach_of); no other query is run. True only if all of this holds:
    importing the tree and the tests never evaluates the probed expression; every listed query
    reproduces its baseline; the probed expression was evaluated at least once, and its old
    and its new value were identical at every evaluation. Then every evaluation left the
    program in the state the pinned code leaves it in, so the queries that were not run, which
    never reach the expression, do what they do at the pinned commit."""
    _arm()
    try:
        _tick(30.0)
        import tests.test_executor  # noqa: F401

        if pstate["n"]:
            if log:
                log("reach: the changed expression runs at import")
            return False
        for tid in test_ids:
            b = base.get("tests", {}).get(tid)
            if b is None:
                return False
            ok = run_one(tid, b, skip_exec_ok, log, only=set(only.get(tid, ())), stop=lambda: pstate["i"])
            if pstate["i"]:
                if log:
                    log(f"reach: old and new value differ (evaluation {pstate['n']})")
                return False
            if ok is not True:
                return False
        if not pstate["n"]:
            if log:
                log("reach: the changed expression never ran")
            return False
        if log:
            log(f"reach: {pstate['n']} evaluations, none differ")
        return True
    except BaseException as exc:  # noqa: BLE001 - any failure means "not proven"
        if log:
            log(f"reach: raised {type(exc).__name__}")
        return False
    finally:
        _disarm()


def reach_of(base: dict, path: str, lo: int, hi: int, test_ids: list[str]) -> dict | None:
    """{test id: queries whose processing runs any of lines lo..hi of `path` at the pinned
    commit}, from the build-time reach map. None when the map cannot vouch for it: one of the
    lines also runs outside the per-query windows (import, set-up), or the set of queries that
    run it differs between the recorded arrangements (it depends on what ran before)."""
    rm = base.get("reach")
    if not rm or lo <= 0 or hi < lo or not path.startswith("sqlglot/"):
        return None

    def hit(ranges) -> bool:
        return any(a <= hi and lo <= b for a, b in ranges)

    if hit(rm.get("outside", {}).get(path, ())):
        return None
    out = {}
    for tid in test_ids:
        n = rm.get("counts", {}).get(tid)
        if n is None or tid not in rm.get("lines", {}) or n != len(base.get("tests", {}).get(tid) or ()):
            return None
        if hit(rm.get("unstable", {}).get(tid, {}).get(path, ())):
            return None
        mask = 0
        for m, ranges in rm["lines"][tid].get(path, {}).items():
            if hit(ranges):
                mask |= int(m, 16)
        out[tid] = [i for i in range(n) if mask >> i & 1]
    return out


def check(test_ids: list[str], base: dict, skip_exec_ok: bool, log=None, only: dict | None = None,
          timed: bool = False) -> dict[str, bool]:
    """{test id: True if provably the pinned verdict}. Runs inside a process whose cwd and
    sys.path[0] are the patched tree, with SKIP_INTEGRATION=0."""
    out: dict[str, bool] = {}
    _arm()
    try:
        for tid in test_ids:
            b = base.get("tests", {}).get(tid)
            if b is None:
                out[tid] = False
                continue
            try:
                # (`only` is used by the audit of reach-limited replays alone, never for a verdict)
                out[tid] = run_one(tid, b, skip_exec_ok, log,
                                   only=None if only is None else set(only.get(tid, ())),
                                   limits=cpu_limits(base, tid) if timed else None) is True
            except BaseException as exc:  # noqa: BLE001 - any failure means "not the same"
                if log:
                    log(f"replay: {tid.rsplit('::', 1)[-1]} raised {type(exc).__name__}")
                out[tid] = False
    finally:
        _disarm()
    return out


def exec_lines_touched(base: dict, changes) -> bool:
    """True unless every change is a function-body/lambda edit in sqlglot/ on lines that are
    never executed outside the parse/optimize phase of the replayed tests (EXEC_LINES)."""
    ex = base.get("exec_lines")
    if not ex or not changes:
        return True
    for path, kind, lo, hi in changes:
        if kind not in ("body", "lambda") or not path.startswith("sqlglot/") or lo <= 0 or hi < lo:
            return True
        for a, b in ex.get(path, ()):
            if a <= hi and lo <= b:
                return True
    return False


# ---------------------------------------------------------------- isolated replay (v7)
#
# For a change confined to function-body lines L, only the TPC-DS queries whose processing
# runs L at the pinned commit can behave differently, PROVIDED such a query leaves nothing
# behind that another query reads. The isolated replay runs each of those queries alone, in a
# fork made right after set-up (the state the reach map's "fresh" arrangement was recorded
# from, and the state every pool worker of the real test starts from), and requires of each:
# its baseline result, no time-out, and the same signature of what the run leaves behind as
# the pinned code left when the image was built:
#   (1) every object created during the run and still alive after a collection: its type and
#       its content two levels deep (atoms by value);
#   (2) every change the run made to the module-level and class-level state of sqlglot's
#       modules: containers in full (nested ones too), atoms by value, function defaults and
#       closure cells, attributes of module-level instances, lru_cache statistics;
#   (3) the TPC tables: the same row objects in the same order, the same attributes;
#   (4) interpreter-wide state: recursion limit, environment, decimal context, random state,
#       and which object every global of every other imported module is bound to.
# A query that reads a memoised result runs the memoised code itself when it runs alone, so
# it is among the queries that reach L: caches need no signature. Anything that does not
# match is "not proven", and the caller does what v6 does.

import enum as _enum
import functools as _functools

_ATOMS = frozenset((str, int, float, bool, type(None), bytes, complex))
_FUNC = type(lambda: 0)
_LRU = type(_functools.lru_cache(maxsize=None)(lambda: 0))
_MODULE = type(sys)
ISO_MAX_SURVIVORS = 200000
DEEP_MAX_LEN = 2000
DEEP_LEVELS = 40


def cpu_limits(base: dict, tid: str, mode: str | None = None) -> list | None:
    """CPU seconds allowed to one step of each query of `tid`, from the times of the isolated
    runs at image build (trees + executor, or one `mode`); None when nothing was recorded."""
    t = (base.get("iso") or {}).get("t", {}).get(tid)
    n = len(base.get("tests", {}).get(tid) or ())
    if not t or len(t.get("trees") or ()) != n or len(t.get("full") or ()) != n:
        return None
    out = []
    for a, b in zip(t["trees"], t["full"]):
        if a is None or b is None:
            out.append(None)
        else:
            x = a + b if mode is None else (a if mode == "trees" else b)
            out.append(CPU_LIMIT_FACTOR * x + CPU_LIMIT_SLACK_S)
    return out


def _d(v, depth: int, deep: int = 0) -> str:
    """Canonical descriptor of a value: atoms by value; containers, functions and instances of
    sqlglot's classes `depth` levels deep; everything else by type. No ids, no hash order.
    With `deep`, builtin containers nested below `depth` are still described in full, up to
    `deep` further levels and DEEP_MAX_LEN entries each (tries, nested mappings)."""
    t = type(v)
    if t in _ATOMS:
        if t is int and not -2147483648 <= v <= 2147483647:
            return "int:big"  # ids and hashes: they differ from process to process
        return f"{t.__name__}:{v!r}"
    if t is tuple or t is list or t is dict or t is set or t is frozenset:
        if depth <= 0:
            if deep <= 0 or len(v) > DEEP_MAX_LEN:
                return f"{t.__name__}#{len(v)}"
            deep -= 1
        else:
            depth -= 1
        if t is dict:
            return "dict{" + ",".join(sorted([_d(k, depth, deep) + "=" + _d(x, depth, deep) for k, x in v.items()])) + "}"
        if t is tuple or t is list:
            return f"{t.__name__}[" + ",".join([_d(x, depth, deep) for x in v]) + "]"
        return f"{t.__name__}{{" + ",".join(sorted([_d(x, depth, deep) for x in v])) + "}"
    if isinstance(v, type):
        return f"class:{v.__module__}.{v.__qualname__}"
    mod = str(getattr(t, "__module__", "?"))
    name = f"{mod}.{t.__qualname__}"
    if t is _FUNC:
        s = f"fn:{v.__module__}.{v.__qualname__}"
        if depth > 0:
            if v.__defaults__:
                s += "|d=" + _d(v.__defaults__, depth, deep)
            if v.__kwdefaults__:
                s += "|k=" + _d(v.__kwdefaults__, depth, deep)
            if v.__closure__:
                cells = []
                for c in v.__closure__:
                    try:
                        cells.append(_d(c.cell_contents, depth - 1, deep))
                    except ValueError:
                        cells.append("empty")
                s += "|c=" + ",".join(cells)
        return s
    if t is staticmethod or t is classmethod:
        return f"{t.__name__}:" + _d(v.__func__, depth, deep)
    if t is property:
        return "property:" + _d(v.fget, depth, deep)
    if t is _LRU:
        try:
            return f"lru:{getattr(v, '__qualname__', '?')}:{tuple(v.cache_info())!r}"
        except Exception:  # noqa: BLE001
            return name
    if isinstance(v, _enum.Enum):
        return f"enum:{name}.{v.name}"
    if t is _MODULE:
        return f"module:{getattr(v, '__name__', '?')}"
    if depth > 0 and (mod.startswith("sqlglot") or mod.startswith("tests")):
        d = getattr(v, "__dict__", None)
        if type(d) is dict:
            rows = name == "sqlglot.executor.table.Table"  # its rows are compared by identity (_tables_fp)
            return name + "{" + ",".join(sorted([
                f"{k}=#{len(x)}" if rows and k == "rows" and type(x) is list else f"{k}={_d(x, depth - 1, deep)}"
                for k, x in d.items()])) + "}"
    return name


def _ours(name) -> bool:
    return isinstance(name, str) and (name == "sqlglot" or name.startswith("sqlglot.")
                                      or name == "tests" or name.startswith("tests."))


def _static_parts() -> dict[str, str]:
    """{name path: descriptor} of the module-level and class-level state of sqlglot's modules
    and of the tests' modules (dunder names apart)."""
    out: dict[str, str] = {}
    seen: set[int] = set()

    def put(path: str, v) -> None:
        if path in out:
            k = 2
            while f"{path}#{k}" in out:
                k += 1
            path = f"{path}#{k}"
        out[path] = _d(v, 2, DEEP_LEVELS)

    def walk(owner: str, d) -> None:
        for k in sorted(d):
            if k[:2] == "__" and k[-2:] == "__":
                continue
            v = d[k]
            put(f"{owner}.{k}", v)
            if isinstance(v, type) and id(v) not in seen and _ours(getattr(v, "__module__", None)):
                seen.add(id(v))
                walk(f"{v.__module__}.{v.__qualname__}", vars(v))

    for name in sorted(sys.modules):
        if _ours(name):
            d = getattr(sys.modules[name], "__dict__", None)
            if type(d) is dict:
                walk(name, dict(d))
    return out


def _tables_fp(tables) -> tuple:
    """The identity of every table and of every row object, in order: rows are tuples of atoms,
    so the same row objects in the same order are the same rows. Compared inside one family
    of forks only (before and after a query)."""
    if not isinstance(tables, dict):
        return ("?", repr(type(tables)))
    return tuple((name, id(t), id(t.rows), len(t.rows), hash(tuple(map(id, t.rows))))
                 for name, t in tables.items())


def _globals_fp() -> dict:
    """Interpreter-wide state outside sqlglot that a later query could read."""
    import decimal
    import random

    return {"recursionlimit": sys.getrecursionlimit(), "environ": dict(os.environ),
            "decimal": repr(decimal.getcontext()), "random": hash(random.getstate()),
            "path": tuple(sys.path), "cwd": os.getcwd()}


def _globals_delta(g0: dict) -> list[str]:
    g1 = _globals_fp()
    out = []
    for k in sorted(g1):
        if g1[k] != g0.get(k):
            if k == "environ":
                keys = sorted(x for x in set(g1[k]) | set(g0[k]) if g1[k].get(x) != g0[k].get(x))
                out.append("global:environ=" + ",".join(f"{x}={g1[k].get(x)!r}" for x in keys))
            elif k in ("recursionlimit", "decimal"):
                out.append(f"global:{k}={g1[k]}")
            else:
                out.append(f"global:{k}=changed")
    return out


def _modules_fp() -> dict:
    """{module: (names, ids of their values)} for every imported module outside sqlglot and the
    tests: which global of which other module (standard library, third party, the selector)
    is bound to which object. Compared inside one family of forks only."""
    out = {}
    for name, mod in list(sys.modules.items()):
        if not _ours(name):
            d = getattr(mod, "__dict__", None)
            if type(d) is dict:
                out[name] = (tuple(d), tuple(map(id, d.values())))
    return out


def _modules_delta(m0: dict) -> list[str]:
    """Names of the other modules' globals that were bound, rebound or removed, and new modules."""
    out = []
    m1 = _modules_fp()
    for name, (keys, ids) in m1.items():
        old = m0.get(name)
        if old is None:
            out.append(f"module+:{name}")
        elif old != (keys, ids):
            a, b = dict(zip(*old)), dict(zip(keys, ids))
            out.append(f"module:{name}:" + ",".join(sorted(str(k) for k in set(a) | set(b) if a.get(k) != b.get(k))))
    out += [f"module-:{name}" for name in m0 if name not in m1]
    return out


def _tables_desc(tables) -> list[str]:
    """Every table's attributes but its rows (columns, range, readers and what they hold)."""
    return [f"table:{name}=" + ",".join(sorted([f"{k}={_d(x, 2)}" for k, x in vars(t).items() if k != "rows"]))
            for name, t in tables.items()]


def _sig(surv: str, delta: dict, skip) -> str:
    """One signature from the survivors' digest and the changed static paths, leaving out the
    paths whose value depends on the hash seed at the pinned commit (found at image build)."""
    m = hashlib.sha1(surv.encode())
    for path in sorted(delta):
        if path not in skip:
            m.update(f"{path}={delta[path]}\0".encode("utf-8", "surrogatepass"))
    return m.hexdigest()[:20]


def _leftover(static0: dict, tables, tables0, dump: str | None = None, globals0: dict | None = None,
              modules0: dict | None = None):
    """What this process did since gc.freeze() (see the head of this section): (digest of the
    surviving objects, {changed static path: digest of its new value}), or None."""
    import gc

    gc.collect()
    objs = gc.get_objects()
    if len(objs) > ISO_MAX_SURVIVORS:
        return None
    if tables0 is not None and _tables_fp(tables) != tables0:
        return None
    # the namespaces of modules and classes created during the run are described by name path
    # below (the static part), where seed-dependent and path-dependent names can be left out
    ns: set[int] = set()
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None)
        if type(d) is dict:
            ns.add(id(d))
    for o in objs:
        if isinstance(o, type):
            for r in gc.get_referents(o):
                if type(r) is dict:
                    ns.add(id(r))
    parts = sorted(["namespace" if type(o) is dict and id(o) in ns else _d(o, 2) for o in objs])
    del objs
    if tables0 is not None:
        parts += _tables_desc(tables)
    if globals0 is not None:
        parts += _globals_delta(globals0)
    if modules0 is not None:
        parts += sorted(_modules_delta(modules0))
    delta: dict[str, str] = {}
    full: list[str] = []
    now = _static_parts()
    for path, desc in now.items():
        if static0.get(path) != hash(desc):
            delta[path] = hashlib.sha1(desc.encode("utf-8", "surrogatepass")).hexdigest()[:12]
            if dump:
                full.append(f"{path}={desc}")
    for path in static0:
        if path not in now:
            delta[path] = "<gone>"
    m = hashlib.sha1()
    m.update(f"{len(parts)}".encode())
    for p in parts:
        m.update(p.encode("utf-8", "surrogatepass"))
        m.update(b"\0")
    if dump:
        with open(dump, "w", encoding="utf-8", errors="surrogatepass") as fh:
            fh.write("\n".join(parts) + "\n--- static delta\n" + "\n".join(sorted(full)) + "\n")
    return m.hexdigest()[:20], delta


def _iso_one(tid: str, i: int, b, skip_exec_ok: bool, limits, static0: dict, tables, tables0, skip,
             globals0=None, modules0=None) -> list:
    """In a fork of its own: query i alone. [reproduces its baseline, signature, CPU s, notes];
    with skip None (image build) the signature is the raw pair of _leftover."""
    import gc
    import time

    gc.collect()
    gc.freeze()  # everything alive now is the post-set-up state: collections and get_objects skip it
    notes: list[str] = []
    t0 = time.process_time()
    _arm()
    try:
        ok = run_one(tid, b, skip_exec_ok, notes.append, only={i}, limits=limits, strict_trees=True) is True
    except BaseException as exc:  # noqa: BLE001 - any failure means "not proven"
        ok = False
        notes.append(f"raised {type(exc).__name__}")
    finally:
        _disarm()
    dt = time.process_time() - t0
    sig = None
    if ok:
        dump = os.environ.get("BR_ISO_DUMP")
        sig = _leftover(static0, tables, tables0,
                        os.path.join(dump, f"q{i}-{'t' if skip_exec_ok else 'f'}-{os.getppid()}.txt") if dump else None,
                        globals0, modules0)
        if sig is not None and skip is not None:
            sig = _sig(sig[0], sig[1], skip)
    return [ok, sig, round(dt, 4), notes]


def iso_run(base: dict, skip_exec_ok: bool, queries, log=None, record: bool = False):
    """Isolated replay of the given TPC-DS queries on the tree this process imports from.

    record=False: "same" when every query, alone, reproduces its baseline and leaves the
    signature recorded at the pinned commit; "diff" when a query does not reproduce its
    baseline (a result differs, an exception, a time-out); "unknown" otherwise (a signature
    is missing or differs: nothing is known about the queries that were not run).
    record=True (image build, pinned tree): {query: [reproduced, signature, CPU seconds]}."""
    tid = T_TPCDS
    b = base.get("tests", {}).get(tid)
    mode = "trees" if skip_exec_ok else "full"
    sigs = limits = skip = None
    if b is None:
        return "unknown"
    if not record:
        skip = frozenset((base.get("iso") or {}).get("skip") or ())
        sigs = (base.get("iso") or {}).get("sig", {}).get(tid, {}).get(mode)
        if not sigs or len(sigs) != len(b) or not queries or any(
                not (0 <= i < len(b)) or sigs[i] is None for i in queries):
            if log:
                log("iso: no recorded signature for a reached query")
            return "unknown"
        limits = cpu_limits(base, tid, mode)
    _arm()
    try:
        _tick(30.0)
        import tests.test_executor as te

        tables = tables0 = None
        if not skip_exec_ok:
            if "tpcds" not in _LOADED:
                _load_tables(te, False)
            tables = te.TestExecutor.tpcds_tables
            tables0 = _tables_fp(tables)
        static0 = {path: hash(desc) for path, desc in _static_parts().items()}
        globals0 = _globals_fp()
        modules0 = _modules_fp()
    except BaseException as exc:  # noqa: BLE001
        if log:
            log(f"iso: set-up raised {type(exc).__name__}")
        return "diff"
    finally:
        _disarm()
    out: dict = {}
    for i in queries:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                os.close(r)
                res = _iso_one(tid, i, b, skip_exec_ok, limits, static0, tables, tables0, skip, globals0, modules0)
                with os.fdopen(w, "w") as fh:
                    json.dump(res, fh)
                code = 0
            finally:
                os._exit(code)
        os.close(w)
        with os.fdopen(r) as fh:
            text = fh.read()
        _, status = os.waitpid(pid, 0)
        res = None
        if status == 0 and text:
            try:
                res = json.loads(text)
            except ValueError:
                res = None
        if record:
            out[i] = res[:3] if res else None
            continue
        if res is None or res[0] is not True:
            if log:
                log(f"iso: query {i} does not reproduce its baseline"
                    + (f" ({'; '.join(res[3])[:80]})" if res and res[3] else ""))
            return "diff"
        if res[1] is None or res[1] != sigs[i]:
            if log:
                log(f"iso: query {i} leaves a different state behind")
            return "unknown"
    if record:
        return out
    if log:
        log(f"iso: {len(queries)} queries alone: baseline results, recorded signatures")
    return "same"


def _iso_build_child(repo: str, out_path: str, base_path: str, mode: str, tmpdir: str) -> None:
    """Record the isolated run of every TPC-DS query at the pinned commit, through the code and
    the kind of process the selector uses (warm parent, overlay, forked replay child)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import warm
    from runner import build_overlay

    with open(base_path, encoding="utf-8") as fh:
        base = json.load(fh)
    n = len(base["tests"][T_TPCDS])
    overlay = os.path.join(tmpdir, f"ov-{mode}-{os.getpid()}")
    build_overlay(repo, overlay, {})
    res, notes = warm.replay_check(overlay, [T_TPCDS], base, mode == "trees", tmpdir, 1200.0,
                                   iso=list(range(n)), record=True)
    got = res.get("iso")
    if not isinstance(got, dict):
        raise RuntimeError(f"iso recording: no result ({notes})")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(got, fh)


def build_iso(repo: str, base_path: str, tmpdir: str) -> dict:
    """The "iso" section of the baseline file: per TPC-DS query and mode (trees / with the
    executor), the leftover signature of its isolated run (None unless two processes with
    different hash seeds reproduce the baseline and agree on it) and its CPU seconds."""
    import subprocess

    with open(base_path, encoding="utf-8") as fh:
        n = len(json.load(fh)["tests"][T_TPCDS])
    out: dict = {"t": {T_TPCDS: {}}, "sig": {T_TPCDS: {}}, "skip": []}
    recs: dict = {}
    for mode in ("trees", "full"):
        procs = []
        for seed in ("1", "777"):
            op = os.path.join(tmpdir, f"iso-{mode}-{seed}.json")
            env = dict(os.environ, PYTHONHASHSEED=seed, SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")
            procs.append((subprocess.Popen([sys.executable, os.path.abspath(__file__), "_iso", repo, op,
                                            base_path, mode, tmpdir], env=env), op))
        runs = []
        for proc, op in procs:
            if proc.wait() != 0:
                raise RuntimeError("iso recording failed")
            with open(op, encoding="utf-8") as fh:
                runs.append(json.load(fh))
        recs[mode] = runs
    # static paths whose value differs between the two hash seeds in any query's run: their
    # value is arbitrary at the pinned commit too (e.g. a loop variable over a set left in a
    # class body), so no signature includes them
    skip: set[str] = set()
    for mode, runs in recs.items():
        for i in range(n):
            x, y = runs[0].get(str(i)), runs[1].get(str(i))
            if x and y and x[0] is True and y[0] is True and x[1] and y[1]:
                dx, dy = x[1][1], y[1][1]
                skip.update(p for p in set(dx) | set(dy) if dx.get(p) != dy.get(p))
    out["skip"] = sorted(skip)
    for mode, runs in recs.items():
        sig, secs = [], []
        for i in range(n):
            x, y = runs[0].get(str(i)), runs[1].get(str(i))
            good = bool(x and y and x[0] is True and y[0] is True and x[1] and y[1])
            sx = _sig(x[1][0], x[1][1], skip) if good else None
            sig.append(sx if good and sx == _sig(y[1][0], y[1][1], skip) else None)
            secs.append(round(max(x[2], y[2]), 4) if x and y else None)
        out["sig"][T_TPCDS][mode] = sig
        out["t"][T_TPCDS][mode] = secs
        print(f"replay iso: {mode}: {sum(s is not None for s in sig)}/{n} queries with a stable signature, "
              f"{sum(t for t in secs if t):.2f}s cpu alone", file=sys.stderr)
    print(f"replay iso: {len(skip)} seed-dependent static paths left out: {sorted(skip)[:8]}", file=sys.stderr)
    return out


# ---------------------------------------------------------------- build time

def _suites(te):
    return (
        (False, "optimizer/tpc-ds/tpc-ds.sql", te.TPCDS_SCHEMA),
        (True, "optimizer/tpc-h/tpc-h.sql", te.TPCH_SCHEMA),
    )


def _blobs_child(repo: str, out_path: str) -> None:
    """Pickles of every parsed query and of every optimized tree handed to the planner."""
    sys.path.insert(0, repo)
    os.chdir(repo)
    os.environ["SKIP_INTEGRATION"] = "0"
    import tests.test_executor as te
    import sqlglot.executor as ex
    from sqlglot import parse_one
    from sqlglot.executor.table import Table

    out = {}
    real_plan = ex.Plan
    for tpch, rel, schema in _suites(te):
        rows = []
        te.initializer(schema, {name: Table(columns=tuple(cols)) for name, cols in schema.items()})
        for meta, sql, _ in te.load_sql_fixture_pairs(rel):
            parsed = pickle.dumps(parse_one(sql))
            optimized = None
            if meta.get("execute"):
                ex.Plan = _capture_plan
                try:
                    te.mp_execute(pickle.loads(parsed), meta)
                except _Captured as cap:
                    optimized = pickle.dumps(cap.args[0])
                finally:
                    ex.Plan = real_plan
            rows.append((parsed, optimized))
        out[tpch] = rows
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)


def _trace_child(repo: str, out_path: str, variant: str, blobs_path: str) -> None:
    """Record every sqlglot line run outside the parse/optimize phase of the execute tests.

    variant "stub": parse and optimize never run in this process (their results are
    unpickled), so anything built lazily is built, and recorded, by the later phase.
    variant "pause": the real pipeline, with recording paused inside parse_one and optimize.
    Recording starts before sqlglot is imported: import-time lines are all included."""
    root = os.path.realpath(repo) + os.sep
    mon = sys.monitoring
    tool = 3
    lines: dict[str, set[int]] = {}
    cache: dict[str, str | None] = {}

    def rel(p):
        if p not in cache:
            if not p or p.startswith("<"):
                cache[p] = None
            else:
                r = os.path.realpath(p)
                cache[p] = r[len(root):] if r.startswith(root) else None
        return cache[p]

    def on_line(code, line):
        r = rel(code.co_filename)
        if r is not None:
            lines.setdefault(r, set()).add(line)
        return mon.DISABLE

    mon.use_tool_id(tool, "replay")
    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.set_events(tool, mon.events.LINE)

    def pause():
        mon.set_events(tool, 0)

    def resume():
        mon.set_events(tool, mon.events.LINE)

    sys.path.insert(0, repo)
    os.chdir(repo)
    os.environ["SKIP_INTEGRATION"] = "0"
    with open(blobs_path, "rb") as fh:
        blobs = pickle.load(fh)
    import tests.test_executor as te
    import sqlglot.executor as ex

    cls = te.TestExecutor
    cls.setUpClass()
    real_optimize = ex.optimize
    for tpch, relpath, schema in _suites(te):
        tables = cls.tpch_tables if tpch else cls.tpcds_tables
        te.initializer(schema, tables)
        for i, (meta, sql, _) in enumerate(te.load_sql_fixture_pairs(relpath)):
            parsed_blob, optimized_blob = blobs[tpch][i]
            if variant == "stub":
                expression = pickle.loads(parsed_blob)
                pickle.dumps(expression)
                if optimized_blob is not None:
                    ex.optimize = lambda *a, _b=optimized_blob, **k: pickle.loads(_b)
            else:
                from sqlglot import parse_one

                pause()
                parsed = parse_one(sql)
                resume()
                expression = pickle.loads(pickle.dumps(parsed))

                def paused_optimize(*a, **k):
                    pause()
                    try:
                        return real_optimize(*a, **k)
                    finally:
                        resume()

                ex.optimize = paused_optimize
            try:
                table = te.mp_execute(expression, meta)
            finally:
                ex.optimize = real_optimize
            if table is not None:
                pickle.loads(pickle.dumps(table))
    cls.tearDownClass()
    mon.set_events(tool, 0)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({f: sorted(ls) for f, ls in lines.items() if f.startswith("sqlglot/")}, fh)


def _reach_child(repo: str, out_path: str) -> None:
    """Which query's processing runs which sqlglot line, at the pinned commit.

    The replayed tests are run through run_one() itself, with every line recorded into the
    window of the query being processed (parse, pickle, transpile, optimize, execute), or into
    "outside" (import of sqlglot and the tests, set-up, everything between windows). Two
    arrangements: every query alone in a fork made before any query ran ("fresh": what is
    built lazily or cached is built by each query that needs it), all queries in order in this
    process ("seq"), and every query alone in a fork made after all of them ran ("warm").
    Recording starts before sqlglot is imported."""
    root = os.path.realpath(repo) + os.sep
    mon = sys.monitoring
    tool = 3
    cache: dict[str, str | None] = {}
    buckets: dict = {None: {}}
    cur = [buckets[None]]
    tid_now = [None]

    def rel(p):
        if p not in cache:
            if not p or p.startswith("<"):
                cache[p] = None
            else:
                r = os.path.realpath(p)
                cache[p] = r[len(root):] if r.startswith(root) else None
        return cache[p]

    def on_line(code, line):
        r = rel(code.co_filename)
        if r is not None:
            cur[0].setdefault(r, set()).add(line)
        return mon.DISABLE

    mon.use_tool_id(tool, "reach")
    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.set_events(tool, mon.events.LINE)

    def window(i):
        cur[0] = buckets.setdefault(None if i is None else (tid_now[0], i), {})
        mon.restart_events()

    sys.path.insert(0, repo)
    os.chdir(repo)
    os.environ["SKIP_INTEGRATION"] = "0"
    import tests.test_executor as te

    te.TestExecutor.setUpClass()
    _LOADED.update(("tpch", "tpcds"))
    outside = buckets[None]
    counts = {tid: len(list(te.load_sql_fixture_pairs(
        "optimizer/tpc-ds/tpc-ds.sql" if tid == T_TPCDS else "optimizer/tpc-h/tpc-h.sql"))) for tid in REPLAYABLE}
    fresh: dict = {}
    warmed: dict = {}
    for view in (fresh, None, warmed):
      if view is None:
        for tid in REPLAYABLE:
            tid_now[0] = tid
            if len(run_one(tid, None, window=window)) != counts[tid]:
                raise RuntimeError("reach: query count")
        window(None)
        seq = {k: v for k, v in buckets.items() if k is not None}
        continue
      for tid in REPLAYABLE:
        for i in range(counts[tid]):
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                code = 1
                try:
                    os.close(r)
                    buckets.clear()
                    buckets[None] = {}
                    cur[0] = buckets[None]
                    tid_now[0] = tid
                    mon.restart_events()
                    run_one(tid, None, only={i}, window=window)
                    mon.set_events(tool, 0)
                    doc = {"in": {f: sorted(ls) for f, ls in buckets.get((tid, i), {}).items()},
                           "out": {f: sorted(ls) for f, ls in buckets[None].items()}}
                    with os.fdopen(w, "w") as fh:
                        json.dump(doc, fh)
                    code = 0
                finally:
                    os._exit(code)
            os.close(w)
            with os.fdopen(r) as fh:
                text = fh.read()
            _, status = os.waitpid(pid, 0)
            if status != 0 or not text:
                raise RuntimeError(f"reach trace failed: {tid} query {i}")
            doc = json.loads(text)
            view[(tid, i)] = doc["in"]
            for f, ls in doc["out"].items():
                outside.setdefault(f, set()).update(ls)
    mon.set_events(tool, 0)

    def masks(view):
        out: dict = {}
        for (tid, i), files in view.items():
            for f, ls in files.items():
                if f.startswith("sqlglot/"):
                    d = out.setdefault(tid, {}).setdefault(f, {})
                    for ln in ls:
                        d[ln] = d.get(ln, 0) | (1 << i)
        return {t: {f: {str(ln): format(m, "x") for ln, m in d.items()} for f, d in fs.items()}
                for t, fs in out.items()}

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"counts": counts,
                   "outside": {f: sorted(ls) for f, ls in outside.items() if f.startswith("sqlglot/")},
                   "masks": {"seq": masks(seq), "fresh": masks(fresh), "warm": masks(warmed)}}, fh)
    os._exit(0)  # duckdb connections were inherited by forks: skip interpreter teardown


def _merge_reach(runs: list[dict]) -> dict:
    """One reach map from several recordings (arrangements x hash seeds): a line's queries are
    the union, and a line whose queries differ between any two recordings is "unstable"."""
    counts = runs[0]["counts"]
    outside: dict[str, set[int]] = {}
    for r in runs:
        if r["counts"] != counts:
            raise RuntimeError("reach: query counts differ between runs")
        for f, ls in r["outside"].items():
            outside.setdefault(f, set()).update(ls)
    views = [r["masks"][k] for r in runs for k in ("seq", "fresh", "warm")]
    lines: dict = {}
    unstable: dict = {}
    for tid in counts:
        keys = set()
        for v in views:
            for f, d in v.get(tid, {}).items():
                keys.update((f, ln) for ln in d)
        by_file: dict = {}
        bad: dict = {}
        for f, ln in keys:
            ints = [int(v.get(tid, {}).get(f, {}).get(ln, "0"), 16) for v in views]
            union = 0
            for x in ints:
                union |= x
            if any(x != ints[0] for x in ints):
                bad.setdefault(f, []).append(int(ln))
            by_file.setdefault(f, {}).setdefault(format(union, "x"), []).append(int(ln))
        lines[tid] = {f: {m: _ranges(ls) for m, ls in sorted(d.items())} for f, d in sorted(by_file.items())}
        unstable[tid] = {f: _ranges(ls) for f, ls in sorted(bad.items())}
    return {"counts": counts, "outside": {f: _ranges(sorted(ls)) for f, ls in sorted(outside.items())},
            "lines": lines, "unstable": unstable}


def start_reach(repo: str, tmpdir: str) -> list:
    """Start the reach recordings, one process per hash seed, and return them: they run
    beside the rest of the build (the build has 4 cores and a time limit)."""
    import subprocess

    procs = []
    for seed in ("1", "777"):
        rp = os.path.join(tmpdir, f"reach-{seed}.json")
        env = dict(os.environ, PYTHONHASHSEED=seed, SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")
        procs.append((subprocess.Popen([sys.executable, os.path.abspath(__file__), "_reach", repo, rp], env=env), rp))
    return procs


def finish_reach(procs: list) -> dict:
    runs = []
    for proc, rp in procs:
        if proc.wait() != 0:
            raise RuntimeError("reach recording failed")
        with open(rp, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    doc = _merge_reach(runs)
    n_lines = sum(len(r) for t in doc["lines"].values() for d in t.values() for r in d.values())
    n_bad = sum(b - a + 1 for t in doc["unstable"].values() for rs in t.values() for a, b in rs)
    print(f"replay reach: {n_lines} line ranges, {n_bad} unstable lines, "
          f"{sum(b - a + 1 for rs in doc['outside'].values() for a, b in rs)} lines outside windows", file=sys.stderr)
    return doc


def build_reach(repo: str, tmpdir: str) -> dict:
    return finish_reach(start_reach(repo, tmpdir))


def _ranges(ls: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for n in sorted(set(ls)):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def _run_self(mode: str, *args: str, seed: str = "0") -> None:
    import subprocess

    env = dict(os.environ, PYTHONHASHSEED=seed, SKIP_INTEGRATION="0", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, os.path.abspath(__file__), mode, *args], check=True, env=env)


def _build_child(repo: str, out_path: str) -> None:
    sys.path.insert(0, repo)
    os.chdir(repo)
    os.environ["SKIP_INTEGRATION"] = "0"
    res = {}
    for tid in REPLAYABLE:
        res[tid] = run_one(tid, None)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh)


def _merge_stable(a, b):
    """Keep only values both runs agree on (None marks 'not reproducible')."""
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_merge_stable(x, y) for x, y in zip(a, b)]
    return a if a == b else None


def build(repo: str, out_path: str) -> None:
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="replay-build-")
    reach_procs = start_reach(repo, tmpdir) if hasattr(sys, "monitoring") else None
    runs = []
    for seed in ("1", "2", "777"):
        tmp_path = os.path.join(tmpdir, f"fp-{seed}.json")
        _run_self("_child", repo, tmp_path, seed=seed)
        with open(tmp_path, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    tests = {}
    for tid in REPLAYABLE:
        merged = runs[0][tid]
        for other in runs[1:]:
            merged = _merge_stable(merged, other[tid])
        tests[tid] = merged
    doc = {"python": sys.version.split()[0], "tests": tests}
    for tid, recs in tests.items():
        n = len(recs) if isinstance(recs, list) else 0
        bad = sum(1 for r in (recs or []) if isinstance(r, list) and any(x is None for x in r))
        print(f"replay baseline: {tid.rsplit('::', 1)[-1]}: {n} queries, {bad} not reproducible", file=sys.stderr)
    if hasattr(sys, "monitoring"):
        blobs = os.path.join(tmpdir, "blobs.pkl")
        _run_self("_blobs", repo, blobs)
        union: dict[str, set[int]] = {}
        for variant in ("stub", "pause"):
            tr = os.path.join(tmpdir, f"trace-{variant}.json")
            _run_self("_trace", repo, tr, variant, blobs)
            with open(tr, encoding="utf-8") as fh:
                for f, ls in json.load(fh).items():
                    union.setdefault(f, set()).update(ls)
        doc["exec_lines"] = {f: _ranges(sorted(ls)) for f, ls in sorted(union.items())}
        print(f"replay exec lines: {sum(len(v) for v in union.values())} lines in {len(union)} files", file=sys.stderr)
        doc["reach"] = finish_reach(reach_procs)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    try:
        doc["iso"] = build_iso(repo, out_path, tmpdir)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, separators=(",", ":"))
    except Exception as exc:  # noqa: BLE001 - without it the selector does what v6 does
        print(f"replay iso: recording failed ({exc!r}): no isolated replays", file=sys.stderr)
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1] == "_child":
        _build_child(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "_blobs":
        _blobs_child(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "_trace":
        _trace_child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == "_reach":
        _reach_child(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "reach":  # add the reach map to an existing baseline file
        import tempfile

        with open(sys.argv[3], encoding="utf-8") as fh:
            _doc = json.load(fh)
        _doc["reach"] = build_reach(os.path.abspath(sys.argv[2]), tempfile.mkdtemp(prefix="reach-build-"))
        with open(sys.argv[4], "w", encoding="utf-8") as fh:
            json.dump(_doc, fh, separators=(",", ":"))
    elif sys.argv[1] == "_iso":
        _iso_build_child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif sys.argv[1] == "iso":  # add the isolated-run records to an existing baseline file
        import tempfile

        with open(sys.argv[3], encoding="utf-8") as fh:
            _doc = json.load(fh)
        _doc["iso"] = build_iso(os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]),
                                tempfile.mkdtemp(prefix="iso-build-"))
        with open(sys.argv[4], "w", encoding="utf-8") as fh:
            json.dump(_doc, fh, separators=(",", ":"))
    elif sys.argv[1] == "build":
        build(os.path.abspath(sys.argv[2]), sys.argv[3])
