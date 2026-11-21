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
_tick_on = False


def _tick(seconds: float = STEP_LIMIT_S) -> None:
    """Re-arm the watchdog: a step that runs longer raises _Slow (a hang is a failure)."""
    if _tick_on:
        import signal

        signal.setitimer(signal.ITIMER_REAL, seconds)


def _arm() -> None:
    global _tick_on
    import signal

    def _late(signum, frame):
        raise _Slow()

    signal.signal(signal.SIGALRM, _late)
    _tick_on = True


def _disarm() -> None:
    global _tick_on
    if _tick_on:
        import signal

        signal.setitimer(signal.ITIMER_REAL, 0)
    _tick_on = False


def _capture_plan(expression, *a, **k):
    raise _Captured(expression)


def _rows_fp(table) -> str:
    cols = table.columns
    rows = table.rows
    return _h(_atom(list(cols)), *(_atom(r) for r in rows))


def _exec_test(tpch: bool, base: list | None, skip_exec_ok: bool, log=None):
    """Mirror of TestExecutor._mp_execute + subtestHelper for one TPC suite.

    Returns the per-query records when `base` is None (building), else True/False (same?)."""
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

    def real_tables():
        if state["tables"] is None:
            _tick(30.0)
            cls.setUpClass()  # the test's own loader (duckdb only defines lazy views here)
            state["tables"] = cls.tpch_tables if tpch else cls.tpcds_tables
        return state["tables"]

    records = []
    # the parent parses and pickles every query before any result is looked at
    blobs = []
    for meta, sql, _ in sqls:
        _tick()
        blobs.append(pickle.dumps(parse_one(sql)))
    for i, (meta, sql, _) in enumerate(sqls):
        _tick()
        expression = pickle.loads(blobs[i])
        if not meta.get("execute"):
            te.initializer(schema, {})
            if te.mp_execute(expression, meta) is not None:
                return False
            records.append(None)
            continue
        b = base[i] if base is not None else None
        if base is not None and (b is None or b[0] is None or b[1] is None or b[2] is None):
            return False  # no reproducible baseline for this query
        duck = _h(transpile(sql, write="duckdb")[0])
        if b is not None and duck != b[1]:
            if log:
                log(f"replay: query {i} duckdb sql differs")
            return False
        fp_opt = None
        if base is None or skip_exec_ok:
            # the optimized tree handed to the planner, captured by stopping there
            from sqlglot.executor.table import Table

            te.initializer(schema, {name: Table(columns=tuple(cols)) for name, cols in schema.items()})
            ex.Plan = _capture_plan
            try:
                te.mp_execute(pickle.loads(blobs[i]), meta)
                return False  # the planner was never reached: not the pinned pipeline
            except _Captured as cap:
                fp_opt = _h(canon(cap.args[0]))
            finally:
                ex.Plan = real_plan
            if b is not None and fp_opt == b[0]:
                records.append((fp_opt, duck, b[2]))
                continue
        te.initializer(schema, real_tables())
        _tick()
        table = te.mp_execute(expression, meta)
        if table is None:
            return False
        table = pickle.loads(pickle.dumps(table))
        res = _rows_fp(table)
        if b is not None and res != b[2]:
            if log:
                log(f"replay: query {i} rows differ")
            return False
        records.append((fp_opt, duck, res))
    return records if base is None else True


def _opt_tpch(base: list | None, log=None):
    import tests.test_executor as te
    from sqlglot import transpile

    sqls = list(te.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
    if base is not None and len(base) != len(sqls):
        return False
    records = []
    for i, (_, sql, optimized) in enumerate(sqls):
        _tick()
        fp = _h(transpile(sql, write="duckdb")[0], transpile(optimized, write="duckdb")[0])
        if base is not None and (base[i] is None or fp != base[i]):
            if log:
                log(f"replay: optimized tpch query {i} differs")
            return False
        records.append(fp)
    return records if base is None else True


def run_one(test_id: str, base, skip_exec_ok: bool = False, log=None):
    if test_id == T_OPT_TPCH:
        return _opt_tpch(base, log)
    return _exec_test(test_id == T_TPCH, base, skip_exec_ok, log)


def check(test_ids: list[str], base: dict, skip_exec_ok: bool, log=None) -> dict[str, bool]:
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
                out[tid] = run_one(tid, b, skip_exec_ok, log) is True
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
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1] == "_child":
        _build_child(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "_blobs":
        _blobs_child(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "_trace":
        _trace_child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == "build":
        build(os.path.abspath(sys.argv[2]), sys.argv[3])
