"""Fork-per-commit pytest runner from a warm parent.

The parent imports pytest and the heavy third-party modules sqlglot's tests use (pandas,
duckdb, numpy) once; each commit's tests run in a forked child, so they pay neither
interpreter start-up nor those imports. sqlglot itself is never imported in the parent:
every child imports it fresh from its own patched overlay.

In the child, outcomes are recorded from pytest reports (any failed report, subtests
included, fails a test; a collection error fails its whole file). Failure messages are not
needed, so unittest's diff building (TestOptimizer sets maxDiff = None and diffs huge SQL)
is short-circuited, and asserts run plain.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

_WARM = False


def _kill(pid: int) -> None:
    for target in (-pid, pid):
        try:
            os.kill(target, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def warm_up() -> None:
    global _WARM
    if _WARM:
        return
    import concurrent.futures  # noqa: F401
    import difflib  # noqa: F401
    import multiprocessing  # noqa: F401
    import unittest  # noqa: F401

    import pytest  # noqa: F401
    for name in ("_pytest.unittest", "_pytest.python", "_pytest.logging", "_pytest.capture",
                 "_pytest.subtests", "_pytest.skipping", "_pytest.junitxml", "pytest_timeout", "numpy", "pandas", "pandas.testing", "duckdb", "dateutil.parser", "pytz"):
        try:
            __import__(name)
        except Exception:  # noqa: BLE001
            pass
    _WARM = True


GC_EVERY = 40  # tests between explicit collections (automatic collection is off in children)


def relax_gc() -> None:
    """Children inherit a large heap that is never garbage (maps, pytest, pandas) and build
    another when they import sqlglot. Automatic collection walks both again and again, which
    is a tenth or more of a run's CPU. So: the inherited heap is frozen, automatic collection
    is switched off, and the runner collects explicitly between tests. Tests that call
    gc.collect() themselves behave as always."""
    import gc

    gc.freeze()
    gc.disable()


class _Collector:
    def __init__(self) -> None:
        self.failed: set[str] = set()
        self.seen: set[str] = set()
        self.errors: set[str] = set()
        self._done = 0

    def pytest_runtest_logreport(self, report):
        self.seen.add(report.nodeid)
        if report.failed:
            self.failed.add(report.nodeid)

    def pytest_runtest_logfinish(self, nodeid, location):
        self._done += 1
        if self._done % GC_EVERY == 0:
            import gc

            gc.collect()

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors.add(report.nodeid)


def _child(overlay: str, test_ids: list[str], out: str, per_test_timeout: float, probe_report: str | None = None,
           abort_on_infection: bool = False, import_mods: list[str] | None = None) -> None:
    import concurrent.futures as cf
    import multiprocessing as mp
    import unittest.case as uc

    import pytest

    # pass/fail only: skip building diffs for failure messages
    class _NoDiff:
        @staticmethod
        def ndiff(a, b, *args, **kw):
            return iter(())

        def __getattr__(self, name):
            import difflib
            return getattr(difflib, name)

    uc.difflib = _NoDiff()

    class _NoPformat:  # only unittest.case's own reference: the pprint module stays intact
        @staticmethod
        def pformat(obj, *a, **k):
            return "<obj>"

        def __getattr__(self, name):
            import pprint
            return getattr(pprint, name)

    uc.pprint = _NoPformat()
    relax_gc()
    orig_init = cf.ProcessPoolExecutor.__init__

    def capped(self, max_workers=None, mp_context=None, *a, **k):
        orig_init(self, 1, mp.get_context("fork"), *a, **k)

    cf.ProcessPoolExecutor.__init__ = capped
    if probe_report:
        import probe

        probe.install(probe_report, abort_on_infection)
    os.chdir(overlay)
    sys.path.insert(0, overlay)
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    col = _Collector()
    args = ["-q", "-p", "no:cacheprovider", "-p", "no:xdist", "--assert=plain", "--tb=no",
            f"--timeout={per_test_timeout:.0f}", "--rootdir", overlay, "-o", "console_output_style=classic",
            *test_ids]
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    imports = None
    try:
        pytest.main(args, plugins=[col])
        if import_mods:
            # the import check (see check_imports), done here to save a fork and a second
            # import of sqlglot: after the tests, so it cannot change what they did
            import importlib

            imports = {}
            for m in import_mods:
                try:
                    importlib.import_module(m)
                    imports[m] = True
                except BaseException:  # noqa: BLE001
                    imports[m] = False
    finally:
        with open(out, "w") as fh:
            json.dump({"failed": sorted(col.failed), "seen": sorted(col.seen), "errors": sorted(col.errors),
                       "imports": imports}, fh)


def run(overlay: str, test_ids: list[str], workdir: str, timeout_s: float, per_test_timeout: float,
        probe_report: str | None = None, abort_on_infection: bool = False,
        import_mods: list[str] | None = None) -> dict:
    """Run test_ids against `overlay` in a forked child. Returns failed/seen/errors sets,
    timed_out flag and wall seconds."""
    warm_up()
    out = os.path.join(workdir, f"run-{os.getpid()}-{time.monotonic_ns()}.json")
    sys.stdout.flush()
    sys.stderr.flush()
    t0 = time.monotonic()
    pid = os.fork()
    if pid == 0:
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        code = 0
        try:
            _child(overlay, test_ids, out, per_test_timeout, probe_report, abort_on_infection, import_mods)
        except BaseException:  # noqa: BLE001
            code = 1
        os._exit(code)
    timed_out = False
    try:
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() - t0 > timeout_s:
                timed_out = True
                _kill(pid)
                break
            time.sleep(0.005)
    except BaseException:
        _kill(pid)
        raise
    res = {"failed": set(), "seen": set(), "errors": set(), "timed_out": timed_out,
           "seconds": time.monotonic() - t0, "imports": None}
    if os.path.exists(out):
        with open(out) as fh:
            d = json.load(fh)
        os.remove(out)
        res["failed"], res["seen"], res["errors"] = set(d["failed"]), set(d["seen"]), set(d["errors"])
        if not timed_out and isinstance(d.get("imports"), dict):
            res["imports"] = d["imports"]
    else:
        res["timed_out"] = True
    return res


def check_imports(overlay: str, modules: list[str], workdir: str, timeout_s: float) -> dict[str, bool] | None:
    """Import each module from the patched overlay in a forked child (sqlglot is never imported
    in the parent, so this is a fresh import). Returns {module: imported_ok}, or None if the
    check itself could not run (caller must then assume the worst)."""
    out = os.path.join(workdir, f"imp-{os.getpid()}-{time.monotonic_ns()}.json")
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        res = {}
        try:
            import importlib

            os.chdir(overlay)
            sys.path.insert(0, overlay)
            os.environ["SKIP_INTEGRATION"] = "0"
            sys.dont_write_bytecode = True
            for m in modules:
                try:
                    importlib.import_module(m)
                    res[m] = True
                except BaseException:  # noqa: BLE001
                    res[m] = False
            with open(out, "w") as fh:
                json.dump(res, fh)
        finally:
            os._exit(0)
    t0 = time.monotonic()
    try:
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() - t0 > timeout_s:
                _kill(pid)
                return None
            time.sleep(0.005)
    except BaseException:
        _kill(pid)
        raise
    try:
        with open(out) as fh:
            d = json.load(fh)
        os.remove(out)
        return d
    except (OSError, ValueError):
        return None


def replay_check(overlay: str, test_ids: list[str], base: dict, skip_exec_ok: bool, workdir: str,
                 timeout_s: float, limited: dict | None = None, only: dict | None = None,
                 iso: list | None = None, record: bool = False, timed: bool = False) -> tuple[dict, list[str]]:
    """Run replay.check on the patched overlay in a forked child. Returns ({test id: same?},
    log lines); every test is False when the child timed out or died. `iso`: run the isolated
    replay of these TPC-DS queries instead (replay.iso_run); its verdict ("same" / "diff" /
    "unknown"), or with `record` its recordings, is returned under the key "iso". `timed`: the
    plain replay runs under the per-query CPU limits."""
    warm_up()
    out = os.path.join(workdir, f"rep-{os.getpid()}-{time.monotonic_ns()}.json")
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        try:
            os.setpgid(0, 0)
        except OSError:
            pass
        code = 0
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.chdir(overlay)
            sys.path.insert(0, overlay)
            os.environ["SKIP_INTEGRATION"] = "0"
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            sys.dont_write_bytecode = True
            relax_gc()  # a replay is short-lived: nothing is collected until it exits
            import replay

            notes: list[str] = []
            extra = None
            if iso is not None:
                extra = replay.iso_run(base, skip_exec_ok, iso, notes.append, record=record)
                res = {t: extra == "same" for t in test_ids}
            elif limited is not None:
                # reach-limited replay: the overlay's changed expression is probed, and only
                # the queries that reach it are run (see replay.check_limited)
                import probe

                ok = replay.check_limited(test_ids, base, skip_exec_ok, limited, probe.install_strict(),
                                          notes.append)
                res = {t: ok for t in test_ids}
            else:
                res = replay.check(test_ids, base, skip_exec_ok, notes.append, only, timed=timed)
            with open(out, "w") as fh:
                json.dump({"res": res, "notes": notes, "iso": extra}, fh)
        except BaseException:  # noqa: BLE001
            code = 1
        os._exit(code)
    t0 = time.monotonic()
    timed_out = False
    try:
        while True:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() - t0 > timeout_s:
                timed_out = True
                _kill(pid)
                break
            time.sleep(0.005)
    except BaseException:
        _kill(pid)
        raise
    res: dict = {t: False for t in test_ids}
    if iso is not None:
        res["iso"] = "diff" if timed_out else "unknown"  # a hang is a failure; no result is no proof
    notes = ["replay: timed out"] if timed_out else []
    if not timed_out:
        try:
            with open(out) as fh:
                d = json.load(fh)
            for t, ok in d["res"].items():
                if t in res:
                    res[t] = ok is True
            notes = list(d.get("notes", []))
            if iso is not None and d.get("iso") is not None:
                res["iso"] = d["iso"]
        except (OSError, ValueError):
            notes = ["replay: no result"]
    try:
        os.remove(out)
    except OSError:
        pass
    return res, notes
