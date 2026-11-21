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


class _Collector:
    def __init__(self) -> None:
        self.failed: set[str] = set()
        self.seen: set[str] = set()
        self.errors: set[str] = set()

    def pytest_runtest_logreport(self, report):
        self.seen.add(report.nodeid)
        if report.failed:
            self.failed.add(report.nodeid)

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors.add(report.nodeid)


def _child(overlay: str, test_ids: list[str], out: str, per_test_timeout: float, probe_report: str | None = None,
           abort_on_infection: bool = False) -> None:
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
    try:
        uc.pprint.pformat = lambda obj, *a, **k: "<obj>"
    except Exception:  # noqa: BLE001
        pass
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
    try:
        pytest.main(args, plugins=[col])
    finally:
        with open(out, "w") as fh:
            json.dump({"failed": sorted(col.failed), "seen": sorted(col.seen), "errors": sorted(col.errors)}, fh)


def run(overlay: str, test_ids: list[str], workdir: str, timeout_s: float, per_test_timeout: float,
        probe_report: str | None = None, abort_on_infection: bool = False) -> dict:
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
            _child(overlay, test_ids, out, per_test_timeout, probe_report, abort_on_infection)
        except BaseException:  # noqa: BLE001
            code = 1
        os._exit(code)
    timed_out = False
    while True:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() - t0 > timeout_s:
            timed_out = True
            for target in (-pid, pid):
                try:
                    os.kill(target, signal.SIGKILL)
                except OSError:
                    pass
            os.waitpid(pid, 0)
            break
        time.sleep(0.005)
    res = {"failed": set(), "seen": set(), "errors": set(), "timed_out": timed_out,
           "seconds": time.monotonic() - t0}
    if os.path.exists(out):
        with open(out) as fh:
            d = json.load(fh)
        os.remove(out)
        res["failed"], res["seen"], res["errors"] = set(d["failed"]), set(d["seen"]), set(d["errors"])
    else:
        res["timed_out"] = True
    return res
