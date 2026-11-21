"""Build step: what each sqlglot test costs *us* to run, at the pinned commit.

(Needs psutil, which is in the kit's requirements.lock, to count the CPU of live pool workers;
without it those tests are under-counted.)

tests.json gives the seconds each test took in the organisers' run. Our runner (warm.py) is
not that run: process pools are forked with one worker instead of spawned, heavy modules are
already imported, failure diffs are skipped. For most tests the two agree; for the ones that
start pools or load fixtures the recorded time is several times what we pay. The decision
"run this candidate now, or just select it" needs our cost, so it is measured here once:
every test of the pinned checkout is run in our runner and its CPU seconds are recorded
(per test, plus what the first test of each class pays for class set-up).

    python costmap.py <sqlglot checkout> <out.json> [test id to skip ...]

Tests that are skipped here (too long to be worth running while deciding) get no entry.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# never worth running while deciding: recorded time is their real cost
SKIP = (
    "tests/test_parser.py::TestParser::test_max_nodes",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
)


class _Timer:
    def __init__(self) -> None:
        self.t: dict[str, list[float]] = {}  # nodeid -> [setup, call, teardown] seconds
        self._cpu0 = 0.0
        self.cpu: dict[str, float] = {}
        self.failed: set[str] = set()

    @staticmethod
    def _cpu_now() -> float:
        """CPU seconds so far of this process, its threads, and every child: finished ones
        (pool workers that were shut down, subprocesses) and the ones still running (a
        class-level pool's worker), which only psutil can see."""
        t = os.times()
        total = t.user + t.system + t.children_user + t.children_system
        try:
            import psutil

            for child in psutil.Process().children(recursive=True):
                try:
                    ct = child.cpu_times()
                    total += ct.user + ct.system
                except psutil.Error:
                    pass
        except ImportError:
            pass
        return total

    def pytest_runtest_logstart(self, nodeid, location):
        self._cpu0 = self._cpu_now()

    def pytest_runtest_logfinish(self, nodeid, location):
        t1 = self._cpu_now()
        t2 = self._cpu_now()  # what taking a reading costs: not the test's
        self.cpu[nodeid] = max(0.0, (t1 - self._cpu0) - (t2 - t1))
        # as in a verification run: no automatic collection, and what a test leaves behind
        # is cleared (off the clock) before the next one
        import gc

        gc.collect()
        gc.freeze()

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed.add(report.nodeid)
        rec = self.t.setdefault(report.nodeid, [0.0, 0.0, 0.0])
        rec[("setup", "call", "teardown").index(report.when)] += report.duration


def measure(repo: str, out_path: str, skip: tuple[str, ...]) -> None:
    import warm

    warm.warm_up()
    import concurrent.futures as cf

    import pytest

    import multiprocessing as mp

    orig_init = cf.ProcessPoolExecutor.__init__

    def capped(self, max_workers=None, mp_context=None, *a, **k):  # exactly as warm._child does
        orig_init(self, 1, mp.get_context("fork"), *a, **k)

    cf.ProcessPoolExecutor.__init__ = capped
    warm.relax_gc()
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    timer = _Timer()
    args = ["-q", "-p", "no:cacheprovider", "-p", "no:xdist", "--assert=plain", "--tb=no",
            "--rootdir", repo, "-o", "console_output_style=classic", "tests"]
    for s in skip:
        args += ["--deselect", s]
    t0 = time.monotonic()
    code = pytest.main(args, plugins=[timer])
    tests: dict[str, float] = {}
    class_setup: dict[str, float] = {}
    for nodeid, (setup, call, teardown) in timer.t.items():
        wall = call + teardown
        cpu = timer.cpu.get(nodeid, 0.0)  # includes set-up, which is mostly waiting-free
        own = max(wall, cpu - setup)
        tests[nodeid] = round(max(own, 0.0), 4)
        if setup > 0.02:
            cls = nodeid.rsplit("::", 1)[0]
            class_setup[cls] = round(max(class_setup.get(cls, 0.0), setup), 4)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"python": sys.version.split()[0], "tests": tests, "class_setup": class_setup,
                   "failed_here": sorted(timer.failed)}, fh, separators=(",", ":"))
    print(f"costmap: {len(tests)} tests, {sum(tests.values()):.1f}s + class set-up "
          f"{sum(class_setup.values()):.1f}s, {len(timer.failed)} failed here, pytest exit {int(code)}, "
          f"wall {time.monotonic() - t0:.1f}s",
          file=sys.stderr)


if __name__ == "__main__":
    measure(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), tuple(sys.argv[3:]) or SKIP)
