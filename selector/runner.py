"""Run chosen sqlglot tests against a patched tree and report which fail.

The patched tree is an overlay in /tmp: directories on the path to a changed file are
real, everything else is a symlink into the read-only checkout, so building it costs a
few dozen syscalls and almost no space. Tests run under pytest, as the labels were made,
with a small plugin that records every failed report (subtests included), collection
errors, and caps process pools at one worker (the container sees the host's CPU count,
and every spawned worker's CPU time is charged).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


_ESCAPERS: dict[str, set[str]] = {}


def _escaping_test_files(repo: str) -> set[str]:
    hit = _ESCAPERS.get(repo)
    if hit is None:
        hit = set()
        for dp, dns, fns in os.walk(os.path.join(repo, "tests")):
            dns[:] = [d for d in dns if d not in ("fixtures", "__pycache__")]
            for fn in fns:
                if fn.endswith(".py"):
                    path = os.path.join(dp, fn)
                    try:
                        text = open(path, encoding="utf-8", errors="replace").read()
                    except OSError:
                        continue
                    if "resolve(" in text or "realpath" in text or "readlink" in text:
                        hit.add(os.path.relpath(path, repo))
        _ESCAPERS[repo] = hit
    return hit


def build_overlay(repo: str, root: str, new_files: dict[str, bytes | None]) -> None:
    """Create `root` mirroring `repo`, with `new_files` (relative path -> content, or None
    to delete) applied."""
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    real_dirs: set[str] = {""}
    # test files that resolve their own path (Path(__file__).resolve(), realpath) would follow a
    # symlink out of the overlay: they are copied instead, and their directories made real
    escapers = _escaping_test_files(repo)
    for rel in list(new_files) + sorted(escapers):
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:i]))

    def fill(rel_dir: str) -> None:
        src_dir = os.path.join(repo, rel_dir)
        dst_dir = os.path.join(root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        try:
            entries = os.listdir(src_dir)
        except FileNotFoundError:
            entries = []
        for name in entries:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if rel in new_files:
                continue
            src = os.path.join(src_dir, name)
            if rel in real_dirs:
                fill(rel)
            elif rel in escapers:
                shutil.copy2(src, os.path.join(dst_dir, name))  # keeps mtime: cached bytecode stays valid
            else:
                os.symlink(src, os.path.join(dst_dir, name))
        for rel, content in new_files.items():
            if os.path.dirname(rel) == rel_dir and content is not None:
                with open(os.path.join(root, rel), "wb") as fh:
                    fh.write(content)
        for rel in real_dirs:
            if rel and os.path.dirname(rel) == rel_dir and not os.path.exists(os.path.join(root, rel)):
                fill(rel)

    fill("")


PLUGIN = r'''
import json, os, multiprocessing as _mp, concurrent.futures as _cf
_out = os.environ["BR_RUN_OUT"]
_POOL_CTX = os.environ.get("BR_POOL_CTX", "fork")
_failed, _seen, _errors = set(), set(), set()
_orig_init = _cf.ProcessPoolExecutor.__init__
def _capped(self, max_workers=None, mp_context=None, *a, **k):
    # one worker; fork instead of spawn: arguments and results still cross a process
    # boundary by pickling (serde stays exercised) without a fresh interpreter per pool
    _orig_init(self, 1, _mp.get_context(_POOL_CTX), *a, **k)
_cf.ProcessPoolExecutor.__init__ = _capped
def pytest_runtest_logreport(report):
    _seen.add(report.nodeid)
    if report.failed:
        _failed.add(report.nodeid)
def pytest_collectreport(report):
    if report.failed:
        _errors.add(report.nodeid)
def pytest_sessionfinish(session, exitstatus):
    with open(_out, "w") as fh:
        json.dump({"failed": sorted(_failed), "seen": sorted(_seen), "errors": sorted(_errors),
                   "exit": int(exitstatus)}, fh)
'''


def run_pytest(overlay: str, test_ids: list[str], timeout_s: float, per_test_timeout: float,
               python: str = sys.executable) -> dict:
    """Run test_ids in `overlay`; returns {"failed": set, "seen": set, "errors": set,
    "timed_out": bool, "seconds": float}."""
    plug_dir = os.path.join(os.path.dirname(overlay), "plug")
    os.makedirs(plug_dir, exist_ok=True)
    plug = os.path.join(plug_dir, "br_run_plugin.py")
    if not os.path.exists(plug):
        with open(plug, "w") as fh:
            fh.write(PLUGIN)
    out = os.path.join(os.path.dirname(overlay), "run.json")
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ)
    env.update({
        "SKIP_INTEGRATION": "0",
        "PYTHONPATH": f"{overlay}{os.pathsep}{plug_dir}",
        "BR_RUN_OUT": out,
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": env.get("HOME", "/tmp"),
    })
    args = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "br_run_plugin",
            "-p", "no:xdist", f"--timeout={per_test_timeout:.0f}", "-o", "console_output_style=classic",
            "--rootdir", overlay, *test_ids]
    t0 = time.monotonic()
    timed_out = False
    try:
        subprocess.run(args, cwd=overlay, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
    secs = time.monotonic() - t0
    res = {"failed": set(), "seen": set(), "errors": set(), "timed_out": timed_out, "seconds": secs}
    if os.path.exists(out):
        with open(out) as fh:
            d = json.load(fh)
        res["failed"], res["seen"], res["errors"] = set(d["failed"]), set(d["seen"]), set(d["errors"])
    else:
        res["timed_out"] = True
    return res
