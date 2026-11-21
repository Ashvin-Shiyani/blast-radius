"""Blast Radius selector: for each patch in /data/commits, print the tests to run.

Contract: exactly one JSON line per commit on stdout, nothing else; logs go to stderr.
Any failure while analysing a commit falls back to selecting every test for it.

Per commit:
  1. static.py finds every test that can observe the change (recall first).
  2. If the candidates are cheap and verifying them is expected to cost less than running
     them all, the cheap ones are run against the patched code (warm.py) and only those
     that fail are kept; expensive candidates are always kept. Any doubt (timeout, crash,
     a test pytest did not report) keeps the candidate.
  3. pack.py splits the selection over workers to minimise the scored cost.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pack import WORKER_COST_MS, duration_ms, pack  # noqa: E402

DATA = os.environ.get("BR_DATA", "/data")
REPO = os.environ.get("BR_REPO", "/app/sqlglot")
MAP = os.environ.get("BR_MAP", os.path.join(os.path.dirname(HERE), "map", "blast.map"))
WORK = os.environ.get("BR_WORK", "/tmp/br")
DYNAMIC = os.environ.get("BR_DYNAMIC", "1") == "1"

CHEAP_S = 1.0  # tests at most this long (recorded) may be verified by running them
MAX_VERIFY_S = 6.0  # never verify a commit whose cheap candidates add up to more than this
OVERHEAD_S = 0.45  # fork + sqlglot import + collection, per verified commit (scorer CPU)
CPU_FACTOR = 1.25  # our CPU seconds per recorded test second, with margin
WALL_PER_COMMIT_S = 4.5  # the limit is 3 min / ~40 commits (4.5 in the final, 1.5x commits)


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def patched_files(A, repo: str) -> dict[str, bytes | None]:
    from applypatch import apply_hunks

    out: dict[str, bytes | None] = {}
    for fc in A.files:
        if fc.is_deleted:
            out[fc.old_path] = None
            continue
        if fc.is_new or not fc.old_path or not os.path.exists(os.path.join(repo, fc.old_path)):
            text = "\n".join(fc.added_text) + ("\n" if fc.added_text else "")
            out[fc.new_path or fc.path] = text.encode("utf-8", "surrogateescape")
            continue
        src = A.new_sources.get(fc.old_path)
        if src is None:
            with open(os.path.join(repo, fc.old_path), encoding="utf-8", errors="surrogateescape") as fh:
                src = apply_hunks(fh.read(), fc)
        if fc.new_path and fc.new_path != fc.old_path:
            out[fc.old_path] = None
            out[fc.new_path] = src.encode("utf-8", "surrogateescape")
        else:
            out[fc.old_path] = src.encode("utf-8", "surrogateescape")
    return out


KEEP_COSTLY_P = 0.6  # chance a probed change turns out visible (so costly candidates stay)


def probe_target(A, repo: str):
    """The single one-line expression change a probe can watch, or None.

    Requires: exactly one code file with an effect, exactly one 1:1 line replacement in it,
    no other code lines added or removed (comment-only files and data files may also change).
    """
    from static import _changed_span, replacement_targets

    code = [fc for fc in A.files if fc.path.endswith(".py")
            and not any(n.startswith(f"{fc.path}: comments") for n in A.notes)]
    if len(code) != 1:
        return None
    fc = code[0]
    reps = replacement_targets(fc)
    if len(reps) != 1 or len(fc.removed) != 1 or len(fc.added_text) != 1:
        return None
    if fc.old_path not in A.new_sources:
        return None
    lo, ln, old_text, new_text = reps[0]
    span_o = _changed_span(old_text, new_text)
    span_n = _changed_span(new_text, old_text)
    return fc.old_path, lo, ln, span_o, span_n


def make_probe(A, cid):
    """(path, probed source, report path) if the change can be watched by a probe."""
    target = probe_target(A, REPO)
    if target is None:
        return None
    try:
        import probe

        path, lo, ln, span_o, span_n = target
        with open(os.path.join(REPO, path), encoding="utf-8") as fh:
            old_src = fh.read()
        report = os.path.join(WORK, "probe.txt")
        src = probe.instrument(old_src, A.new_sources[path], lo, ln, span_o, span_n, report)
        return (path, src, report) if src is not None else None
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: probe setup failed ({exc!r})")
        return None


def choose_verify(A, cands: set[int], dsec: dict[int, float], dms: dict[int, int],
                  probeable: bool = False) -> tuple[list[int], list[int], str]:
    """Return (to_run, keep_static, reason). to_run empty means: no dynamic verification."""
    if not cands:
        return [], [], "no candidates"
    py_effect = any(fc.path.endswith(".py") and not any(n.startswith(f"{fc.path}: comments") for n in A.notes)
                    for fc in A.files)
    only_data = not py_effect
    cheap = sorted(n for n in cands if dsec[n] <= CHEAP_S)
    costly = sorted(n for n in cands if dsec[n] > CHEAP_S)
    if only_data:
        return [], sorted(cands), "data-only change: its readers nearly always fail"
    if not cheap:
        return [], sorted(cands), "nothing cheap to verify"
    cheap_s = sum(dsec[n] for n in cheap)
    if cheap_s > MAX_VERIFY_S:
        return [], sorted(cands), f"cheap candidates too long to verify ({cheap_s:.1f}s)"
    static_cost = pack(sorted(cands), dms)[0] / 1000
    floor = pack(costly, dms)[0] / 1000 if costly else 0.0
    if probeable and costly:
        floor *= KEEP_COSTLY_P
    verify_cost = OVERHEAD_S + CPU_FACTOR * cheap_s + floor
    if verify_cost >= static_cost:
        return [], sorted(cands), f"verification ({verify_cost:.1f}s) not cheaper than static ({static_cost:.1f}s)"
    return cheap, costly, f"verify {len(cheap)} ({cheap_s:.2f}s): est {verify_cost:.1f}s vs static {static_cost:.1f}s"


def main() -> int:
    t0 = time.monotonic()
    with open(os.path.join(DATA, "tests.json"), encoding="utf-8") as fh:
        tests = json.load(fh)["tests"]
    dms = {t["number"]: duration_ms(t["seconds"]) for t in tests}
    dsec = {t["number"]: float(t["seconds"]) for t in tests}
    id_of = {t["number"]: t["id"] for t in tests}
    num_of = {t["id"]: t["number"] for t in tests}
    file_tests: dict[str, list[int]] = {}
    for t in tests:
        file_tests.setdefault(t["file"], []).append(t["number"])
    all_tests = sorted(dms)
    cdir = os.path.join(DATA, "commits")
    commits = sorted(f[: -len(".patch")] for f in os.listdir(cdir) if f.endswith(".patch"))
    wall_budget = WALL_PER_COMMIT_S * len(commits) * 0.55
    try:
        from static import Map, analyze_patch

        M = Map(MAP, tests)
    except Exception as exc:  # noqa: BLE001
        log(f"map unavailable ({exc!r}): selecting every test")
        M = None
    dyn_ok = DYNAMIC
    if dyn_ok:
        try:
            os.makedirs(WORK, exist_ok=True)
        except OSError as exc:
            log(f"no work dir ({exc!r}): static only")
            dyn_ok = False
    lines = []
    for cid in commits:
        select = None
        if M is not None:
            try:
                with open(os.path.join(cdir, cid + ".patch"), encoding="utf-8", errors="surrogateescape") as fh:
                    text = fh.read()
                A = analyze_patch(text, REPO, M)
                if A.broad:
                    select = set(all_tests)
                    log(f"{cid}: broad | {'; '.join(A.notes)[:200]}")
                else:
                    cands = {n for n in A.cands | set(M.unmapped) if n in dms}
                    probed = make_probe(A, cid) if dyn_ok else None
                    run_ids, keep, why = (choose_verify(A, cands, dsec, dms, probed is not None)
                                          if dyn_ok else ([], sorted(cands), "static"))
                    elapsed = time.monotonic() - t0
                    if run_ids and elapsed + OVERHEAD_S + 3 * sum(dsec[n] for n in run_ids) > wall_budget:
                        run_ids, keep, why = [], sorted(cands), "out of time budget: static"
                    if run_ids:
                        costly_ms = pack(keep, dms)[0] if keep else 0
                        abort = probed is not None and costly_ms >= 0.7 * pack(sorted(cands), dms)[0]
                        failing, clean = verify(A, run_ids, id_of, num_of, file_tests, dsec, cid, probed, abort)
                        if clean and not failing:
                            why += " | probe: never visible -> costly dropped"
                            select = set()
                        else:
                            select = set(keep) | failing
                    else:
                        select = set(keep)
                    log(f"{cid}: {len(cands)} cands -> {len(select)} | {why} | {'; '.join(A.notes)[:200]}")
            except Exception as exc:  # noqa: BLE001
                log(f"{cid}: analysis failed ({exc!r}): selecting every test")
                select = None
        if select is None:
            select = set(all_tests)
        _, workers = pack(sorted(n for n in select if n in dms), dms)
        lines.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
    sys.stdout.write("".join(line + "\n" for line in lines))
    sys.stdout.flush()
    log(f"done: {len(commits)} commits in {time.monotonic() - t0:.2f}s wall")
    return 0


def verify(A, run_ids, id_of, num_of, file_tests, dsec, cid, probed=None, abort=False) -> tuple[set[int], bool]:
    """Run the cheap candidates on the patched tree; keep failures and anything uncertain.

    Returns (tests to keep, clean): clean is True only when a probe watched the changed
    expression, the tests reached it, and its value never differed from the original's.
    """
    import warm
    from runner import build_overlay

    overlay = os.path.join(WORK, "ov")
    files = patched_files(A, REPO)
    report = None
    if probed is not None:
        path, src, report = probed
        files[path] = src.encode("utf-8")
        if os.path.exists(report):
            os.remove(report)
    try:
        build_overlay(REPO, overlay, files)
        limit = 20 + 4 * sum(dsec[n] for n in run_ids)
        per_test = max(5.0, 5 * max(dsec[n] for n in run_ids))  # a hang is a failure; cap its cost
        res = warm.run(overlay, [id_of[n] for n in run_ids], WORK, limit, per_test, report, abort)
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: verification failed to run ({exc!r}): keeping all")
        return set(run_ids), False
    finally:
        shutil.rmtree(overlay, ignore_errors=True)
    if res["timed_out"]:
        if report is not None and abort:
            import probe

            if probe.read_report(report)[1]:
                log(f"{cid}: probe saw the change take effect: stopped early, keeping all")
                return set(run_ids), False
        log(f"{cid}: verification timed out after {res['seconds']:.1f}s: keeping all")
        return set(run_ids), False
    keep = {num_of[x] for x in res["failed"] if x in num_of}
    seen = {num_of[x] for x in res["seen"] if x in num_of}
    keep |= set(run_ids) - seen  # never reported: cannot vouch for it
    for err in res["errors"]:
        path = err.split("::")[0]
        keep |= {n for n in file_tests.get(path, []) if n in set(run_ids)}
    clean = False
    if report is not None:
        import probe

        evaluated, infected, fallback = probe.read_report(report)
        clean = evaluated and not infected and not fallback and not res["errors"]
        log(f"{cid}: probe evaluated={evaluated} infected={infected} fallback={fallback}")
    return keep, clean


if __name__ == "__main__":
    sys.exit(main())
