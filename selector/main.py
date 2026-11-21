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

HUGE_S = 3.0  # a clean probe drops only unverified tests longer than this; shorter ones stay
# fork + sqlglot import + collection, per verification run, on the cost map's machine
# (measured: 0.09-0.15 s in the child for one cheap test, plus the overlay in the parent)
OVERHEAD_S = float(os.environ.get("BR_OVERHEAD", "0.2"))
# Scorer CPU seconds per CPU second on the cost map's machine. The cost map's own "scale"
# (1.3, a guess) is not used: the board charged v4 about twice what the cost map's machine
# measures for the same work.
CPU_SCALE = float(os.environ.get("BR_SCALE", "1.8"))
# The same for the constants below that were measured by hand on that machine (run overhead,
# replay costs). Kept apart so that a cost map built on another machine can have its own scale.
FIXED_SCALE = float(os.environ.get("BR_FIXED_SCALE") or CPU_SCALE)
CPU_FACTOR = 1.25  # our CPU seconds per recorded test second, for tests the cost map lacks
MIN_CONSIDER_S = 0.1  # tests cheaper than this are never weighed one by one
MARGIN_S = 0.1  # expected gains smaller than this are not worth the bet
WALL_PER_COMMIT_S = 4.5  # the limit is 3 min / ~40 commits (4.5 in the final, 1.5x commits)
# replay.py: expected CPU seconds on the cost map's machine, (trees only, with the python
# executor). A full TPC-DS replay takes about 8.3 s when nothing differs and stops within a
# second or two when something does (about three times in ten): 0.7 * 8.3 + 0.3 * 1.5.
REPLAY_OVERHEAD_S = 0.35
REPLAY_COST = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": (1.4, 6.3),
    "tests/test_executor.py::TestExecutor::test_execute_tpch": (0.1, 0.1),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": (0.15, 0.15),
}
REPLAY_FULL_S = 8.3  # the same replay when it runs to its end (for time limits)
REPLAY_CHEAP_S = 2.5  # replays at most this long also run when a probe says "invisible"

# Reach-limited replay (replay.check_limited): only the queries whose processing runs the
# changed expression are replayed, under a probe on that expression. Tried first when it is
# expected to cost at most LIM_MAX_SHARE of the plain replay; the plain replay follows
# whenever it proves nothing.
REACH = os.environ.get("BR_REACH", "1") == "1"
REACH_AUDIT = os.environ.get("BR_REACH_AUDIT", "0") == "1"  # run the plain replay too, log both
REPLAY_RUN = {  # the plain replay when it runs to its end: (trees only, with the executor)
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": (1.5, 8.3),
    "tests/test_executor.py::TestExecutor::test_execute_tpch": (0.1, 0.1),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": (0.15, 0.15),
}
LIM_TABLES_S = float(os.environ.get("BR_LIM_TABLES", "0.3"))  # loading the TPC tables, part of a replay with the executor
LIM_MAX_SHARE = float(os.environ.get("BR_LIM_SHARE", "0.6"))
P_LIM = float(os.environ.get("BR_P_LIM", "0.5"))  # share of limited replays that prove "unchanged"

# v7. Isolated replay (replay.iso_run): the TPC-DS queries that reach the changed lines are run
# alone, each in a fork of its own, and must reproduce their baseline and the signature of
# what they leave behind that was recorded at image build. No probe is needed. It runs before
# the plain replay, which follows whenever it proves nothing. BR_ISO=0 restores v6 exactly
# (and removes the per-query CPU limits of the plain replay).
ISO = os.environ.get("BR_ISO", "1") == "1"
ISO_PER_QUERY_S = float(os.environ.get("BR_ISO_PER_QUERY", "0.1"))  # fork + collection + signature (measured 0.09-0.11)
ISO_MAX_SHARE = float(os.environ.get("BR_ISO_SHARE", "0.6"))
P_ISO = float(os.environ.get("BR_P_ISO", "0.5"))  # share of isolated replays that prove "unchanged"
ISO_HEUR_MAX_S = float(os.environ.get("BR_ISO_HEUR_MAX", "1.5"))  # dearest proof asked in place of the clean-probe heuristic

# v8. The clean-probe heuristic (decide): after MIN_PROBE_TESTS probed tests ran the changed
# expression with its old and new value never differing, unverified tests longer than HUGE_S
# are dropped without proof. BR_HEUR=0 turns it off: then nothing is ever dropped unproven
# (replayable TPC tests are replayed in full, the others stay selected). While it is on, a
# replayable TPC test is dropped only after a sample of the plain replay on the real tables
# reproduced the baseline: the first HEUR_BOX_N executed TPC-DS queries (of those that reach
# the changed lines, when the reach map can tell; about 1.9 s on the cost map's machine when
# all ten run in full), and any replay that differs keeps every huge test.
HEUR = os.environ.get("BR_HEUR", "1") == "1"
HEUR_BOX_N = int(os.environ.get("BR_HEUR_BOX", "10"))

MAP_DIR = os.path.dirname(MAP)
COST = os.environ.get("BR_COST", os.path.join(MAP_DIR, "cost.json"))
REPLAY = os.environ.get("BR_REPLAY", os.path.join(MAP_DIR, "replay.json"))


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


def import_check(A, mods: list[str], cid) -> dict[str, bool] | None:
    """Import each module from the patched tree in a child of its own; None if it cannot run."""
    import warm
    from runner import build_overlay

    overlay = os.path.join(WORK, "ovi")
    try:
        build_overlay(REPO, overlay, patched_files(A, REPO))
        return warm.check_imports(overlay, mods, WORK, 60)
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: import check failed to run ({exc!r})")
        return None
    finally:
        shutil.rmtree(overlay, ignore_errors=True)


def import_breaks(M, mods: list[str], res: dict[str, bool] | None, cid) -> set[int]:
    """Tests that fail because a patched module no longer imports (e.g. a class-body
    expression that now raises). Narrowing by class users assumes the import still works;
    `res` is the check of that ({module: imported}). If the check could not run (None), or
    did not cover a module, every user of that module is kept."""
    failed = [m for m in mods if res is None or res.get(m) is not True]
    out: set[int] = set()
    for m in failed:
        rel = m.replace(".", "/")
        for cand in (rel + ".py", rel + "/__init__.py"):
            if os.path.exists(os.path.join(REPO, cand)):
                out |= M.file_users(cand)
    if failed:
        log(f"{cid}: import check: {len(failed)} module(s) fail to import -> {len(out)} users added")
    return out


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


def reach_probe(A, cid, base, ids: list[str]):
    """(path, probed source, report path, {test id: queries}) when a reach-limited replay can
    speak for the replayed tests `ids`, else None. Needs: the patch is one 1:1 line
    replacement in one file and nothing else; the change sits in an expression the strict
    probe can watch (probe.instrument_wide); the reach map vouches for that expression's
    lines (replay.reach_of); and at least one query reaches them."""
    if base is None or not base.get("reach") or len(A.files) != 1 or A.code_opaque:
        return None
    target = probe_target(A, REPO)
    if target is None:
        return None
    try:
        import probe
        import replay

        path, lo, ln, span_o, span_n = target
        if lo != ln or not path.startswith("sqlglot/"):
            return None
        with open(os.path.join(REPO, path), encoding="utf-8") as fh:
            old_src = fh.read()
        report = os.path.join(WORK, "probe-reach.txt")
        got = probe.instrument_wide(old_src, A.new_sources[path], lo, ln, span_o, span_n, report)
        if got is None:
            return None
        src, n_lo, n_hi = got
        only = replay.reach_of(base, path, n_lo, n_hi, ids)
        if only is None or not any(only.values()):
            return None
        return path, src, report, only
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: reach probe setup failed ({exc!r})")
        return None


def limited_cost(base, tid: str, queries: list[int], trees: bool) -> float:
    """Expected CPU seconds (cost map's machine) of the reach-limited replay of one test."""
    recs = base["tests"][tid]
    heavy = [i for i, r in enumerate(recs) if r is not None]  # executed (or, optimized tpch: every) query
    reached = len(set(queries) & set(heavy))
    share = reached / len(heavy) if heavy else len(queries) / max(1, len(recs))
    full_s = REPLAY_RUN[tid][0 if trees else 1]
    if trees or not tid.endswith(("::test_execute_tpcds", "::test_execute_tpch")):
        return full_s * share
    tables = min(LIM_TABLES_S, full_s) if reached else 0.0
    return tables + (full_s - min(LIM_TABLES_S, full_s)) * share


def iso_target(A, base, skip_exec_ok: bool):
    """(queries, expected CPU seconds on the cost map's machine) when the isolated replay can
    speak for test_execute_tpcds, else None. Needs: one changed file, every judged change an
    edit inside a function body or a lambda; the reach map vouches for every changed range
    (replay.reach_of: never run at import or set-up, the same queries in every recorded
    arrangement); at least one query reaches them; each of those has a recorded signature."""
    if not ISO or base is None or not base.get("iso") or not base.get("reach"):
        return None
    if A.code_opaque or len(A.files) != 1 or not A.changes:
        return None
    try:
        import replay

        tid = replay.T_TPCDS
        qs: set[int] = set()
        for path, kind, lo, hi in A.changes:
            if kind not in ("body", "lambda"):
                return None
            got = replay.reach_of(base, path, lo, hi, [tid])
            if got is None:
                return None
            qs.update(got[tid])
        mode = "trees" if skip_exec_ok else "full"
        sig = base["iso"]["sig"][tid][mode]
        secs = base["iso"]["t"][tid][mode]
        if not qs or any(sig[i] is None or secs[i] is None for i in qs):
            return None
        # (every query runs twice: alone, then again in the state its first run left)
        cost = 2 * sum(secs[i] for i in qs) + ISO_PER_QUERY_S * len(qs) + (0.0 if skip_exec_ok else LIM_TABLES_S)
        return sorted(qs), cost
    except Exception:  # noqa: BLE001
        return None


class Costs:
    """What a test costs us to run (costmap.py), with recorded seconds as the fallback."""

    def __init__(self, path: str, tests: list[dict]):
        self.sec: dict[int, float] = {}
        self.setup: dict[str, float] = {}
        self.scale = 1.0
        self.cls = {t["number"]: t["id"].rsplit("::", 1)[0] for t in tests}
        self.rec = {t["number"]: float(t["seconds"]) for t in tests}
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.scale = CPU_SCALE
            num = {t["id"]: t["number"] for t in tests}
            self.sec = {num[i]: float(v) for i, v in d["tests"].items() if i in num}
            self.setup = {c: float(v) for c, v in d.get("class_setup", {}).items()}
        except Exception as exc:  # noqa: BLE001
            log(f"cost map unavailable ({exc!r}): using recorded seconds")

    def run(self, n: int) -> float:
        c = self.sec.get(n)
        return CPU_FACTOR * self.rec[n] if c is None else self.scale * c

    def class_setup(self, n: int) -> float:
        return self.scale * self.setup.get(self.cls[n], 0.0)


def replay_valid(A, M, n: int) -> bool:
    """replay.py mirrors test n faithfully only while its test code and fixtures are the
    pinned ones: every changed file must be an existing sqlglot/ module, or a fixture the
    test never opens."""
    for fc in A.files:
        if fc.is_binary or fc.is_new or fc.is_deleted or fc.is_rename:
            return False
        if fc.new_path and fc.old_path and fc.new_path != fc.old_path:
            return False
        path = fc.old_path or fc.path
        if path.endswith(".py"):
            if not path.startswith("sqlglot/") or not os.path.isfile(os.path.join(REPO, path)):
                return False
        else:
            if not path.startswith("tests/fixtures/") or not os.path.isfile(os.path.join(REPO, path)):
                return False
            if any(n in M.openers(q) for q in fc.paths):
                return False
    return True


GAIN_CODE_ONE = float(os.environ.get("BR_GAIN_CODE_ONE", "0.35"))
GAIN_CODE = float(os.environ.get("BR_GAIN_CODE", "0.1"))
GAIN_DECL = float(os.environ.get("BR_GAIN_DECL", "0.6"))


def gain_for(n: int, code_only: bool = False) -> float:
    """Share of the possible saving that verifying n candidates realizes on average,
    measured on our own labelled mutants. It depends on the kind of change far more than
    on the number of candidates: tests that execute an edited function body nearly always
    fail (share 0.1-0.2 of the selection's cost is recovered), while tests that merely load
    a class whose attribute or declaration changed mostly pass (0.6 and more). A single
    candidate of a body edit passes more often than several do (0.2-0.45)."""
    if code_only:
        return GAIN_CODE_ONE if n <= 1 else GAIN_CODE
    return GAIN_DECL


GAIN_REPLAY = 0.7  # a replayed TPC test is unchanged about seven times in ten
GAIN_VISIBLE = float(os.environ.get("BR_GAIN_VISIBLE", "0.35"))  # ... and once a probe has seen the change take effect, failures are likely
GAIN_PROBED = 0.9  # a probed run only runs to its end while the change stays invisible
MIN_PROBE_TESTS = 20  # a clean probe speaks for unverified tests only after this many ran
BIG_TRIES = 12


PROBE_RUN_SHARE = 0.5  # a probed run may cost up to this share of what selecting everything costs


def plan(verifiable: set[int], fixed_keep: set[int], vcost: dict[int, float], costs: Costs,
         replays: set[int], dms: dict[int, int], gain: float | None = None,
         probed: bool = False, code_only: bool = False) -> tuple[set[int], set[int], float, str]:
    """Split `verifiable` into tests to verify now and tests to simply select.

    Verifying a test costs vcost CPU seconds and removes it from the selection if it passes.
    Two plans are costed, and the cheaper is returned if it beats selecting everything:
    verify all but what is not worth it (dearest first), or verify only the few tests that
    dominate the selection. Returns (verify, keep, estimated CPU seconds, reason); keep
    includes fixed_keep.
    """
    everything = sorted(verifiable | fixed_keep)
    static_ms = pack(everything, dms)[0]
    if gain is None:
        gain = GAIN_PROBED if probed else gain_for(len(verifiable - replays), code_only)

    def g(n: int) -> float:
        return GAIN_REPLAY if n in replays else gain

    def finish(verify: set[int], keep: set[int]):
        """Estimated total cost of a plan, after charging class set-up."""
        keep = set(keep)
        keep_ms = pack(sorted(keep), dms)[0]
        groups: dict[str, set[int]] = {}
        for n in verify:
            if n not in replays and costs.class_setup(n) >= 0.2:
                groups.setdefault(costs.cls[n], set()).add(n)
        setup_s = 0.0
        for cls_name in sorted(groups):
            grp = groups[cls_name]
            setup = costs.class_setup(next(iter(grp)))
            extra = pack(sorted(keep | grp), dms)[0] - keep_ms
            if (setup + sum(vcost[n] for n in grp)) * 1000 >= gain * extra:
                keep |= grp  # set-up makes this class not worth running
                keep_ms += extra
            else:
                setup_s += setup
        verify = set(verify) - keep
        if not verify:
            return None
        cpu = setup_s + sum(vcost[n] for n in verify)
        if verify - replays:
            cpu += OVERHEAD_S * FIXED_SCALE
        if verify & replays:
            cpu += REPLAY_OVERHEAD_S * FIXED_SCALE
        weight = sum(dms[n] for n in verify) or 1
        blend = sum(g(n) * dms[n] for n in verify) / weight
        est = cpu + (keep_ms + (1.0 - blend) * (static_ms - keep_ms)) / 1000
        return est, cpu, verify, keep

    # plan A: verify everything except tests that cost more to verify than they add
    keep = set(fixed_keep)
    keep_ms = pack(sorted(keep), dms)[0]
    verify = set(verifiable) - keep
    for n in sorted(verify, key=lambda n: (-vcost[n], n)):
        if vcost[n] < MIN_CONSIDER_S:
            break
        extra = pack(sorted(keep | {n}), dms)[0] - keep_ms
        if vcost[n] * 1000 >= g(n) * extra:
            keep.add(n)
            keep_ms += extra
    plan_a = finish(verify - keep, keep)
    plans = [plan_a]
    # plan B: select everything except the few long tests whose removal pays for itself
    keep = set(everything)
    keep_ms = static_ms
    for n in sorted(set(verifiable) - fixed_keep, key=lambda n: (-dms[n], n))[:BIG_TRIES]:
        less = keep_ms - pack(sorted(keep - {n}), dms)[0]
        if vcost[n] * 1000 < g(n) * less:
            keep.discard(n)
            keep_ms -= less
    if len(keep) < len(everything):
        plans.append(finish(set(everything) - keep, keep))
    plans = [p for p in plans if p is not None]
    if not plans:
        return set(), set(everything), 0.0, "nothing worth verifying"
    est, cpu, verify, keep = min(plans, key=lambda p: p[0])
    if probed and plan_a is not None and plan_a[2] - replays:
        # a probe watches the run: it stops at the first visible difference, and a clean one
        # settles the whole commit, so running the cheap candidates is worth more than the
        # estimate above credits. Take that plan whenever the run itself is affordable.
        run_cpu = plan_a[1] - sum(vcost[n] for n in plan_a[2] & replays)
        if run_cpu * 1000 <= PROBE_RUN_SHARE * static_ms:
            _, cpu, verify, keep = plan_a
            return verify, keep, cpu, (f"probe run {len(verify)} (~{cpu:.1f}s cpu) vs static "
                                       f"{static_ms / 1000:.1f}s")
    if est + MARGIN_S >= static_ms / 1000:
        return set(), set(everything), 0.0, f"verification ({est:.1f}s) not cheaper than static ({static_ms / 1000:.1f}s)"
    return verify, keep, cpu, (f"verify {len(verify)} (~{cpu:.1f}s cpu): est {est:.1f}s vs static "
                               f"{static_ms / 1000:.1f}s")


def limited_replay(A, nums: list[int], id_of, cid: str, base: dict, skip_exec_ok: bool, lim) -> bool:
    """Reach-limited replay of the given tests (see replay.check_limited). True only when it
    proves all of them unchanged."""
    import warm
    from runner import build_overlay

    path, src, report, only = lim
    ids = [id_of[n] for n in nums]
    if not any(only.get(t) for t in ids):
        return False
    overlay = os.path.join(WORK, "ovl")
    try:
        files = patched_files(A, REPO)
        files[path] = src.encode("utf-8")
        if os.path.exists(report):
            os.remove(report)
        build_overlay(REPO, overlay, files)
        import resource

        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        res, notes = warm.replay_check(overlay, ids, base, skip_exec_ok, WORK, 20 + 4 * REPLAY_FULL_S,
                                       limited={t: list(only.get(t, ())) for t in ids})
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        import probe

        ok = all(res.get(t) is True for t in ids) and not probe.read_report(report)[2]
        log(f"{cid}: reach cpu {r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime:.2f}s "
            f"({'trees' if skip_exec_ok else 'full'}, {'+'.join(str(len(only.get(t, ()))) for t in ids)} queries): "
            f"{'unchanged' if ok else 'not proven'}" + (f" | {'; '.join(notes)[:120]}" if notes else ""))
        return ok
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: reach-limited replay failed to run ({exc!r})")
        return False
    finally:
        shutil.rmtree(overlay, ignore_errors=True)


def iso_replay(A, cid: str, base: dict, skip_exec_ok: bool, iso) -> str:
    """Isolated replay of test_execute_tpcds (replay.iso_run): "same" (proven unchanged), "diff"
    (a reached query does not reproduce its baseline) or "unknown"."""
    import warm
    from runner import build_overlay

    queries, est = iso
    overlay = os.path.join(WORK, "ovs")
    try:
        import replay
        import resource

        build_overlay(REPO, overlay, patched_files(A, REPO))
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        res, notes = warm.replay_check(overlay, [replay.T_TPCDS], base, skip_exec_ok, WORK,
                                       20 + 4 * max(est, 2.0), iso=list(queries))
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        verdict = res.get("iso") if res.get("iso") in ("same", "diff", "unknown") else "unknown"
        log(f"{cid}: iso cpu {r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime:.2f}s "
            f"({'trees' if skip_exec_ok else 'full'}, {len(queries)} queries, est {est:.2f}s): {verdict}"
            + (f" | {'; '.join(notes)[:120]}" if notes else ""))
        return verdict
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: isolated replay failed to run ({exc!r})")
        return "unknown"
    finally:
        shutil.rmtree(overlay, ignore_errors=True)


def box_replay(A, nums: list[int], id_of, cid: str, base: dict, skip_exec_ok: bool) -> bool:
    """v8. A sample of the plain replay of the given TPC tests on the patched tree, asked
    before the clean-probe heuristic drops them unproven. test_execute_tpcds: the first
    HEUR_BOX_N executed queries in the test's order, taken from the queries that reach the
    changed lines when the reach map vouches for those lines (with at most HEUR_BOX_N of
    them, that is every query that can tell). The two TPC-H tests: every query. True only when
    every replayed query reproduces its baseline; False when one differs or raises, runs over
    its limit, or the replay cannot run."""
    import warm
    from runner import build_overlay

    overlay = os.path.join(WORK, "ovb")
    try:
        import replay
        import resource

        ids = [id_of[n] for n in nums]
        only: dict[str, list[int]] = {}
        for tid in ids:
            recs = base.get("tests", {}).get(tid)
            if recs is None:
                return False
            if tid != replay.T_TPCDS:
                only[tid] = list(range(len(recs)))
                continue
            pool = [i for i, r in enumerate(recs) if r is not None]  # the executed queries
            reached: set[int] | None = set()
            if A.code_opaque or len(A.files) != 1 or not A.changes:
                reached = None
            else:
                for path, kind, lo, hi in A.changes:
                    got = replay.reach_of(base, path, lo, hi, [tid]) if kind in ("body", "lambda") else None
                    if got is None:
                        reached = None
                        break
                    reached.update(got[tid])
            if reached is not None and any(i in reached for i in pool):
                pool = [i for i in pool if i in reached]
            only[tid] = pool[:max(1, HEUR_BOX_N)]
        build_overlay(REPO, overlay, patched_files(A, REPO))
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        res, notes = warm.replay_check(overlay, ids, base, skip_exec_ok, WORK, 20 + 4 * REPLAY_FULL_S,
                                       only=only, timed=ISO)
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        ok = all(res.get(t) is True for t in ids)
        log(f"{cid}: sample replay cpu {r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime:.2f}s "
            f"({'trees' if skip_exec_ok else 'full'}, {'+'.join(str(len(only[t])) for t in ids)} queries): "
            f"{'same' if ok else 'differs'}" + (f" | {'; '.join(notes)[:120]}" if notes else ""))
        return ok
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: sample replay failed to run ({exc!r}): keeping")
        return False
    finally:
        shutil.rmtree(overlay, ignore_errors=True)


def run_replay(A, nums: list[int], id_of, cid: str, base: dict, skip_exec_ok: bool, est_s: float,
               lim=None, iso=None) -> set[int]:
    """Replay the given tests on the patched tree; returns those NOT proven unchanged."""
    import warm
    from runner import build_overlay

    if not nums:
        return set()
    proven = False
    if lim is not None:
        proven = limited_replay(A, nums, id_of, cid, base, skip_exec_ok, lim)
        if proven and not REACH_AUDIT:
            return set()
    iso_verdict = None
    if iso is not None and not proven:
        import replay as _rp

        ds = [n for n in nums if id_of[n] == _rp.T_TPCDS]
        if ds:
            iso_verdict = iso_replay(A, cid, base, skip_exec_ok, iso)
            if iso_verdict == "same" and not REACH_AUDIT:
                nums = [n for n in nums if n not in ds]  # proven; the plain replay speaks for the rest
                if not nums:
                    return set()
    overlay = os.path.join(WORK, "ovr")
    try:
        build_overlay(REPO, overlay, patched_files(A, REPO))
        import resource

        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        res, notes = warm.replay_check(overlay, [id_of[n] for n in nums], base, skip_exec_ok, WORK,
                                       20 + 4 * max(est_s, REPLAY_FULL_S), timed=ISO)
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        same = [n for n in nums if res.get(id_of[n]) is True]
        if iso_verdict is not None and REACH_AUDIT:
            import replay as _rp

            plain_ds = all(res.get(id_of[n]) is True for n in nums if id_of[n] == _rp.T_TPCDS)
            log(f"{cid}: iso-audit iso={iso_verdict} plain={'same' if plain_ds else 'diff'} "
                f"({'trees' if skip_exec_ok else 'full'}, {len(iso[0])} queries)"
                + (" DISAGREE" if iso_verdict == "same" and not plain_ds else ""))
        log(f"{cid}: replay cpu {r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime:.2f}s "
            f"({'trees' if skip_exec_ok else 'full'}): {len(same)}/{len(nums)} unchanged"
            + (f" | {'; '.join(notes)[:120]}" if notes else ""))
        if lim is not None and REACH_AUDIT:
            log(f"{cid}: reach-audit limited={'same' if proven else 'no'} plain={len(same)}/{len(nums)}"
                + (" DISAGREE" if proven and len(same) != len(nums) else ""))
        if REACH_AUDIT:
            # measurement only: what replaying just the reached queries, with no probe, would say
            import replay

            only: dict | None = {id_of[n]: set() for n in nums}
            for path, kind, lo, hi in (A.changes if not A.code_opaque and len(A.files) == 1 else [("", "", 0, 0)]):
                got = replay.reach_of(base, path, lo, hi, [id_of[n] for n in nums]) if kind in ("body", "lambda") else None
                if got is None:
                    only = None
                    break
                for t, qs in got.items():
                    only[t].update(qs)
            if only is None:
                log(f"{cid}: reachonly-audit n/a plain={len(same)}/{len(nums)}")
            else:
                r2 = resource.getrusage(resource.RUSAGE_CHILDREN)
                res2, _ = warm.replay_check(overlay, [id_of[n] for n in nums], base, skip_exec_ok, WORK,
                                            20 + 4 * max(est_s, REPLAY_FULL_S), only={t: sorted(q) for t, q in only.items()})
                r3 = resource.getrusage(resource.RUSAGE_CHILDREN)
                same2 = sum(1 for n in nums if res2.get(id_of[n]) is True)
                log(f"{cid}: reachonly-audit queries {'+'.join(str(len(only[id_of[n]])) for n in nums)} "
                    f"cpu {r3.ru_utime + r3.ru_stime - r2.ru_utime - r2.ru_stime:.2f}s -> {same2}/{len(nums)} plain={len(same)}/{len(nums)}"
                    + (" DISAGREE" if same2 != len(same) else ""))
        return set() if proven else set(nums) - set(same)
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: replay failed to run ({exc!r}): keeping")
        return set() if proven else set(nums)
    finally:
        shutil.rmtree(overlay, ignore_errors=True)


def decide(A, cands: set[int], ctx: dict, cid: str, probed, time_left: float) -> tuple[set[int], str]:
    """The tests to select for a commit whose static candidates are `cands`."""
    dsec, dms, id_of, costs, M = ctx["dsec"], ctx["dms"], ctx["id_of"], ctx["costs"], ctx["M"]
    if not cands:
        return set(), "no candidates"
    py_effect = any(fc.path.endswith(".py") and not any(n.startswith(f"{fc.path}: comments") for n in A.notes)
                    for fc in A.files)
    if not py_effect:
        return set(cands), "data-only change: its readers nearly always fail"
    base = ctx["replay"]
    skip_exec_ok = False
    replays: set[int] = set()
    vcost: dict[int, float] = {}
    if base is not None:
        import replay

        # (a changed fixture does not stand in the way: replay_valid admits a test only if
        # it never opens any changed data file)
        skip_exec_ok = not A.code_opaque and not replay.exec_lines_touched(base, A.changes)
        for n in cands:
            rc = REPLAY_COST.get(id_of[n])
            if rc is not None and id_of[n] in base.get("tests", {}) and replay_valid(A, M, n):
                replays.add(n)
                vcost[n] = FIXED_SCALE * (rc[0] if skip_exec_ok else rc[1])
    lim = None
    if replays and REACH:
        lim = reach_probe(A, cid, base, sorted(id_of[n] for n in replays))
        if lim is not None:
            run_s = {n: limited_cost(base, id_of[n], lim[3][id_of[n]], skip_exec_ok) for n in replays}
            if FIXED_SCALE * sum(run_s.values()) > LIM_MAX_SHARE * sum(vcost[n] for n in replays):
                lim = None  # nearly every query is reached: the plain replay costs the same
            else:
                for n in replays:  # the limited replay, then the plain one when it proves nothing
                    vcost[n] = FIXED_SCALE * run_s[n] + (1.0 - P_LIM) * vcost[n]
    vcost6 = dict(vcost)  # what v6 expects a replay to cost: its rules for "cheap" are kept
    iso = None
    n_ds = None
    if replays and ISO:
        n_ds = ctx["num_of"].get(replay.T_TPCDS)
        if n_ds in replays:
            iso = iso_target(A, base, skip_exec_ok)
            if iso is not None and FIXED_SCALE * iso[1] > ISO_MAX_SHARE * vcost[n_ds]:
                iso = None  # too many queries are reached: the plain replay costs about the same
            if iso is not None:  # the isolated replay, then the plain one when it proves nothing
                vcost[n_ds] = FIXED_SCALE * iso[1] + (1.0 - P_ISO) * vcost[n_ds]
    for n in cands:
        if n not in vcost:
            vcost[n] = costs.run(n)
    static_ms = pack(sorted(cands), dms)[0]
    # readers of a changed fixture nearly always fail: selected without being run
    fixed = set(cands) & A.data_cands
    kinds = {k for _, k, _, _ in A.changes}
    # every code change is an edit inside a function body or a lambda of an existing module
    code_only = bool(kinds) and kinds <= {"body", "lambda"} and not A.code_opaque
    verify_set, keep, cpu, why = plan(set(cands) - fixed, fixed, vcost, costs, replays, dms,
                                      probed=probed is not None, code_only=code_only)
    if not verify_set:
        return set(cands), why
    worst = cpu + (REPLAY_FULL_S * FIXED_SCALE if replays & verify_set else 0.0)
    if 1.5 * worst + 1.0 > time_left:
        return set(cands), "out of time budget: static"
    rep = sorted(verify_set & replays)
    run = sorted(verify_set - replays)
    rep_est = sum(vcost[n] for n in rep)

    def do_run(ids: list[int], probe, abort: bool):
        return verify(A, ids, id_of, ctx["num_of"], ctx["file_tests"], dsec, cid, probe, abort, ctx["imports"])

    if probed is not None and run:
        # a probe watches the changed expression: the run stops at the first difference
        failing, clean, aborted = do_run(run, probed, True)
        if clean and not failing:
            # the changed expression ran and never differed. Cheap replays still run (they
            # are exact). Other unverified huge tests are dropped only on broad evidence;
            # with few tests behind the probe, a replayable one is replayed in full instead.
            broad = HEUR and len(run) >= MIN_PROBE_TESTS
            now = [n for n in rep if vcost6.get(n, vcost[n]) <= REPLAY_CHEAP_S * FIXED_SCALE or not broad]
            differ = run_replay(A, now, id_of, cid, base, skip_exec_ok, sum(vcost[n] for n in now), lim, iso)
            rest = (keep | set(rep)) - set(now)
            # (never a reader of a changed fixture: the probe says nothing about data)
            huge = {n for n in rest - fixed if dsec[n] > HUGE_S} if broad else set()
            note = ""
            proven: set[int] = set()
            if huge and differ:
                # v8: an exact replay has just shown the change at work: nothing is dropped unproven
                huge, note = set(), " (a replay differed: every huge test kept)"
            if huge and ISO and iso is not None and n_ds in huge and iso[1] <= ISO_HEUR_MAX_S:
                # v7: the heuristic is about to drop these unverified. Where a cheap isolated
                # replay can speak for test_execute_tpcds, it is asked: a proof replaces the
                # heuristic. v8: a query that no longer reproduces its baseline keeps every
                # huge test, not that one alone.
                verdict = iso_replay(A, cid, base, skip_exec_ok, iso)
                if verdict == "diff":
                    huge, note = set(), " (the isolated replay differed: every huge test kept)"
                elif verdict == "same":
                    proven.add(n_ds)
            box = sorted(n for n in huge & replays if n not in proven)
            if box and not box_replay(A, box, id_of, cid, base, skip_exec_ok):
                # v8: a sample of the plain replay differs, raises or runs over its limit
                huge, note = set(), " (the sample replay differed: every huge test kept)"
            if huge:
                note = f" ({len(huge - proven)} on the heuristic alone)"
            return (rest - huge) | differ, why + f" | probe: never visible -> {len(huge)} huge dropped" + note
        if aborted:
            # nothing is known about the tests yet, except that failures are now likely
            differ = run_replay(A, rep, id_of, cid, base, skip_exec_ok, rep_est, lim, iso)
            verify2, keep2, _, why2 = plan(set(run), keep | differ, vcost, costs, set(), dms, GAIN_VISIBLE)
            if not verify2:
                return keep2, why + " | visible: " + why2
            failing, _, _ = do_run(sorted(verify2), None, False)
            return keep2 | failing, why + " | visible: " + why2
        differ = run_replay(A, rep, id_of, cid, base, skip_exec_ok, rep_est, lim, iso)
        return keep | failing | differ, why
    differ = run_replay(A, rep, id_of, cid, base, skip_exec_ok, rep_est, lim, iso)
    if not run:
        return keep | differ, why
    if differ:
        verify2, keep2, _, why2 = plan(set(run), keep | differ, vcost, costs, set(), dms,
                                       code_only=code_only)
        if not verify2:
            return keep2, why + " | after replay: " + why2
        failing, _, _ = do_run(sorted(verify2), probed, False)
        return keep2 | failing, why + " | after replay: " + why2
    failing, _, _ = do_run(run, probed, False)
    return keep | failing, why


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
        from static import Map, analyze_patch, import_check_modules

        M = Map(MAP, tests)
    except Exception as exc:  # noqa: BLE001
        log(f"map unavailable ({exc!r}): selecting every test")
        M = None
    ctx = {"dsec": dsec, "dms": dms, "id_of": id_of, "num_of": num_of, "file_tests": file_tests,
           "M": M, "costs": Costs(COST, tests), "replay": None}
    try:
        with open(REPLAY, encoding="utf-8") as fh:
            ctx["replay"] = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log(f"replay baseline unavailable ({exc!r}): TPC tests are not replayed")
    dyn_ok = DYNAMIC
    if dyn_ok:
        try:
            os.makedirs(WORK, exist_ok=True)
        except OSError as exc:
            log(f"no work dir ({exc!r}): static only")
            dyn_ok = False
    import signal

    class _Late(BaseException):  # not an Exception: no verification handler may swallow it
        pass

    def _alarm(signum, frame):
        raise _Late()

    signal.signal(signal.SIGALRM, _alarm)
    hard_budget = WALL_PER_COMMIT_S * len(commits) * 0.8  # the run's limit, with margin
    lines = []
    import resource as _res

    def _cpu() -> float:
        a = _res.getrusage(_res.RUSAGE_SELF)
        b = _res.getrusage(_res.RUSAGE_CHILDREN)
        return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime

    c_prev = _cpu()
    log(f"start: cpu {c_prev:.2f}s")
    for cid in commits:
        select = None
        remaining = hard_budget - (time.monotonic() - t0)
        margin = min(10.0, 0.2 * hard_budget)  # time kept back to print the remaining answers
        if remaining < margin + 1.0:
            log(f"{cid}: out of time: selecting every test")
            M_use = None
        else:
            M_use = M
            signal.setitimer(signal.ITIMER_REAL, max(1.0, min(60.0, remaining - margin)))
        if M_use is not None:
            try:
                with open(os.path.join(cdir, cid + ".patch"), encoding="utf-8", errors="surrogateescape") as fh:
                    text = fh.read()
                A = analyze_patch(text, REPO, M)
                if A.broad:
                    select = set(all_tests)
                    log(f"{cid}: broad | {'; '.join(A.notes)[:200]}")
                else:
                    cands = {n for n in A.cands | set(M.unmapped) if n in dms}
                    # modules whose import could now raise: checked inside the verification
                    # run when there is one, else in a child of their own
                    mods = import_check_modules(A, REPO) if dyn_ok else []
                    ctx["imports"] = {"mods": mods, "res": None}
                    if dyn_ok and cands:
                        probed = make_probe(A, cid)
                        time_left = wall_budget - (time.monotonic() - t0)
                        select, why = decide(A, cands, ctx, cid, probed, time_left)
                    else:
                        select, why = set(cands), ("static" if cands else "no candidates")
                    if mods:
                        res = ctx["imports"]["res"]
                        if res is None:
                            res = import_check(A, mods, cid)
                        broken = {n for n in import_breaks(M, mods, res, cid) if n in dms}
                        if broken - select:
                            why += f" | +{len(broken - select)} users of modules that no longer import"
                            select = set(select) | broken
                            cands |= broken
                    log(f"{cid}: {len(cands)} cands -> {len(select)} | {why} | {'; '.join(A.notes)[:200]}")
            except _Late:
                log(f"{cid}: analysis over its time limit: selecting every test")
                select = None
            except Exception as exc:  # noqa: BLE001
                log(f"{cid}: analysis failed ({exc!r}): selecting every test")
                select = None
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        if select is None:
            select = set(all_tests)
        cost_ms, workers = pack(sorted(n for n in select if n in dms), dms)
        lines.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
        c_now = _cpu()
        log(f"{cid}: cost {cost_ms / 1000:.3f}s cpu {c_now - c_prev:.2f}s")
        c_prev = c_now
    sys.stdout.write("".join(line + "\n" for line in lines))
    sys.stdout.flush()
    log(f"done: {len(commits)} commits in {time.monotonic() - t0:.2f}s wall")
    return 0


def verify(A, run_ids, id_of, num_of, file_tests, dsec, cid, probed=None, abort=False,
           imports=None) -> tuple[set[int], bool, bool]:
    """Run candidates on the patched tree; keep failures and anything uncertain.

    Returns (tests to keep, clean, aborted): clean is True only when a probe watched the
    changed expression, the tests reached it, and its value never differed from the
    original's; aborted is True when the run was stopped at the first visible difference
    (nothing is then known about the tests, and all are returned).
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
        import resource

        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        want = imports["mods"] if imports is not None and imports["res"] is None else None
        res = warm.run(overlay, [id_of[n] for n in run_ids], WORK, limit, per_test, report, abort, want or None)
        if want and res.get("imports") is not None:
            imports["res"] = res["imports"]
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        log(f"{cid}: verify cpu {r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime:.2f}s wall {res['seconds']:.2f}s")
    except Exception as exc:  # noqa: BLE001
        log(f"{cid}: verification failed to run ({exc!r}): keeping all")
        return set(run_ids), False, False
    finally:
        shutil.rmtree(overlay, ignore_errors=True)
    if res["timed_out"]:
        if report is not None and abort:
            import probe

            if probe.read_report(report)[1]:
                log(f"{cid}: probe saw the change take effect: stopped early")
                return set(run_ids), False, True
        log(f"{cid}: verification timed out after {res['seconds']:.1f}s: keeping all")
        return set(run_ids), False, False
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
    return keep, clean, False


if __name__ == "__main__":
    sys.exit(main())
