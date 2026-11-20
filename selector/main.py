"""Blast Radius selector: for each patch in /data/commits, print the tests to run.

Contract: exactly one JSON line per commit on stdout, nothing else; logs go to stderr.
Any failure while analysing a commit falls back to selecting every test for it.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pack import duration_ms, pack  # noqa: E402

DATA = os.environ.get("BR_DATA", "/data")
REPO = os.environ.get("BR_REPO", "/app/sqlglot")
MAP = os.environ.get("BR_MAP", os.path.join(os.path.dirname(HERE), "map", "blast.map"))


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def main() -> int:
    t0 = time.monotonic()
    with open(os.path.join(DATA, "tests.json"), encoding="utf-8") as fh:
        tests = json.load(fh)["tests"]
    dms = {t["number"]: duration_ms(t["seconds"]) for t in tests}
    all_tests = sorted(dms)
    cdir = os.path.join(DATA, "commits")
    commits = sorted(f[: -len(".patch")] for f in os.listdir(cdir) if f.endswith(".patch"))
    try:
        from static import Map, analyze_patch

        M = Map(MAP, tests)
    except Exception as exc:  # noqa: BLE001
        log(f"map unavailable ({exc!r}): selecting every test")
        M = None
    lines = []
    for cid in commits:
        cands = None
        if M is not None:
            try:
                with open(os.path.join(cdir, cid + ".patch"), encoding="utf-8", errors="surrogateescape") as fh:
                    text = fh.read()
                A = analyze_patch(text, REPO, M)
                cands = set(all_tests) if A.broad else (A.cands | set(M.unmapped))
                log(f"{cid}: {len(cands)} candidates | {'; '.join(A.notes)[:300]}")
            except Exception as exc:  # noqa: BLE001
                log(f"{cid}: analysis failed ({exc!r}): selecting every test")
                cands = None
        if cands is None:
            cands = set(all_tests)
        cands = sorted(n for n in cands if n in dms)
        _, workers = pack(cands, dms)
        lines.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
    sys.stdout.write("".join(line + "\n" for line in lines))
    sys.stdout.flush()
    log(f"done: {len(commits)} commits in {time.monotonic() - t0:.2f}s wall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
