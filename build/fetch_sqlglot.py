"""Build step: fetch sqlglot at the pinned commit from its own GitHub repository and verify it.

python fetch_sqlglot.py <dest_dir> [--digest-only <existing_dir>]
Fails the build if the content digest of sqlglot/ and tests/ differs from EXPECTED.
"""

import hashlib
import io
import os
import sys
import tarfile
import time
import urllib.request

COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
URL = f"https://codeload.github.com/tobymao/sqlglot/tar.gz/{COMMIT}"
EXPECTED = os.environ.get("SQLGLOT_DIGEST", "e514bf38e738b6ef3c5426ad81242103fbff2578eb63ea0035838afe8d2ef806")
PARTS = ("sqlglot/", "tests/")


def digest(root: str) -> str:
    h = hashlib.sha256()
    files = []
    for part in PARTS:
        for dp, dns, fns in os.walk(os.path.join(root, part)):
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for fn in fns:
                if fn.endswith(".pyc"):
                    continue
                files.append(os.path.relpath(os.path.join(dp, fn), root))
    for rel in sorted(files):
        h.update(rel.encode() + b"\0")
        with open(os.path.join(root, rel), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def fetch(dest: str, tarball: str | None = None) -> None:
    data = open(tarball, "rb").read() if tarball else None
    for attempt in range(0 if data else 5):
        try:
            with urllib.request.urlopen(URL, timeout=120) as r:
                data = r.read()
            break
        except Exception as exc:  # noqa: BLE001
            print(f"download attempt {attempt + 1} failed: {exc!r}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    if data is None:
        sys.exit("could not download sqlglot")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            if m.issym() or m.islnk() or ".." in parts[1].split("/"):
                continue
            m.name = parts[1]
            tf.extract(m, dest, filter="data")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--digest-only":
        print(digest(sys.argv[2]))
        sys.exit(0)
    dest = sys.argv[1]
    fetch(dest, sys.argv[3] if len(sys.argv) == 4 and sys.argv[2] == "--tarball" else None)
    got = digest(dest)
    if got != EXPECTED:
        sys.exit(f"sqlglot content digest mismatch: got {got}, expected {EXPECTED}")
    print(f"sqlglot {COMMIT[:7]} fetched and verified ({got[:12]})")
