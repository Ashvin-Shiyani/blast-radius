
"""Static selector using fixture consumers and source dependencies."""

import json
import re
from pathlib import Path

DATA = Path("/data")

FIXTURE_MAP = json.loads(
    (Path(__file__).parent / "fixture_map.json").read_text()
)


def files_in_patch(text):
    return re.findall(r"^\+\+\+ b/(.+)$", text, re.M)


def is_comment_only(text):
    changed = [
        s[1:].strip()
        for s in text.splitlines()
        if s[:1] in ("+", "-")
        and not s.startswith(("+++", "---"))
    ]

    return bool(changed) and all(
        s.startswith("#") or not s for s in changed
    )


def select(patch, tests):
    files = files_in_patch(patch)

    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t)

    picked = set()

    def add_file(file):
        picked.update(
            t["number"] for t in by_file.get(file, ())
        )

    def add_contains(fragment):
        picked.update(
            t["number"]
            for t in tests
            if fragment in t["id"]
        )

    if (
        files
        and all(f.endswith(".py") for f in files)
        and is_comment_only(patch)
    ):
        return []

    for f in files:

        if f.startswith("tests/fixtures/"):
            for suffix in FIXTURE_MAP.get(f, ()):
                add_contains(
                    "tests/test_optimizer.py::" + suffix
                )

            if f not in FIXTURE_MAP:
                add_file("tests/test_optimizer.py")

        elif f.startswith("tests/"):
            add_file(f)

        elif f.startswith((
            "sqlglot/dialects/",
            "sqlglot/parsers/",
            "sqlglot/generators/",
        )):
            name = Path(f).stem

            add_file(f"tests/dialects/test_{name}.py")

            if f.startswith((
                "sqlglot/dialects/",
                "sqlglot/parsers/",
            )):
                add_file("tests/test_optimizer.py")

            if f.startswith("sqlglot/parsers/"):
                add_file("tests/test_parser.py")
                add_file("tests/test_expressions.py")

        elif f.startswith("sqlglot/optimizer/"):
            name = Path(f).stem

            add_contains(
                "tests/test_optimizer.py::TestOptimizer::test_"
                + name
            )

            add_contains(
                "tests/test_optimizer.py::TestOptimizer::test_optimize"
            )

            if name == "eliminate_joins":
                add_file("tests/test_executor.py")
                add_file("tests/test_optimizer.py")

            if name in (
                "scope",
                "qualify",
                "qualify_columns",
                "annotate_types",
            ):
                add_file("tests/test_optimizer.py")

        elif f == "sqlglot/serde.py":
            picked.update(t["number"] for t in tests)

        elif f.startswith("sqlglot/"):
            add_file(
                "tests/test_" + Path(f).stem + ".py"
            )
            add_file("tests/test_optimizer.py")

    if files and all(
        f.startswith((
            "sqlglot/parsers/",
            "sqlglot/serde.py",
        ))
        for f in files
    ):
        picked.difference_update(
            t["number"]
            for t in tests
            if t["id"]
            == "tests/test_parser.py::TestParser::test_max_nodes"
        )

    if files and all(
        f.startswith((
            "sqlglot/dialects/",
            "sqlglot/parsers/",
            "sqlglot/generators/",
        ))
        for f in files
    ):
        shared = {
            "test_simplify",
            "test_optimize",
            "test_qualify_columns",
            "test_annotate_funcs",
            "test_qualify_tables",
            "test_quote_identifiers",
            "test_ref_type",
        }

        picked = {
            n for n in picked
            if not (
                tests[n]["file"] == "tests/test_optimizer.py"
                and tests[n]["id"].split("::")[-1] not in shared
            )
        }

        if all(
            f.startswith("sqlglot/parsers/")
            for f in files
        ):
            picked = {
                n for n in picked
                if (
                    tests[n]["file"] != "tests/test_parser.py"
                    or tests[n]["seconds"] < 0.5
                )
            }

    return [
        t for t in tests
        if t["number"] in picked
    ]


def schedule(tests):
    if not tests:
        return []

    best_cost = float("inf")
    best = []

    for k in range(1, min(8, len(tests)) + 1):
        bins = [[] for _ in range(k)]
        loads = [0] * k

        for t in sorted(
            tests,
            key=lambda t: t["seconds"],
            reverse=True,
        ):
            i = loads.index(min(loads))
            bins[i].append(t["number"])
            loads[i] += round(t["seconds"] * 1000)

        cost = max(loads) + 2000 * k

        if cost < best_cost:
            best_cost = cost
            best = bins

    return best


def main():
    tests = json.loads(
        (DATA / "tests.json").read_text()
    )["tests"]

    for p in sorted(
        (DATA / "commits").glob("*.patch")
    ):
        print(json.dumps({
            "commit": p.stem,
            "workers": schedule(
                select(p.read_text(), tests)
            ),
        }))


if __name__ == "__main__":
    main()
