
"""Coverage-guided test selection for SQLGlot."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("/data")

FIXTURES = json.loads(
    (ROOT / "fixture_map.json").read_text()
)
COVERAGE = json.loads(
    (ROOT / "line_map.json").read_text()
)


def changed_lines(patch):
    result = {}
    filename = None
    lineno = None

    for s in patch.splitlines():
        if s.startswith("+++ b/"):
            filename = s[6:]
            result.setdefault(filename, [])
            lineno = None

        elif s.startswith("@@"):
            match = re.search(r"\+(\d+)", s)
            lineno = int(match.group(1)) if match else None

        elif lineno is not None and filename:
            if s.startswith("+") and not s.startswith("+++"):
                result[filename].append(lineno)
                lineno += 1
            elif s.startswith(" "):
                lineno += 1

    return result


def select(patch, tests):
    from selector_v3_candidate import select as baseline_select

    baseline = {
        t["number"]
        for t in baseline_select(patch, tests)
    }

    picked = set()
    changed = changed_lines(patch)

    by_id_suffix = {}
    for t in tests:
        suffix = t["id"].split("::")[-1]
        by_id_suffix.setdefault(
            suffix, []
        ).append(t["number"])

    for f, lines in changed.items():

        if f.startswith("tests/fixtures/"):
            if f in FIXTURES:
                for suffix in FIXTURES[f]:
                    name = suffix.split("::")[-1]
                    picked.update(
                        by_id_suffix.get(name, ())
                    )
            else:
                picked.update(baseline)
            continue

        mapping = COVERAGE.get(f)

        if not mapping:
            picked.update(baseline)
            continue

        # Include tests covering 16 lines around changes.
        for lineno in lines:
            for offset in range(-16, 17):
                picked.update(
                    mapping.get(str(lineno + offset), ())
                )

        if f.startswith("sqlglot/parsers/"):
            for name in (
                "test_optimize",
                "test_annotate_funcs",
                "test_qualify_columns",
                "test_qualify_tables",
                "test_quote_identifiers",
                "test_ref_type",
            ):
                picked.update(
                    by_id_suffix.get(name, ())
                )

        if f.startswith("sqlglot/dialects/"):
            picked.update(
                by_id_suffix.get("test_simplify", ())
            )

        if f.startswith("sqlglot/optimizer/"):
            for name in (
                "test_optimize",
                "test_tpcds",
                "test_tpch",
                "test_execute_tpcds",
                "test_execute_tpch",
                "test_execute_subqueries",
            ):
                picked.update(
                    by_id_suffix.get(name, ())
                )

        if f == "sqlglot/serde.py":
            picked.update(
                t["number"]
                for t in tests
                if t["file"] in (
                    "tests/test_serde.py",
                    "tests/test_executor.py",
                    "tests/test_optimizer.py",
                )
            )

    # Restrict selections to the V3 safety boundary.
    picked.intersection_update(baseline)

    return [
        t for t in tests
        if t["number"] in picked
    ]


def main():
    from selector_v3_candidate import schedule

    tests = json.loads(
        (DATA / "tests.json").read_text()
    )["tests"]

    for patch_file in sorted(
        (DATA / "commits").glob("*.patch")
    ):
        workers = schedule(
            select(patch_file.read_text(), tests)
        )

        print(json.dumps({
            "commit": patch_file.stem,
            "workers": workers,
        }))


if __name__ == "__main__":
    main()
