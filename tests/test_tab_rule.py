"""Unit tests of the table-entry rule (selector/static.py, `_tab_*`) on the pinned checkout.

Run: BR_REPO=<pinned sqlglot checkout> BR_MAP=<blast.map> BR_TESTS=<tests.json> python -B tests/test_tab_rule.py
"""
import difflib
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("BR_REPO", "/app/sqlglot")
sys.path.insert(0, os.path.join(HERE, "..", "selector"))

import static  # noqa: E402

TESTS = json.load(open(os.environ.get("BR_TESTS", "/data/tests.json")))["tests"]
M = static.Map(os.environ.get("BR_MAP", "/app/map/blast.map"), TESTS)
SNOW = "sqlglot/dialects/snowflake.py"
OLD = '        NUMERIC_ESCAPES = {"x": (16, 2, 2, 0xFF), "u": (16, 4, 4, 0xFFFF), "0": (8, 1, 3, 0xFF)}'


def patch(rel, old, new, count=1):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        src = fh.read()
    assert src.count(old) == count, (rel, old, src.count(old))
    out = src.replace(old, new, 1)
    diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True), "a/" + rel, "b/" + rel, n=3)
    return "diff --git a/%s b/%s\n" % (rel, rel) + "".join(diff)


def run(rel, old, new):
    A = static.analyze_patch(patch(rel, old, new), REPO, M)
    return A, [n for n in A.notes if "table entry" in n]


class TabRule(unittest.TestCase):
    def test_map_is_for_this_checkout(self):
        doc = static._tab_doc()
        self.assertTrue(doc, "no table-entry map for this Python")
        self.assertTrue(static._tab_fresh(doc, REPO, M))

    def test_literal_value_of_a_listed_entry(self):
        A, notes = run(SNOW, OLD, OLD.replace('"u": (16, 4', '"u": (17, 4'))
        self.assertEqual(len(notes), 1, A.notes)
        self.assertIn("NUMERIC_ESCAPES['u']", notes[0])
        doc = static._tab_doc()
        ent = doc["sites"][SNOW + "|133|NUMERIC_ESCAPES"]
        acc, want = 0, set()
        for d in doc["sets"][ent["133:50"]]:
            acc += d
            want.add(M.num[doc["tests"][acc]])
        self.assertEqual(A.cands, want | M.openers(SNOW))
        self.assertFalse(A.broad or A.opaque)

    def refused(self, new, rel=SNOW, old=OLD):
        A, notes = run(rel, old, new)
        self.assertEqual(notes, [], new)
        self.assertGreater(len(A.cands), 100, new)  # the rules for a class-level statement decide

    def test_refused_shapes(self):
        self.refused(OLD.replace('"u": (16, 4', '"U": (16, 4'))  # a key
        self.refused(OLD.replace('"u": (16, 4', '"u": (16, 4').replace('"0": (8', '"0": (9').replace('"x": (16', '"x": (17'))  # two entries
        self.refused(OLD.replace("(16, 4, 4, 0xFFFF)", "(int('16'), 4, 4, 0xFFFF)"))  # a call
        self.refused(OLD.replace("(16, 4, 4, 0xFFFF)", "(DROP, 4, 4, 0xFFFF)"))  # a name
        self.refused(OLD.replace("(16, 4, 4, 0xFFFF)", "(16, 4, 4, 1 // 0)"))  # an operation
        self.refused(OLD.replace(', "0": (8, 1, 3, 0xFF)}', "}"))  # an entry removed
        self.refused(OLD.replace("NUMERIC_ESCAPES =", "NUMERIC_ESCAPEZ ="))  # the name
        self.refused(OLD.replace('"u": (16, 4', '**{}, "u": (17, 4'))  # an entry added

    def test_closure_factory_argument(self):
        rel = "sqlglot/parsers/singlestore.py"
        old = '        "GROUPS FOR USER": _show_parser("GROUPS FOR USER", target=True),'
        A, notes = run(rel, old, old.replace("target=True", "target=False"))
        self.assertEqual(len(notes), 1, A.notes)
        self.assertIn("SHOW_PARSERS['GROUPS FOR USER']", notes[0])
        self.assertLess(len(A.cands), 5)
        # the function called is replaced: not this rule
        A, notes = run(rel, old, old.replace('_show_parser("GROUPS FOR USER", target=True)', '_show_parsers("GROUPS FOR USER", target=True)'))
        self.assertEqual(notes, [])
        # an argument that is not a literal
        A, notes = run(rel, old, old.replace("target=True", "target=bool(1)"))
        self.assertEqual(notes, [])
        # another keyword: the call has another shape
        A, notes = run(rel, old, old.replace("target=True", "targets=True"))
        self.assertEqual(notes, [])
        static.TAB_CALLS = False
        try:
            A, notes = run(rel, old, old.replace("target=True", "target=False"))
            self.assertEqual(notes, [])
        finally:
            static.TAB_CALLS = True

    def test_factory_shapes(self):
        import ast

        def ok(src, name="f", ln=99):
            return static._tab_closure_factory(ast.parse(src), name, REPO, ln)

        self.assertTrue(ok("def f(a):\n    return lambda self: a\n"))
        self.assertTrue(ok("def f(*a, **k):\n    'doc'\n    def g(self):\n        return a\n    return g\n"))
        self.assertTrue(ok("from __future__ import annotations\ndef f(a):\n    def g(self: X) -> Y:\n        return a\n    return g\n"))
        self.assertFalse(ok("def f(a):\n    def g(self: X) -> Y:\n        return a\n    return g\n"))  # annotations are evaluated
        self.assertFalse(ok("def f(a):\n    def g(self, b=a.x):\n        return a\n    return g\n"))  # a default is evaluated
        self.assertFalse(ok("def f(a):\n    check(a)\n    return lambda self: a\n"))  # runs code
        self.assertFalse(ok("def f(a):\n    return make(a)\n"))  # calls
        self.assertFalse(ok("@deco\ndef f(a):\n    return lambda self: a\n"))  # decorated
        self.assertFalse(ok("def f(a):\n    @deco\n    def g(self):\n        return a\n    return g\n"))
        self.assertFalse(ok("def f(a):\n    return lambda self: a\nf = other\n"))  # bound twice
        self.assertFalse(ok("def f(a):\n    return lambda self: a\nclass C:\n    f = 1\n"))  # bound in a class body
        self.assertFalse(ok("if X:\n    def f(a):\n        return lambda self: a\n"))  # not at the top level
        self.assertFalse(ok("def f(a):\n    return lambda self: a\n", ln=1))  # defined after the table
        self.assertFalse(ok("async def f(a):\n    return lambda self: a\n"))
        self.assertFalse(ok("def f(a):\n    def g(self):\n        return a\n    return h\n"))
        self.assertFalse(ok("from .x import f\n"))  # relative import
        self.assertTrue(ok("from sqlglot.dialects.dialect import rename_func\n", "rename_func"))

    def test_switch(self):
        static.TAB = False
        try:
            self.refused(OLD.replace('"u": (16, 4', '"u": (17, 4'))
        finally:
            static.TAB = True

    def test_entry_read_at_import_is_not_listed(self):
        # Expression.__init_subclass__ reads every arg_types table when the class is created
        doc = static._tab_doc()
        self.assertFalse(any(s.endswith("|arg_types") for s in doc["sites"]))
        # tables spread into a subclass's table at import are not listed either
        self.assertFalse(any(s.startswith("sqlglot/parser.py|") and s.endswith("|FUNCTIONS") for s in doc["sites"]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
