"""Unit tests of the v11f rules (selector/static.py): readers of a directory's source, and defs
that hold statements only a whole-suite session runs (map/sess.json), on the pinned checkout.

Run: BR_REPO=<pinned sqlglot checkout> BR_MAP=<blast.map> BR_TESTS=<tests.json> python -B tests/test_sess_rule.py
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
TYPING = M.num.get("tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata")
DIALECT = "sqlglot/dialects/dialect.py"


def patch(rel, old, new):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        src = fh.read()
    assert src.count(old) == 1, (rel, old, src.count(old))
    out = src.replace(old, new, 1)
    diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True), "a/" + rel, "b/" + rel, n=3)
    return "diff --git a/%s b/%s\n" % (rel, rel) + "".join(diff)


def run(rel, old, new):
    return static.analyze_patch(patch(rel, old, new), REPO, M)


class SourceReaders(unittest.TestCase):
    def test_lambda_in_a_typing_module(self):
        self.assertIsNotNone(TYPING)
        A = run("sqlglot/typing/hive.py", 'exp.If: {"annotator": lambda self, e: self._annotate_by_args(e, "true", "false", promote=True)}',
                'exp.If: {"annotator": lambda self, e: self._annotate_by_args(e, "true", "false")}')
        self.assertIn(TYPING, A.cands)

    def test_typing_package_init(self):
        # the package's own module is never opened, but its entries are what the others are compared with
        self.assertIn(TYPING, M.src_readers("sqlglot/typing/__init__.py"))

    def test_elsewhere_nothing(self):
        self.assertEqual(M.src_readers("sqlglot/dialects/duckdb.py"), set())
        self.assertEqual(M.src_readers("tests/dialects/test_duckdb.py"), set())


class SessionOnly(unittest.TestCase):
    def test_map_is_for_this_checkout(self):
        doc = static._sess_doc()
        self.assertTrue(doc)
        self.assertEqual(static._cmp_tree_digest(REPO), doc["digest"])
        self.assertIn(DIALECT, doc["lines"])

    def test_statement_in_try_load(self):
        A = run(DIALECT, "                        cls._classes[key] = attr\n                        break",
                "                        pass\n                        break")
        users = M.file_users(DIALECT)
        self.assertGreater(len(users), 1000)
        self.assertEqual(A.sess_cands, users)

    def test_condition_that_guards_it(self):
        A = run(DIALECT, "            if key not in cls._classes:\n                for attr_name in dir(module):",
                "            if key in cls._classes:\n                for attr_name in dir(module):")
        self.assertEqual(A.sess_cands, M.file_users(DIALECT))

    def test_same_def_elsewhere(self):
        A = run(DIALECT, "        if isinstance(key, Dialects):\n            key = key.value\n\n        # 1.",
                "        if isinstance(key, Dialects):\n            key = str(key.value)\n\n        # 1.")
        self.assertEqual(A.sess_cands, M.file_users(DIALECT))  # same def (_try_load)

    def test_other_def_untouched(self):
        A = run(DIALECT, "        return hash(cls.__name__.lower())", "        return hash(cls.__name__.upper())")
        self.assertEqual(A.sess_cands, set())
        A = run("sqlglot/dialects/duckdb.py", "class DuckDB(Dialect):", "class DuckDB(Dialect):  # x")
        self.assertEqual(A.sess_cands, set())


if __name__ == "__main__":
    unittest.main(verbosity=1)
