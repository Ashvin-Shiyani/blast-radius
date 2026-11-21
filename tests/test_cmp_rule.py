"""Unit tests of the identity/equality rule (selector/static.py, `_cmp_*`) with real objects.

Every case the rule accepts is checked on live objects: `(a is b) == (a == b)` and
`(a is not b) == (a != b)`, both ways round, with exact bools. Every trap is checked to be
refused, with a live counterexample where one exists.

Run: BR_REPO=<pinned sqlglot checkout> python -B tests/test_cmp_rule.py
"""
import dataclasses
import datetime
import decimal
import enum
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("BR_REPO", "/app/sqlglot")
sys.path.insert(0, os.path.join(HERE, "..", "selector"))
sys.path.insert(0, REPO)

import static  # noqa: E402
from sqlglot import exp  # noqa: E402
from sqlglot.dialects.dialect import Dialect, NormalizationStrategy  # noqa: E402
from sqlglot.errors import ErrorLevel  # noqa: E402
from sqlglot.executor.env import reverse_key  # noqa: E402
from sqlglot.optimizer.scope import ScopeType  # noqa: E402
from sqlglot.tokenizer_core import TokenType  # noqa: E402

_tmro = type.__dict__["__mro__"].__get__
_tdict = type.__dict__["__dict__"].__get__


def key(c):
    return "%s:%s" % (type.__dict__["__module__"].__get__(c), type.__dict__["__qualname__"].__get__(c))


def enum_ok(t):
    objs = []
    for m in t.__members__.values():
        if not any(m is o for o in objs):
            objs.append(m)
    return all(((x == y) is (x is y)) and ((x != y) is (x is not y)) for x in objs for y in objs)


_FACTS = {}


def facts(t, uniq=1):
    """The class facts exactly as the lab profile reads them from the live class."""
    if (id(t), uniq) not in _FACTS:
        _FACTS[(id(t), uniq)] = (t, _facts(t, uniq))
    return _FACTS[(id(t), uniq)][1]


def _facts(t, uniq):
    mro = _tmro(t)

    def owner(name):
        for c in mro:
            if name in _tdict(c):
                return key(c)

    eq, ne = owner("__eq__"), owner("__ne__")
    en = 1 if enum.Enum in mro else 0
    fl = 1 if enum.Flag in mro else 0
    ok = enum_ok(t) if (en and eq.startswith("builtins:") and eq != "builtins:object") else None
    return [key(t), eq, ne, en, fl, ok, uniq]


def rule(x, y, expr=True):
    return static._cmp_pair(facts(type(x)), facts(type(y)), lambda: expr)


def same(x, y):
    try:
        return (((x == y) is (x is y)) and ((x != y) is (x is not y))
                and ((y == x) is (y is x)) and ((y != x) is (y is not x)))
    except Exception:  # noqa: BLE001
        return False


class Plain:
    pass


class Plain2:
    __slots__ = ("a",)


def func():
    pass


class StrA(str, enum.Enum):
    X = "x"
    Y = "y"
    ALIAS = "x"


class StrB(str, enum.Enum):
    X = "x"


class IntE(enum.IntEnum):
    ONE = 1
    TWO = 2


class PlainE(enum.Enum):
    A = 1
    B = "b"


class Fl(enum.Flag):
    A = 1
    B = 2


class IFl(enum.IntFlag):
    A = 1
    B = 2


class OwnEqEnum(str, enum.Enum):
    A = "a"
    B = "b"

    def __eq__(self, other):
        return True

    __hash__ = str.__hash__


class AlwaysEq:
    def __eq__(self, other):
        return True


class OnlyNe:
    def __ne__(self, other):
        return False


class StrEq(str):
    def __eq__(self, other):
        return True

    __hash__ = str.__hash__


class StrSub(str):
    pass


class Meta(type):
    def __eq__(cls, other):
        return True

    __hash__ = type.__hash__


class WithMeta(metaclass=Meta):
    pass


@dataclasses.dataclass
class Data:
    a: int = 1


NONE = [None]
BOOLS = [True, False]
DEFAULTS = [object(), Plain(), Plain(), Plain2(), func, os, object, Plain, exp.Column, exp.Table, type(None), ErrorLevel]
NUMS = [0, 1, 2, 10**30, 1.0, 0.0, float("nan"), float("inf"), 1j]
STRS = ["", "a", "x", "UPPERCASE", StrSub("a")]
TUPLES = [(), (1,), ("a", "b")]
LISTS = [[], [1], ["a"]]
DICTS = [{}, {1: 2}]
SETS = [set(), {1}, frozenset(), frozenset({1})]
OTHERB = [b"", b"a", range(3), slice(1), len, isinstance]
EXPRS = [exp.column("a"), exp.column("a"), exp.to_table("t"), exp.Literal.number(1), exp.Literal.string("a"),
         exp.Anonymous(this="f"), exp.select("a").from_("t"), exp.Null(), exp.Boolean(this=True)]
FAMILIES = {"num": NUMS, "str": STRS, "tuple": TUPLES, "list": LISTS, "dict": DICTS, "set": SETS}


class Accepted(unittest.TestCase):
    def ok(self, xs, ys, skip_same_class=False):
        n = 0
        for x in xs:
            for y in ys:
                if skip_same_class and type(x) is type(y):
                    continue
                self.assertIs(rule(x, y), True, (x, y))
                self.assertIs(rule(y, x), True, (y, x))
                self.assertTrue(same(x, y), (x, y))
                n += 1
        self.assertGreater(n, 0)

    def test_a_none_none_and_bool_bool(self):
        self.ok(NONE, NONE)
        self.ok(BOOLS, BOOLS)

    def test_b_none_with_anything_listed(self):
        everything = (BOOLS + DEFAULTS + NUMS + STRS + TUPLES + LISTS + DICTS + SETS + OTHERB + EXPRS
                      + list(ErrorLevel) + list(NormalizationStrategy) + list(TokenType)[:20] + list(StrA) + list(IntE) + list(PlainE))
        self.ok(NONE, everything)

    def test_c_two_members_of_one_enum(self):
        for e in (ErrorLevel, ScopeType, NormalizationStrategy, TokenType, StrA, IntE, PlainE):
            self.ok(list(e.__members__.values()), list(e.__members__.values()))
        self.assertIs(StrA.X, StrA.ALIAS)  # an alias is the same object
        self.assertEqual(facts(TokenType)[1], "builtins:int")
        self.assertIs(facts(TokenType)[5], True)
        self.assertEqual(facts(NormalizationStrategy)[1], "builtins:str")
        self.assertIs(facts(NormalizationStrategy)[5], True)

    def test_d_default_eq_on_both_sides(self):
        self.ok(DEFAULTS, DEFAULTS)
        self.ok(list(ErrorLevel), list(ScopeType) + list(PlainE) + DEFAULTS)  # plain Enums compare by identity

    def test_e_default_eq_with_a_builtin(self):
        self.ok(DEFAULTS, BOOLS + NUMS + STRS + TUPLES + LISTS + DICTS + SETS + OTHERB)

    def test_f_builtins_of_different_families(self):
        names = sorted(FAMILIES)
        for i in names:
            for j in names:
                if i != j:
                    self.ok(FAMILIES[i], FAMILIES[j])
        self.ok(BOOLS, STRS + TUPLES + LISTS + DICTS + SETS)

    def test_g_expression_classes(self):
        self.ok(EXPRS, DEFAULTS + BOOLS + NUMS + STRS + TUPLES + LISTS + DICTS + SETS + OTHERB)
        self.ok(EXPRS, EXPRS, skip_same_class=True)
        self.assertEqual(facts(exp.Column)[1], static._CMP_EXPR)
        self.assertEqual(facts(exp.Column)[2], static._CMP_EXPR)

    def test_class_objects_and_metaclasses(self):
        self.assertEqual(static._cmp_family(facts(type(exp.Column)), lambda: True), "default")
        self.assertEqual(static._cmp_family(facts(type(ErrorLevel)), lambda: True), "default")
        self.assertEqual(static._cmp_family(facts(type(Plain)), lambda: True), "default")


class Traps(unittest.TestCase):
    def no(self, x, y, counter=True):
        self.assertIs(rule(x, y), False, (x, y))
        self.assertIs(rule(y, x), False, (y, x))
        if counter:
            self.assertFalse(same(x, y), (x, y))

    def test_same_builtin_family(self):
        nan = float("nan")
        self.no(1, 1.0)
        self.no(True, 1)
        self.no(nan, nan)
        self.no(10**30, int("1" + "0" * 30))
        self.no("ab", "".join(["a", "b"]))
        self.no((1,), tuple([1]))
        self.no([], [])
        self.no({}, {})
        self.no(set(), frozenset())
        self.no(b"a", bytes(bytearray(b"a")))
        self.no(range(3), range(3))
        self.no(1j, complex(0, 1))
        self.no(StrSub("a"), "a")

    def test_bound_and_builtin_methods(self):
        p, lst = Plain2(), []
        d = Dialect()
        self.no(d.normalize_identifier, d.normalize_identifier)
        self.no(lst.append, lst.append)
        self.no(d.normalize_identifier, None, counter=False)

    def test_dialect_classes_and_instances(self):
        sf = Dialect.get_or_raise("snowflake")
        self.no(type(sf), "snowflake")  # the metaclass _Dialect defines __eq__: a class equals its name
        self.no(sf, type(sf))  # Dialect.__eq__: type(self) == other
        self.no(sf, Dialect.get_or_raise("snowflake"), counter=False)
        self.no(sf, None, counter=False)
        self.no(type(sf), None, counter=False)
        self.assertEqual(static._cmp_family(facts(type(type(sf))), lambda: True), "other")

    def test_reverse_key(self):
        self.no(reverse_key(1), None)  # its __eq__ raises on None
        self.no(reverse_key(1), reverse_key(1))

    def test_same_exact_expression_class(self):
        self.no(exp.column("a"), exp.column("a"))
        self.no(exp.Literal.number(1), exp.Literal.number(1))

    def test_expression_without_the_audited_text(self):
        for other in (None, object(), 1, "a", exp.to_table("t")):
            self.assertIs(rule(exp.column("a"), other, expr=False), False)
            self.assertIs(rule(other, exp.column("a"), expr=False), False)

    def test_enums(self):
        self.no(NormalizationStrategy.UPPERCASE, "UPPERCASE")  # a str mix-in compares by value
        self.no(list(TokenType)[5], int(list(TokenType)[5]))
        self.no(IntE.ONE, 1)
        self.no(IntE.ONE, True)
        self.no(IntE.ONE, 1.0)
        self.no(StrA.X, StrB.X)  # two Enum classes with a str mix-in
        self.no(StrA.X, "x")
        self.no(StrA.X, StrSub("x"))
        self.no(PlainE.A, 1, counter=False)  # by instruction: never an Enum member with an int or a str
        self.no(PlainE.B, "b", counter=False)
        self.no(ErrorLevel.RAISE, "RAISE", counter=False)
        self.no(StrA.X, object(), counter=False)  # a mix-in Enum is accepted with None and its own class only
        self.no(StrA.X, exp.column("a"), counter=False)
        self.no(StrA.X, (1,), counter=False)
        self.no(OwnEqEnum.A, OwnEqEnum.B)
        self.no(OwnEqEnum.A, None)

    def test_flags_are_never_equivalent(self):
        for x, y in ((Fl.A, Fl.A), (Fl.A, Fl.B), (Fl.A | Fl.B, Fl.A), (IFl.A, IFl.A), (IFl.A | IFl.B, IFl.B)):
            self.assertIs(rule(x, y), False)
        self.assertIs(rule(Fl.A, None), False)
        self.assertIs(rule(IFl.A, None), False)
        self.no(IFl.A, 1)

    def test_classes_with_their_own_eq(self):
        self.no(AlwaysEq(), None)
        self.no(AlwaysEq(), object())
        self.no(AlwaysEq(), "a")
        self.no(OnlyNe(), OnlyNe())
        self.no(OnlyNe(), None)
        self.no(StrEq("a"), None)
        self.no(StrEq("a"), 1)
        self.no(WithMeta, None)  # a class whose metaclass defines __eq__
        self.no(WithMeta, Plain)
        self.no(Data(), Data())
        self.no(Data(), None, counter=False)
        self.no(decimal.Decimal(1), 1)
        self.no(decimal.Decimal(1), None, counter=False)
        self.no(datetime.date(2020, 1, 1), datetime.date(2020, 1, 1))
        self.no({1: 2}.keys(), {1})
        self.no(bytearray(b"a"), b"a")
        self.no(bytearray(b"a"), object(), counter=False)
        self.no(b"a", "a", counter=False)  # bytes with str or int warns under -b: not in the disjoint list
        self.no(b"a", 1, counter=False)
        self.no(len, isinstance, counter=False)
        self.no(len, 1, counter=False)

    def test_numpy_and_pandas(self):
        try:
            import numpy as np
        except Exception:  # noqa: BLE001
            self.skipTest("numpy not installed")
        self.no(np.int64(1), 1)
        self.no(np.bool_(True), True)
        self.no(np.float64(1.0), None, counter=False)
        self.no(np.array([1]), None, counter=False)

    def test_a_class_seen_with_differing_facts_is_refused(self):
        a = facts(Plain, uniq=0)
        self.assertIs(static._cmp_pair(a, facts(type(None)), lambda: True), False)
        self.assertIs(static._cmp_pair(facts(type(None)), a, lambda: True), False)
        self.assertIs(static._cmp_pair(a, a, lambda: True), False)


class SwapSite(unittest.TestCase):
    def site(self, old, new, ln=1):
        return static._cmp_swap_site(old, new, ln)

    def test_the_four_swaps(self):
        self.assertEqual(self.site("x = a is b\n", "x = a == b\n"), (1, 4, 1, 10))
        self.assertEqual(self.site("x = a == b\n", "x = a is b\n"), (1, 4, 1, 10))
        self.assertEqual(self.site("x = a is not b\n", "x = a != b\n"), (1, 4, 1, 14))
        self.assertEqual(self.site("x = a != b\n", "x = a is not b\n"), (1, 4, 1, 10))

    def test_places(self):
        self.assertEqual(self.site("f = lambda a: a is None\n", "f = lambda a: a == None\n"), (1, 14, 1, 23))
        self.assertEqual(self.site("y = [i for i in z if i is not q]\n", "y = [i for i in z if i != q]\n"), (1, 21, 1, 31))
        self.assertEqual(self.site("class C:\n    x = a is b\n", "class C:\n    x = a == b\n", 2), (2, 8, 2, 14))
        self.assertEqual(self.site("@d(a is b)\ndef f():\n    pass\n", "@d(a == b)\ndef f():\n    pass\n", 1), (1, 3, 1, 9))
        self.assertEqual(self.site("def f():\n    if (a\n            is b):\n        return 1\n    return 2\n",
                                   "def f():\n    if (a\n            == b):\n        return 1\n    return 2\n", 3), (2, 8, 3, 16))
        self.assertEqual(self.site("x = f(a is b)[c < d]\n", "x = f(a == b)[c < d]\n"), (1, 6, 1, 12))
        self.assertEqual(self.site("x = (a is b) is c\n", "x = (a is b) == c\n"), (1, 4, 1, 17))

    def test_not_a_single_swap(self):
        for old, new in (
            ("x = a == b\n", "x = a != b\n"),
            ("x = a is b\n", "x = a is not b\n"),
            ("x = a is not b\n", "x = a == b\n"),
            ("x = a is b\n", "x = a != b\n"),
            ("x = a < b\n", "x = a <= b\n"),
            ("x = a is b is c\n", "x = a == b is c\n"),
            ("x = a < b is c\n", "x = a < b == c\n"),
            ("x = a is b\n", "x = c == b\n"),
            ("x = a is b\n", "x = a == c\n"),
            ("x = a is b\n", "y = a == b\n"),
            ("x = (a is b) and (c is d)\n", "x = (a == b) and (c == d)\n"),
            ("x = (a is b) and c\n", "x = (a == b) or c\n"),
            ("x = 'a is b'\n", "x = 'a == b'\n"),
            ("x = a is b  # is\n", "x = a is b  # ==\n"),
            ("x = a is b\n", "x = a is b\n"),
            ("x = a is 1\n", "x = a == 1.0\n"),
            ("x = a is 1\n", "x = a == True\n"),
            ("x = a in b\n", "x = a == b\n"),
            ("x = not a is b\n", "x = a == b\n"),
        ):
            self.assertIsNone(self.site(old, new), (old, new))

    def test_other_statement_changed_too(self):
        old = "def f():\n    x = a is b\n    return 1\n"
        self.assertEqual(self.site(old, "def f():\n    x = a == b\n    return 1\n", 2), (2, 8, 2, 14))
        self.assertIsNone(self.site(old, "def f():\n    x = a == b\n    return 2\n", 2))
        self.assertIsNone(self.site(old, "def g():\n    x = a == b\n    return 1\n", 2))
        self.assertIsNone(self.site(old, "def f():\n    x = a == b\n    return 1\n    pass\n", 2))


class Literal(unittest.TestCase):
    def lit(self, text):
        import ast

        return static._cmp_literal(ast.parse(text, mode="eval").body)

    def test_literals_the_compiler_warns_about(self):
        for text in ("1", "'a'", "b'a'", "1.5", "1j", "(1, 2)", "()", "-1", "1 + 2", "'a' 'b'", "[1]", "{1: 2}", "{1}", "f'a'", "not 1"):
            self.assertIs(self.lit(text), True, text)

    def test_not_literals(self):
        for text in ("None", "True", "False", "...", "x", "x.y", "f()", "x[0]", "(x, 1)", "-x", "[x]", "lambda: x"):
            self.assertIs(self.lit(text), False, text)

    def test_the_compiler_warns_exactly_there(self):
        import warnings

        import ast

        # want None: depends on the Python version (constant folding before or after the check)
        for text, want in (("x is 1", True), ("x is 'a'", True), ("x is not (1, 2)", None), ("x is -1", None),
                           ("x is 1 + 2", None), ("x is b'a'", True), ("x is 1.5", True), ("x is ()", None),
                           ("x is None", False), ("x is not True", False), ("x is ...", False), ("x is y", False),
                           ("x is (y, 1)", False), ("1 is x", True), ("x is [1]", None), ("x is f'a'", None)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                compile(text, "<t>", "eval")
            if want is not None:
                self.assertEqual(bool(w), want, text)
            n = ast.parse(text, mode="eval").body
            if w:  # wherever the compiler warns, the rule refuses
                self.assertTrue(static._cmp_literal(n.left) or static._cmp_literal(n.comparators[0]), text)


class Audit(unittest.TestCase):
    def test_the_pinned_tree_has_the_audited_text(self):
        self.assertIs(static._cmp_expr_audit(REPO, "sqlglot/parser.py", ""), True)

    def test_a_swap_inside_expression_eq_is_refused(self):
        with open(os.path.join(REPO, static._CMP_EXPR_FILE), encoding="utf-8") as fh:
            src = fh.read()
        self.assertEqual(src.count("return self is other or (type(self) is type(other)"), 1)
        for a, b in (("return self is other or (type(self)", "return self == other or (type(self)"),
                     ("(type(self) is type(other) and", "(type(self) == type(other) and"),
                     ("and hash(self) == hash(other))", "and hash(self) is hash(other))")):
            self.assertEqual(src.count(a), 1, a)
            self.assertIs(static._cmp_expr_audit(REPO, static._CMP_EXPR_FILE, src.replace(a, b)), False)
        self.assertIs(static._cmp_expr_audit(REPO, static._CMP_EXPR_FILE, src), True)
        self.assertIs(static._cmp_expr_audit(REPO, static._CMP_EXPR_FILE, src + "\nExpression.__eq__ = None\n"), True)

    def test_no_other_binding_of_eq_in_the_package(self):
        """The audit of the pinned tree: nothing assigns Expression.__eq__ / __ne__ from outside."""
        hits = []
        for dp, _dns, fns in os.walk(os.path.join(REPO, "sqlglot")):
            for fn in fns:
                if fn.endswith(".py"):
                    with open(os.path.join(dp, fn), encoding="utf-8") as fh:
                        for i, line in enumerate(fh, 1):
                            if ("__eq__" in line or "__ne__" in line) and "def __eq__" not in line and "def __ne__" not in line:
                                hits.append((os.path.relpath(os.path.join(dp, fn), REPO), i, line.strip()))
        allowed = [h for h in hits if h[2] in ("return not self.__eq__(other)",)]
        self.assertEqual([h for h in hits if h not in allowed], [])


class EndToEnd(unittest.TestCase):
    """The rule through `analyze_patch`, with the shipped maps, on the swap of the organisers' sample
    (dialect.py: `self.normalization_strategy is NormalizationStrategy.UPPERCASE`)."""

    OLD = "            if self.normalization_strategy is NormalizationStrategy.UPPERCASE\n"
    NEW = "            if self.normalization_strategy == NormalizationStrategy.UPPERCASE\n"
    PATH = "sqlglot/dialects/dialect.py"

    @classmethod
    def setUpClass(cls):
        import difflib
        import json
        import tempfile

        root = os.path.join(HERE, "..")
        cls.map_path = os.environ.get("BR_MAP") or os.path.join(root, "map", "blast.map")
        cls.cmp_path = os.path.join(os.path.dirname(cls.map_path), "cmp.json")
        with open(os.path.join(REPO, cls.PATH), encoding="utf-8") as fh:
            old = fh.readlines()
        assert old.count(cls.OLD) == 1
        new = [cls.NEW if line == cls.OLD else line for line in old]
        cls.patch = f"diff --git a/{cls.PATH} b/{cls.PATH}\n" + "".join(
            difflib.unified_diff(old, new, f"a/{cls.PATH}", f"b/{cls.PATH}", n=3))
        cls.other = cls.patch.replace("+" + cls.NEW, "+" + cls.NEW.replace("==", "!="))
        with open(cls.cmp_path, encoding="utf-8") as fh:
            cls.doc = json.load(fh)
        with open(os.path.join(os.path.dirname(cls.map_path), "..", "tests.json")
                  if os.path.exists(os.path.join(os.path.dirname(cls.map_path), "..", "tests.json"))
                  else os.environ["BR_TESTS"], encoding="utf-8") as fh:
            cls.tests = json.load(fh)["tests"]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.M = static.Map(cls.map_path, cls.tests)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        os.environ.pop("BR_CMPMAP", None)
        static._CMP_DOC = static._CMP_FRESH = None
        static.CMP = True

    def run_with(self, doc=None, raw=None, missing=False, patch=None, cmp=True):
        import json

        p = os.path.join(self.tmp.name, "cmp.json")
        if missing:
            p = os.path.join(self.tmp.name, "nothing-here.json")
        elif raw is not None:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(raw)
        else:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
        os.environ["BR_CMPMAP"] = p
        static._CMP_DOC = static._CMP_FRESH = None
        static.CMP = cmp
        A = static.analyze_patch(patch or self.patch, REPO, self.M)
        return A, [n.split(" cmp: ", 1)[1] for n in A.notes if " cmp: " in n]

    def here(self):
        return dict(self.doc, python=list(sys.version_info[:2]))

    def test_shipped_map_is_for_the_image_python(self):
        self.assertEqual(self.doc["python"], [3, 13])
        self.assertEqual(self.doc["v"], 1)

    def test_fires_and_selects_nothing(self):
        A, notes = self.run_with(self.here())
        self.assertEqual(notes, ["1 class pairs, all identity-equivalent"])
        self.assertEqual((len(A.cands), A.broad, A.changes, A.opaque), (0, False, [], False))

    def test_off_gives_the_usual_candidates(self):
        base, notes = self.run_with(self.here(), cmp=False)
        self.assertEqual(notes, [])
        self.assertGreater(len(base.cands), 0)
        for kw, want in (
            (dict(missing=True), "no class map for this Python"),
            (dict(raw="{not json"), "no class map for this Python"),
            (dict(doc=dict(self.doc, python=[2, 7])), "no class map for this Python"),
            (dict(doc=dict(self.here(), v=2)), "no class map for this Python"),
            (dict(doc=dict(self.here(), digest="0" * 64)), "class map is stale for this checkout or test list"),
            (dict(doc=dict(self.here(), tests=self.doc["tests"][1:])), "class map is stale for this checkout or test list"),
            (dict(doc=dict(self.here(), sites={})), None),
            (dict(doc=dict(self.here(), classes=[])), "rule failed"),
        ):
            A, notes = self.run_with(**kw)
            self.assertEqual((A.cands, A.broad, A.changes), (base.cands, base.broad, base.changes), kw.keys())
            if want is None:
                self.assertEqual(notes, [])
            else:
                self.assertTrue(notes and notes[0].startswith(want), (notes, want))

    def test_another_operator_is_not_this_rule(self):
        A, notes = self.run_with(self.here(), patch=self.other)
        self.assertEqual(notes, [])
        self.assertGreater(len(A.cands), 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
