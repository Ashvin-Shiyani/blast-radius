"""Weak-mutation probe: does a one-expression change ever change a value while tests run?

For a patch that changes one expression inside a function body, the patched module is
rewritten so that expression becomes

    _br_probe(lambda: (<original expression>), lambda: (<patched expression>))

which evaluates both, notes whether they ever differ, and returns the patched value, so the
tests still run the patched code. Only side-effect-free expressions are probed (names,
attributes, constants, operators, subscripts, and a few pure calls), because both versions
are evaluated. Results go to an append-only file so forked pool workers report too.

If the tests reach the expression and the two versions never differ, the change was
invisible to everything those tests ran.
"""

from __future__ import annotations

import ast
import builtins
import os

PURE_CALLS = {"len", "isinstance", "issubclass", "bool", "type", "callable", "hasattr"}
PURE_METHODS = {"get", "startswith", "endswith", "lower", "upper", "isdigit", "isupper", "islower", "keys"}
PROBE_TYPES = (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp)


def _pure(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, (ast.NamedExpr, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom,
                          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Starred,
                          ast.FormattedValue, ast.JoinedStr)):
            return False
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in PURE_CALLS:
                continue
            if isinstance(f, ast.Attribute) and f.attr in PURE_METHODS:
                continue
            return False
    return True


def _abs_offset(lines: list[str], lineno: int, col_bytes: int) -> int:
    pre = sum(len(x) + 1 for x in lines[: lineno - 1])
    line = lines[lineno - 1]
    return pre + len(line.encode("utf-8")[:col_bytes].decode("utf-8", "ignore"))


def _smallest(tree: ast.AST, line: int, a: int, b: int, src_lines: list[str]):
    """Smallest probe-able expression node containing chars [a, b) of `line`."""
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, PROBE_TYPES) or not hasattr(node, "lineno"):
            continue
        s = _abs_offset(src_lines, node.lineno, node.col_offset)
        e = _abs_offset(src_lines, node.end_lineno, node.end_col_offset)
        la = _abs_offset(src_lines, line, len(src_lines[line - 1][:a].encode("utf-8")))
        lb = _abs_offset(src_lines, line, len(src_lines[line - 1][:b].encode("utf-8")))
        if s <= la and lb <= e and (best is None or e - s < best[2] - best[1]):
            best = (node, s, e)
    return best


def _in_function_body(tree: ast.AST, node: ast.AST) -> bool:
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            b0 = f.body[0].lineno
            if b0 <= node.lineno and node.end_lineno <= f.end_lineno:
                return True
    return False


def instrument(old_src: str, new_src: str, line_old: int, line_new: int,
               span_old: tuple[int, int], span_new: tuple[int, int], report_path: str) -> str | None:
    """Return the patched source with the changed expression probed, or None if unsafe."""
    try:
        t_old = ast.parse(old_src)
        t_new = ast.parse(new_src)
    except SyntaxError:
        return None
    ol, nl = old_src.split("\n"), new_src.split("\n")
    bo = _smallest(t_old, line_old, *span_old, ol)
    bn = _smallest(t_new, line_new, *span_new, nl)
    if bo is None or bn is None:
        return None
    node_o, so, eo = bo
    node_n, sn, en = bn
    if type(node_o) is not type(node_n) or not _pure(node_o) or not _pure(node_n):
        return None
    if (so != sn or len(old_src) - eo != len(new_src) - en) and not isinstance(node_o, ast.BoolOp):
        # The old and the new expression do not occupy the same place: comparing their values
        # says nothing about the statement (a + b * c -> a * b * c would compare a + b * c
        # with a * b). Only an and/or chain regrouped by the edit is still compared soundly.
        return None
    if not _in_function_body(t_new, node_n):
        return None
    text_o, text_n = old_src[so:eo], new_src[sn:en]
    if "\n" in text_o or "\n" in text_n:
        text_o, text_n = f"({text_o})", f"({text_n})"
    rep = f"_br_probe(lambda: ({text_o}), lambda: ({text_n}))"
    out = new_src[:sn] + rep + new_src[en:]
    # fallback only for processes without the installed probe (e.g. a `python -c`
    # subprocess): defined at module level only if builtins lacks it, so it never shadows it
    out += (
        "\n\nimport builtins as _br_builtins\n"
        "if not hasattr(_br_builtins, '_br_probe'):\n"
        "    _br_seen = []\n"
        "    def _br_probe(_o, _n, _p=" + repr(report_path) + "):\n"
        "        if not _br_seen:\n"
        "            _br_seen.append(1)\n"
        "            import os as _os\n"
        "            _fd = _os.open(_p, _os.O_WRONLY | _os.O_APPEND | _os.O_CREAT, 0o666)\n"
        "            try:\n"
        "                _os.write(_fd, b'X')\n"
        "            finally:\n"
        "                _os.close(_fd)\n"
        "        return _n()\n"
    )
    try:
        compile(out, "<probe>", "exec", dont_inherit=True)
    except SyntaxError:
        return None
    return out


def install(report_path: str, abort_on_infection: bool = False) -> None:
    """Define builtins._br_probe for this process and its forks: writes 'E' on the first
    evaluation and 'I' on the first infection in each process."""
    state = {"e": False, "i": False, "pid": None}

    def _write(tag: bytes) -> None:
        fd = os.open(report_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
        try:
            os.write(fd, tag)
        finally:
            os.close(fd)

    def _br_probe(fo, fn):
        pid = os.getpid()
        if state["pid"] != pid:  # a forked worker: report again
            state.update(e=False, i=False, pid=pid)
        ro = rn = None
        eo = en = None
        try:
            ro = fo()
        except BaseException as exc:  # noqa: BLE001
            eo = exc
        try:
            rn = fn()
        except BaseException as exc:  # noqa: BLE001
            en = exc
        if not state["e"]:
            state["e"] = True
            _write(b"E")
        if not state["i"]:
            if eo is not None or en is not None:
                same = eo is not None and en is not None and type(eo) is type(en)
            elif ro is rn:
                same = True
            elif type(ro) is type(rn) and type(ro) in (bool, int, float, str, bytes, type(None)):
                same = ro == rn
            else:
                same = False
            if not same:
                state["i"] = True
                _write(b"I")
                if abort_on_infection:
                    import signal

                    try:
                        os.killpg(os.getpgid(0), signal.SIGKILL)
                    except OSError:
                        os._exit(3)
        if en is not None:
            raise en
        return rn

    builtins._br_probe = _br_probe


def read_report(report_path: str) -> tuple[bool, bool, bool]:
    """(evaluated, infected, fallback_hit)."""
    try:
        data = open(report_path, "rb").read()
    except OSError:
        return False, False, False
    return b"E" in data, b"I" in data, b"X" in data


# ------------------------------------------------- reach-limited replay (strict probe)

def _line_starts(lines: list[str]) -> list[int]:
    out = [0]
    for x in lines:
        out.append(out[-1] + len(x) + 1)
    return out


def _off(lines: list[str], starts: list[int], lineno: int, col_bytes: int) -> int:
    line = lines[lineno - 1]
    if line.isascii():
        return starts[lineno - 1] + col_bytes
    return starts[lineno - 1] + len(line.encode("utf-8")[:col_bytes].decode("utf-8", "ignore"))


def _scope_ok(node: ast.AST, par: dict) -> bool:
    """True if `node` is evaluated in the scope of a function or lambda body. In a class body
    or at module level, wrapping it in a lambda could change what its names refer to; inside
    an f-string positions are not trusted. A default, decorator or annotation of a function
    belongs to the scope around that function."""
    child = node
    while True:
        p = par.get(child)
        if p is None or isinstance(p, (ast.Module, ast.ClassDef, ast.JoinedStr, ast.FormattedValue)):
            return False
        if isinstance(p, ast.Lambda):
            if child is p.body:
                return True
        elif isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(child is b for b in p.body):
                return True
        child = p


def _evaluates(node: ast.AST) -> bool:
    """The expression loads at least one name, attribute, item or call result, so the compiler
    cannot fold it away: evaluating it always runs an instruction on one of its own lines."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == "__debug__":
            return False
    return any(isinstance(n, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)) for n in ast.walk(node))


_OPERATORS = (ast.Compare, ast.BinOp, ast.UnaryOp)


def _shape(node: ast.AST):
    """What an operator expression evaluates besides its operators and constant operands: the
    operand expressions, in order. None unless it is a comparison of two operands, a binary
    or a unary operation whose operands are constants, such operations again, or expressions
    free of operators, conditionals (and / or / if-else: they decide which operands run) and
    item access (it can insert, as in a defaultdict).

    Two expressions of the same shape evaluate the very same operands the same number of
    times; they can differ only in an operator or a constant."""

    def operand(n):
        if isinstance(n, ast.Constant):
            return "const"
        if isinstance(n, _OPERATORS):
            return _shape(n)
        d = ast.dump(n)
        if any(k in d for k in ("BoolOp(", "IfExp(", "Subscript(", "Compare(", "BinOp(", "UnaryOp(")):
            return None
        return d

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            return None
        parts = [operand(node.left), operand(node.comparators[0])]
        kind = "compare"
    elif isinstance(node, ast.BinOp):
        parts = [operand(node.left), operand(node.right)]
        kind = "binary"
    elif isinstance(node, ast.UnaryOp):
        parts = [operand(node.operand)]
        kind = "unary"
    else:
        return None
    if any(p is None for p in parts):
        return None
    return (kind, *parts)


def instrument_wide(old_src: str, new_src: str, line_old: int, line_new: int,
                    span_old: tuple[int, int], span_new: tuple[int, int], report_path: str):
    """(probed source, first line, last line of the original expression) or None.

    Like instrument(), for the reach-limited replay, and stricter: the probed pair is the
    smallest pair of operator expressions that contain the change, occupy the same place in
    the old and the new source (same start, same distance from the end of the file), have the
    same _shape() (same operands, so the old and the new code read exactly the same things
    and differ only in an operator or a constant), are side-effect free by _pure(), are
    evaluated in a function or lambda scope, and cannot be folded away by the compiler."""
    try:
        t_old = ast.parse(old_src)
        t_new = ast.parse(new_src)
    except SyntaxError:
        return None
    ol, nl = old_src.split("\n"), new_src.split("\n")
    if line_old != line_new or not (0 < line_old <= len(ol)) or not (0 < line_new <= len(nl)):
        return None
    if ol[:line_old - 1] != nl[:line_new - 1] or ol[line_old:] != nl[line_new:]:
        return None  # more than this one line differs

    def cands(tree, lines, total, line, a, b):
        starts = _line_starts(lines)
        la, lb = starts[line - 1] + a, starts[line - 1] + b
        out = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.expr) or isinstance(node, (ast.Slice, ast.Starred)):
                continue
            if getattr(node, "end_lineno", None) is None or not (node.lineno <= line <= node.end_lineno):
                continue
            s = _off(lines, starts, node.lineno, node.col_offset)
            e = _off(lines, starts, node.end_lineno, node.end_col_offset)
            if s <= la and lb <= e:
                out.setdefault((s, total - e), (node, s, e))
        return out

    co = cands(t_old, ol, len(old_src), line_old, *span_old)
    cn = cands(t_new, nl, len(new_src), line_new, *span_new)
    par_o: dict = {}
    par_n: dict = {}
    for tree, par in ((t_old, par_o), (t_new, par_n)):
        for p in ast.walk(tree):
            for c in ast.iter_child_nodes(p):
                par[c] = p
    for key in sorted(set(co) & set(cn), key=lambda k: co[k][2] - co[k][1]):
        node_o, so, eo = co[key]
        node_n, sn, en = cn[key]
        if not isinstance(getattr(node_o, "ctx", ast.Load()), ast.Load):
            continue
        if not isinstance(getattr(node_n, "ctx", ast.Load()), ast.Load):
            continue
        if not _evaluates(node_o) or not _evaluates(node_n):
            continue
        shape = _shape(node_o)
        if shape is None or shape != _shape(node_n):
            continue
        if not _pure(node_o) or not _pure(node_n):
            return None  # every larger pair contains this one
        if not _scope_ok(node_o, par_o) or not _scope_ok(node_n, par_n):
            return None
        rep = f"_br_probe(lambda: ({old_src[so:eo]}), lambda: ({new_src[sn:en]}))"
        out = new_src[:sn] + rep + new_src[en:]
        out += (
            "\n\nimport builtins as _br_builtins\n"
            "if not hasattr(_br_builtins, '_br_probe'):\n"
            "    def _br_probe(_o, _n, _p=" + repr(report_path) + "):\n"
            "        import os as _os\n"
            "        _fd = _os.open(_p, _os.O_WRONLY | _os.O_APPEND | _os.O_CREAT, 0o666)\n"
            "        try:\n"
            "            _os.write(_fd, b'X')\n"
            "        finally:\n"
            "            _os.close(_fd)\n"
            "        return _n()\n"
        )
        try:
            compile(out, "<probe>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            return None
        return out, node_o.lineno, node_o.end_lineno
    return None


def install_strict() -> dict:
    """The probe of the reach-limited replay, for this process only. Counts evaluations, and
    calls anything but two identical plain values a difference: an exception on either side,
    two equal but distinct objects, 0.0 and -0.0. Returns its state: {"n": evaluations,
    "i": a difference was seen}."""
    state = {"n": 0, "i": False}
    plain = (bool, int, float, str, bytes, type(None))

    def _br_probe(fo, fn):
        state["n"] += 1
        try:
            ro = fo()
        except Exception:  # noqa: BLE001
            state["i"] = True
            return fn()
        try:
            rn = fn()
        except Exception:  # noqa: BLE001
            state["i"] = True
            raise
        if ro is not rn:
            t = type(ro)
            if t is not type(rn) or t not in plain or ro != rn or (t is float and repr(ro) != repr(rn)):
                state["i"] = True
        return rn

    builtins._br_probe = _br_probe
    return state
