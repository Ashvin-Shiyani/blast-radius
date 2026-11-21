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
