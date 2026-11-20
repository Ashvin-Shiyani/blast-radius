"""Static scope tables of a Python source file, for classifying changed lines."""

from __future__ import annotations

import ast
import io
import tokenize


class FileScopes:
    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        # (start incl. decorators, end, body_start) for def / async def
        self.funcs: list[tuple[int, int, int]] = []
        # lambda bodies: (line, col, end_line, end_col)
        self.lambdas: list[tuple[int, int, int, int]] = []
        self.classes: list[tuple[int, int, str]] = []
        self.stmts: list[tuple[int, int]] = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                self.funcs.append((start, node.end_lineno, node.body[0].lineno))
            elif isinstance(node, ast.Lambda):
                b = node.body
                self.lambdas.append((b.lineno, b.col_offset, b.end_lineno, b.end_col_offset))
            elif isinstance(node, ast.ClassDef):
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                self.classes.append((start, node.end_lineno, node.name))
            if isinstance(node, ast.stmt):
                self.stmts.append((node.lineno, node.end_lineno))

    def func_of(self, line: int) -> tuple[int, int, int] | None:
        """Innermost def containing the line (decorators through end)."""
        best = None
        for f in self.funcs:
            if f[0] <= line <= f[1] and (best is None or f[1] - f[0] < best[1] - best[0]):
                best = f
        return best

    def stmt_of(self, line: int) -> tuple[int, int] | None:
        best = None
        for s in self.stmts:
            if s[0] <= line <= s[1] and (best is None or s[1] - s[0] < best[1] - best[0]):
                best = s
        return best

    def stmt_within(self, line: int, lo: int, hi: int) -> tuple[int, int]:
        """Innermost statement containing the line that lies inside [lo, hi]."""
        best = None
        for s in self.stmts:
            if lo <= s[0] and s[1] <= hi and s[0] <= line <= s[1]:
                if best is None or s[1] - s[0] < best[1] - best[0]:
                    best = s
        return best or (line, line)

    def lambda_body_at(self, line: int, a: int, b: int) -> tuple[int, int] | None:
        """Line range of the innermost lambda body that holds columns [a, b) of `line`."""
        best = None
        for (l0, c0, l1, c1) in self.lambdas:
            inside = (l0, c0) <= (line, a) and (line, b) <= (l1, c1)
            if inside and (best is None or (l1 - l0, c1 - c0) < (best[2] - best[0], best[3] - best[1])):
                best = (l0, c0, l1, c1)
        return (best[0], best[2]) if best else None

    def lambdas_touching(self, line: int) -> list[tuple[int, int]]:
        return [(l0, l1) for (l0, _, l1, _) in self.lambdas if l0 <= line <= l1]


def significant_tokens(source: str) -> list[tuple[int, str]] | None:
    """Token stream without comments and non-logical newlines; None if it cannot tokenize."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL):
                continue
            out.append((tok.type, tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return out
