"""Identifier reference index over the sqlglot checkout (built once, at image build time).

build(repo) -> {"files": [...], "refs": {name: [code, ...]}} where each code packs
(file index, line, flags): code = (file << 23) | (line << 3) | flags.
flag 1: the name is followed by "." (its attributes are read there), so mere mentions (a
dict key, a base-class list, an isinstance check) can be told apart.
flag 2: the name is followed by "=" (an assignment target or keyword argument: not a read).
flag 4: the name is a word inside a string literal (getattr / dispatch by name); also flag 1.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import sys
import tokenize

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _py_files(repo: str) -> list[str]:
    out = []
    for part in ("sqlglot", "tests"):
        for dp, dns, fns in os.walk(os.path.join(repo, part)):
            dns[:] = sorted(d for d in dns if d not in ("__pycache__", "fixtures"))
            for fn in sorted(fns):
                if fn.endswith(".py"):
                    out.append(os.path.relpath(os.path.join(dp, fn), repo))
    return sorted(out)


def build(repo: str) -> dict:
    files = _py_files(repo)
    refs: dict[str, list[int]] = {}
    classes: dict[str, list[tuple[int, int, str]]] = {}
    for fi, rel in enumerate(files):
        try:
            src = open(os.path.join(repo, rel), encoding="utf-8").read()
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except (OSError, tokenize.TokenError, SyntaxError, UnicodeDecodeError):
            continue
        try:
            spans = []

            def visit(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.ClassDef):
                        start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                        q = f"{prefix}.{child.name}" if prefix else child.name
                        spans.append((start, child.end_lineno, q))
                        visit(child, q)
                    elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit(child, (prefix + "." if prefix else "") + child.name + ".<locals>")
                    else:
                        visit(child, prefix)

            visit(ast.parse(src), "")
            classes[rel] = spans
        except SyntaxError:
            classes[rel] = []
        for i, t in enumerate(toks):
            if t.type == tokenize.NAME:
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                flag = 0
                if nxt is not None and nxt.type == tokenize.OP:
                    if nxt.string == ".":
                        flag = 1
                    elif nxt.string == "=":
                        flag = 2
                refs.setdefault(t.string, []).append((fi << 23) | (t.start[0] << 3) | flag)
            elif t.type == tokenize.STRING:
                for ln_off, chunk in enumerate(t.string.split("\n")):
                    for w in set(_WORD.findall(chunk)):
                        refs.setdefault(w, []).append((fi << 23) | ((t.start[0] + ln_off) << 3) | 5)
    return {"files": files, "refs": {k: sorted(set(v)) for k, v in refs.items()}, "classes": classes}


class RefIndex:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        self.files: list[str] = d["files"]
        self.refs: dict[str, list[int]] = d["refs"]
        self.classes: dict[str, list[list]] = d.get("classes", {})

    def class_at(self, path: str, line: int) -> str | None:
        """Qualified name of the innermost class whose span holds the line (header included)."""
        best = None
        for start, end, q in self.classes.get(path, ()):
            if start <= line <= end and (best is None or end - start < best[1] - best[0]):
                best = (start, end, q)
        return best[2] if best else None

    def entries(self, name: str) -> list[tuple[str, int, int]]:
        return [(self.files[c >> 23], (c >> 3) & 0xFFFFF, c & 7) for c in self.refs.get(name, ())]

    def lines(self, name: str, attr_only: bool = False, reads_only: bool = False) -> list[tuple[str, int]]:
        out = []
        for code in self.refs.get(name, ()):
            if attr_only and not (code & 1):
                continue
            if reads_only and (code & 2):
                continue
            out.append((self.files[code >> 23], (code >> 3) & 0xFFFFF))
        return out

    def count(self, name: str) -> int:
        return len(self.refs.get(name, ()))


if __name__ == "__main__":
    repo, out = sys.argv[1], sys.argv[2]
    d = build(repo)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(d, fh, separators=(",", ":"))
    n = sum(len(v) for v in d["refs"].values())
    print(f"refindex: {len(d['files'])} files, {len(d['refs'])} names, {n} references", file=sys.stderr)
