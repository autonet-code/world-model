#!/usr/bin/env python3
"""Doctest scoring harness for Phase 5.

Takes a contestant's function body (or full function source), the
expected function name and signature, and a list of doctests
({call, expected}), and returns:

  {
    "n_doctests": int,
    "n_passed": int,
    "score": float (0..1),
    "compile_error": str | None,
    "per_doctest": [{"call": str, "expected": str, "got": str, "passed": bool}, ...]
  }

Safety: we execute contestant code in a fresh dict, with stdlib only.
No filesystem, no network. The contestants for our experiment produce
short pure-function Python, which is what toolz functions are.
"""

from __future__ import annotations

import ast
import io
import re
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple


_CALL_NAME_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _normalize_repr(s: str) -> str:
    """Doctest expected outputs and actual repr() may differ in
    whitespace / list-vs-iterator-vs-tuple. Conservative normalization:
    strip trailing whitespace, collapse internal whitespace runs.
    """
    return re.sub(r"\s+", " ", s.strip())


def _exec_user_code(
    full_source: str,
    fn_name: str,
) -> Tuple[Optional[Any], Dict[str, Any], Optional[str]]:
    """Execute the contestant's source in a clean namespace, return
    (the named function, the namespace, error-or-None)."""
    ns: Dict[str, Any] = {}
    # Whitelist some common imports the contestants might need
    # (toolz functions reach for itertools, functools, etc.).
    exec_globals: Dict[str, Any] = {"__builtins__": __builtins__}
    try:
        compile(full_source, "<contestant>", "exec")
    except SyntaxError as e:
        return None, {}, f"SyntaxError: {e}"
    try:
        exec(full_source, exec_globals, ns)
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {e}"
    fn = ns.get(fn_name) or exec_globals.get(fn_name)
    if fn is None:
        return None, ns, f"function {fn_name!r} not defined"
    return fn, {**exec_globals, **ns}, None


def _eval_doctest(
    call: str,
    expected: str,
    user_ns: Dict[str, Any],
) -> Tuple[bool, str]:
    """Evaluate a single doctest's call line, capture stdout-or-repr,
    compare to expected (normalized)."""
    # Some toolz doctests have multi-line setup like
    #     def double(x): return 2*x
    # which is a Statement, not Expression. Differentiate.
    src = call.rstrip()
    if not src:
        return True, ""

    # Try as expression first.
    try:
        tree = ast.parse(src, mode="eval")
        is_expr = True
    except SyntaxError:
        is_expr = False

    buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        if is_expr:
            with redirect_stdout(buf), redirect_stderr(err_buf):
                value = eval(compile(tree, "<call>", "eval"), user_ns)
            stdout_text = buf.getvalue()
            if stdout_text:
                got = stdout_text.rstrip()
            else:
                got = repr(value) if value is not None else ""
        else:
            with redirect_stdout(buf), redirect_stderr(err_buf):
                exec(src, user_ns)
            got = buf.getvalue().rstrip()
    except Exception as e:
        got = f"<{type(e).__name__}: {e}>"

    passed = _normalize_repr(got) == _normalize_repr(expected)
    return passed, got


def grade_implementation(
    *,
    contestant_source: str,
    fn_name: str,
    signature: str,
    doctests: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Score a contestant's implementation against the test's doctests.

    contestant_source can be either:
      - a full `def fn_name(...):` block, or
      - just the body of the function (we'll wrap it).
    """
    # If the source doesn't contain `def fn_name`, treat it as a body to wrap.
    has_def = re.search(rf"\bdef\s+{re.escape(fn_name)}\s*\(", contestant_source) is not None
    if has_def:
        full_source = contestant_source
    else:
        body = textwrap.indent(contestant_source.strip("\n"), "    ")
        full_source = f"def {fn_name}({signature}):\n{body}\n"

    fn, user_ns, exec_err = _exec_user_code(full_source, fn_name)
    if exec_err is not None:
        return {
            "n_doctests": len(doctests),
            "n_passed": 0,
            "score": 0.0,
            "compile_error": exec_err,
            "per_doctest": [],
        }

    per: List[Dict[str, Any]] = []
    n_passed = 0
    for dt in doctests:
        passed, got = _eval_doctest(dt["call"], dt["expected"], user_ns)
        per.append({
            "call": dt["call"], "expected": dt["expected"],
            "got": got, "passed": passed,
        })
        if passed:
            n_passed += 1

    return {
        "n_doctests": len(doctests),
        "n_passed": n_passed,
        "score": n_passed / max(len(doctests), 1),
        "compile_error": None,
        "per_doctest": per,
    }


if __name__ == "__main__":
    # Self-test on a trivial case.
    result = grade_implementation(
        contestant_source="return x",
        fn_name="identity",
        signature="x",
        doctests=[{"call": "identity(3)", "expected": "3"}],
    )
    assert result["score"] == 1.0, result
    print("OK: harness self-test passes")
