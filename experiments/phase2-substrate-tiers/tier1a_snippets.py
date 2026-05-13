"""Tier 1A snippet set: 12 beginner-Python pieces in 6 categories.

Each entry has:
  id          — short label like "S1"
  category    — one of: gold, quirky, complex_correct, buggy,
                narrow, bad_all
  snippet     — the Python code (multi-line string, dedented)
  rationale   — what category each one is meant to illustrate
                (for human readers, not the LLM)

Categories map to the Tier 0 W shape:
  gold            ↔ W1   (correct, simple, idiomatic)
  quirky          ↔ W2   (correct, simple, NOT idiomatic)
  complex_correct ↔ W3   (correct, idiomatic, complex)
  buggy           ↔ W4   (incorrect, otherwise nice)
  narrow          ↔ W5   (correctness-only post)
  bad_all         ↔ W6   (incorrect, complex, unidiomatic)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Snippet:
    id: str
    category: str
    snippet: str
    rationale: str


SNIPPETS: list[Snippet] = [
    # --- gold (S1, S2): correct, simple, idiomatic ---
    Snippet(
        id="S1",
        category="gold",
        snippet=(
            "def first_n_squares(n):\n"
            "    return [x * x for x in range(n)]\n"
        ),
        rationale="list comprehension, clearly pythonic",
    ),
    Snippet(
        id="S2",
        category="gold",
        snippet=(
            "def total(numbers):\n"
            "    return sum(numbers)\n"
        ),
        rationale="uses builtin sum, minimal",
    ),

    # --- quirky (S3, S4): correct, simple, NOT idiomatic ---
    Snippet(
        id="S3",
        category="quirky",
        snippet=(
            "def first_n_squares(n):\n"
            "    result = []\n"
            "    for i in range(n):\n"
            "        result.append(i * i)\n"
            "    return result\n"
        ),
        rationale="works but builds list with .append in a loop",
    ),
    Snippet(
        id="S4",
        category="quirky",
        snippet=(
            "def show_index_and_value(items):\n"
            "    for i in range(len(items)):\n"
            "        print(i, items[i])\n"
        ),
        rationale="C-style indexing, should be enumerate",
    ),

    # --- complex_correct (S5, S6): correct, idiomatic, complex ---
    Snippet(
        id="S5",
        category="complex_correct",
        snippet=(
            "def flatten_and_filter(matrix):\n"
            "    return [x for row in matrix for x in row if x > 0]\n"
        ),
        rationale="nested comprehension with predicate, harder to read",
    ),
    Snippet(
        id="S6",
        category="complex_correct",
        snippet=(
            "def merge_dicts_summing(dicts):\n"
            "    out = {}\n"
            "    for d in dicts:\n"
            "        for k, v in d.items():\n"
            "            out[k] = out.get(k, 0) + v\n"
            "    return out\n"
        ),
        rationale="correct, idiomatic, but cognitively dense",
    ),

    # --- buggy (S7, S8): incorrect, otherwise nice ---
    Snippet(
        id="S7",
        category="buggy",
        snippet=(
            "def total(numbers):\n"
            "    result = 0\n"
            "    for i in range(1, len(numbers)):\n"
            "        result += numbers[i]\n"
            "    return result\n"
        ),
        rationale="off-by-one: skips first element",
    ),
    Snippet(
        id="S8",
        category="buggy",
        snippet=(
            "def is_zero(x):\n"
            "    if x is 0:\n"
            "        return True\n"
            "    return False\n"
        ),
        rationale="`is 0` instead of `== 0`",
    ),

    # --- narrow (S9, S10): correct, scope unclear ---
    Snippet(
        id="S9",
        category="narrow",
        snippet=(
            "x = 42\n"
        ),
        rationale="trivially correct as a statement, no scope to judge",
    ),
    Snippet(
        id="S10",
        category="narrow",
        snippet=(
            "import os\n"
        ),
        rationale="syntactically fine, no scope",
    ),

    # --- bad_all (S11, S12): incorrect, complex, unidiomatic ---
    Snippet(
        id="S11",
        category="bad_all",
        snippet=(
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n - 1):\n"
            "        for j in range(n - i):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                tmp = arr[j]\n"
            "                arr[j] = arr[j + 1]\n"
            "                arr[j + 1] = tmp\n"
            "    return arr\n"
        ),
        rationale="off-by-one in inner range; manual swap instead of tuple",
    ),
    Snippet(
        id="S12",
        category="bad_all",
        snippet=(
            "def factorial(n):\n"
            "    return 1 if n == 0 else n * factorial(n - 2)\n"
        ),
        rationale="recurses with n-2 instead of n-1; ternary obscures the bug",
    ),
]


CATEGORY_PAIRS: dict[str, list[str]] = {}
for s in SNIPPETS:
    CATEGORY_PAIRS.setdefault(s.category, []).append(s.id)
