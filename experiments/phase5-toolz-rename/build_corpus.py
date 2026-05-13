#!/usr/bin/env python3
"""Build the Phase 5 renamed-toolz corpus.

Pulls candidate functions from toolz, filters for those with usable
doctests, applies the rename map, emits a JSON corpus file with:

    {
      "rename_map": {orig_name: renamed_name, ...},
      "train": [{name, docstring, impl, doctests, ...}, ...],
      "test":  [{name, sparse_docstring, doctests, ...}, ...]
    }

Train entries carry the full renamed docstring + impl. Test entries
carry a SPARSE one-line docstring + the same doctest examples (so we
can score implementations against them).
"""

from __future__ import annotations

import ast
import doctest
import inspect
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import toolz.functoolz as F  # type: ignore
import toolz.itertoolz as I  # type: ignore
import toolz.dicttoolz as D  # type: ignore


HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "corpus.json"
RENAME_MAP_PATH = HERE / "rename_map.json"


# --------------------------------------------------------------------------
# Rename map. Hand-picked to defeat surface pattern-match without obscuring
# concepts.
# --------------------------------------------------------------------------

RENAME_MAP: Dict[str, str] = {
    # functoolz
    "identity":          "passthrough",
    "complement":        "negate_pred",
    "compose":           "fold_funcs",
    "compose_left":      "fold_funcs_left",
    "do":                "tap_side_effect",
    "apply":             "invoke_with",
    "juxt":              "multicast",
    "pipe":              "thread_value",
    "thread_first":      "weave_first",
    "thread_last":       "weave_last",
    "is_arity":          "has_arg_count",
    "is_partial_args":   "could_partially_apply",
    "is_valid_args":     "could_invoke",
    "has_keywords":      "accepts_kwargs",
    "has_varargs":       "accepts_starargs",
    "num_required_args": "required_arg_count",
    "instanceproperty":  "instance_only_property",
    # itertoolz
    "take":          "first_n",
    "drop":          "skip_n",
    "tail":          "last_n",
    "take_nth":      "every_nth",
    "partition":     "chunks_of",
    "partition_all": "chunks_of_at_most",
    "sliding_window":"windows_of",
    "interleave":    "weave_seqs",
    "concat":        "flatten_one_level",
    "concatv":       "flatten_args",
    "cons":          "prepend_item",
    "mapcat":        "flat_map",
    "frequencies":   "tally",
    "unique":        "dedupe",
    "isiterable":    "is_iterable",
    "isdistinct":    "all_distinct",
    "first":         "head",
    "second":        "head_after",
    "nth":           "at_index",
    "last":          "final",
    "remove":        "drop_where",
    "accumulate":    "running_reduce",
    "diff":          "first_diff",
    "topk":          "top_n",
    "peek":          "peek_one",
    "peekn":         "peek_many",
    "pluck":         "pluck_at",
    "interpose":     "weave_separator",
    "groupby":       "bucket_by",
    "merge_sorted":  "merge_ordered",
    "iterate":       "iterate_fn",
    "random_sample": "sample_at_rate",
    # dicttoolz
    "assoc":         "with_kv",
    "assoc_in":      "with_kv_at_path",
    "dissoc":        "without_keys",
    "get_in":        "lookup_path",
    "itemfilter":    "filter_items",
    "itemmap":       "map_items",
    "keyfilter":     "filter_keys",
    "keymap":        "map_keys",
    "merge":         "combine_dicts",
    "merge_with":    "combine_dicts_with",
    "valfilter":     "filter_values",
    "valmap":        "map_values",
}


def make_renamer(rename_map: Dict[str, str]):
    """Return a function that renames orig identifiers in a string,
    using word-boundary regex so we don't munge substrings."""
    if not rename_map:
        return lambda s: s
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in rename_map.keys()) + r")\b"
    )

    def rename(s: str) -> str:
        return pattern.sub(lambda m: rename_map[m.group(0)], s)

    return rename


# --------------------------------------------------------------------------
# Doctest parsing + minimal correctness harness
# --------------------------------------------------------------------------


def extract_doctests(docstring: str) -> List[Dict[str, str]]:
    """Pull (call, expected) pairs out of a docstring's >>> lines.
    Returns a list of {'call': '...', 'expected': '...'} dicts."""
    finder = doctest.DocTestParser()
    try:
        examples = finder.get_examples(docstring or "")
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for ex in examples:
        out.append({
            "call": ex.source.rstrip("\n"),
            "expected": ex.want.rstrip("\n"),
        })
    return out


# --------------------------------------------------------------------------
# Pull candidate functions from a module, filter by criteria, return
# usable training/testing entries.
# --------------------------------------------------------------------------


def candidate_functions(module, rename_map: Dict[str, str]):
    """Yield (orig_name, function) for each callable in `module` whose
    orig_name is in rename_map and which has source available."""
    for orig_name in rename_map:
        obj = getattr(module, orig_name, None)
        if obj is None:
            continue
        try:
            src = inspect.getsource(obj)
        except (TypeError, OSError):
            continue
        if not callable(obj):
            continue
        yield orig_name, obj, src


def parse_signature_safely(src: str, name: str) -> str:
    """Extract `def name(...):` line from source, return just the signature
    body (the part inside the parens). Returns empty string on failure."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node.args)
    return ""


# --------------------------------------------------------------------------
# Build train/test entries
# --------------------------------------------------------------------------


def build_train_entry(orig_name: str, src: str, rename) -> Dict[str, Any]:
    """Build a training-corpus entry from a function's source.
    Carries full renamed docstring + impl, plus parsed doctests."""
    # Parse to get docstring and body separately.
    tree = ast.parse(src)
    func_node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == orig_name),
        None,
    )
    if func_node is None:
        return {}

    doc_orig = ast.get_docstring(func_node) or ""
    if not doc_orig.strip():
        return {}

    doctests = extract_doctests(rename(doc_orig))
    if not doctests:
        # No doctests = no correctness signal at inference time. Skip.
        return {}

    renamed_name = rename(orig_name)
    renamed_src = rename(src)
    renamed_doc = rename(doc_orig)
    signature = rename(parse_signature_safely(src, orig_name))

    return {
        "orig_name": orig_name,
        "name": renamed_name,
        "signature": signature,
        "docstring": renamed_doc,
        "impl_full_source": renamed_src,
        "doctests": doctests,
    }


def build_test_entry(orig_name: str, src: str, rename) -> Dict[str, Any]:
    """Build a held-out inference entry. Same shape as train but with
    a SPARSE docstring: one-line summary + 1-2 doctest examples instead
    of the full original docs."""
    entry = build_train_entry(orig_name, src, rename)
    if not entry:
        return {}

    # Sparse docstring: take first sentence of original (already renamed),
    # plus the first 2 doctest examples.
    first_sentence = (entry["docstring"].split(".")[0].strip() + ".").strip(". ").strip()
    if not first_sentence:
        first_sentence = f"{entry['name']}: short utility function."

    shown_doctests = entry["doctests"][:2]
    example_block = "\n".join(
        f">>> {ex['call']}\n{ex['expected']}" for ex in shown_doctests
    )
    sparse_doc = f"{first_sentence}\n\n{example_block}"

    entry["sparse_docstring"] = sparse_doc
    entry["shown_doctests"] = shown_doctests
    return entry


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    rename = make_renamer(RENAME_MAP)

    entries: List[Dict[str, Any]] = []
    for module in (F, I, D):
        for orig_name, fn, src in candidate_functions(module, RENAME_MAP):
            entry = build_train_entry(orig_name, src, rename)
            if entry:
                entries.append(entry)

    print(f"  candidates with doctests: {len(entries)}")
    if len(entries) < 20:
        print("  WARN: corpus too small; consider broadening RENAME_MAP")

    # Deterministic shuffle, 80/20 split.
    random.seed(42)
    random.shuffle(entries)
    n_test = max(8, min(12, len(entries) // 5))
    n_train = len(entries) - n_test
    train_entries = entries[:n_train]
    test_entries_raw = entries[n_train:]

    # Test entries get sparse-docstring rewrite.
    test_entries: List[Dict[str, Any]] = []
    for entry in test_entries_raw:
        first_sentence = (entry["docstring"].split(".")[0].strip() + ".").strip(". ").strip()
        shown = entry["doctests"][:2]
        example_block = "\n".join(
            f">>> {ex['call']}\n{ex['expected']}" for ex in shown
        )
        sparse_doc = f"{first_sentence}\n\n{example_block}"
        test_entries.append({
            **{k: v for k, v in entry.items() if k != "impl_full_source"},
            "sparse_docstring": sparse_doc,
            "shown_doctests": shown,
        })

    print(f"  split: {len(train_entries)} train, {len(test_entries)} test")

    OUT_PATH.write_text(
        json.dumps({
            "rename_map": RENAME_MAP,
            "train": train_entries,
            "test": test_entries,
        }, indent=2),
        encoding="utf-8",
    )
    RENAME_MAP_PATH.write_text(json.dumps(RENAME_MAP, indent=2), encoding="utf-8")
    print(f"  wrote {OUT_PATH}")
    print(f"  wrote {RENAME_MAP_PATH}")

    # Quick sanity: print a renamed train entry and a test entry.
    print()
    print("--- example train entry ---")
    e = train_entries[0]
    print(f"name: {e['name']}  (orig: {e['orig_name']})")
    print(f"signature: ({e['signature']})")
    print(f"doctests: {len(e['doctests'])}")
    print("docstring excerpt:")
    print("  " + "\n  ".join(e["docstring"].splitlines()[:5]))

    print()
    print("--- example test entry ---")
    e = test_entries[0]
    print(f"name: {e['name']}  (orig: {e['orig_name']})")
    print("sparse_docstring:")
    print("  " + "\n  ".join(e["sparse_docstring"].splitlines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
