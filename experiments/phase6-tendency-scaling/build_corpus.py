#!/usr/bin/env python3
"""Phase 6 corpus: toolz + more_itertools, renamed to defeat memorization.

Expands the Phase 5 rename map to cover more_itertools' doctested functions
in addition to toolz. Target: ~130 train, ~26 test.
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
import more_itertools as M  # type: ignore


HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "corpus.json"


# Phase 5 toolz renames (kept) + many more_itertools renames.
RENAME_MAP: Dict[str, str] = {
    # toolz.functoolz
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
    # toolz.itertoolz
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
    # toolz.dicttoolz
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
    # more_itertools (selected, distinctive renames)
    "chunked":             "fixed_chunks",
    "chunked_even":        "even_chunks",
    "ichunked":            "lazy_chunks",
    "constrained_batches": "size_bounded_batches",
    "flatten":             "single_level_flatten",
    "ncycles":             "repeat_seq",
    "repeatfunc":          "repeat_call",
    "split_at":            "cut_at_pred",
    "split_before":        "cut_before_pred",
    "split_after":         "cut_after_pred",
    "split_when":          "cut_on_change",
    "split_into":          "cut_by_sizes",
    "always_iterable":     "ensure_iterable",
    "always_reversible":   "ensure_reversible",
    "consume":             "discard_items",
    "tabulate":            "build_via_fn",
    "tail":                "yield_last",
    "first_true":          "first_truthy_or",
    "all_equal":           "all_same",
    "all_unique":          "every_unique",
    "is_sorted":           "in_order",
    "minmax":              "min_and_max",
    "before_and_after":    "split_at_first_failing",
    "raise_":              "throw_exc",
    "lstrip":              "drop_leading",
    "rstrip":              "drop_trailing",
    "strip":               "drop_both_ends",
    "padded":              "right_pad_with",
    "longest_common_prefix": "shared_head",
    "dotproduct":          "inner_product",
    "convolve":            "linear_convolve",
    "polynomial_from_roots": "poly_from_roots",
    "factor":              "prime_factors",
    "prepend":             "place_first",
    "value_chain":         "deep_value_chain",
    "unique_in_window":    "novel_in_window",
    "set_partitions":      "all_set_partitions",
    "powerset":            "all_subsets",
    "powerset_of_sets":    "all_subset_sets",
    "circular_shifts":     "all_rotations",
    "duplicates_everseen": "repeated_items",
    "unique_everseen":     "once_each",
    "unique_justseen":     "runs_collapsed",
    "filter_except":       "keep_if_safe",
    "map_except":          "transform_if_safe",
    "rstrip_seqs":         "drop_trailing_seqs",
    "callback_iter":       "iter_to_callbacks",
    "iter_index":          "indices_of",
    "iter_suppress":       "stop_on_exc",
    "filter_map":          "transform_keeping_truthy",
    "windowed":            "fixed_windows",
    "stagger":             "offset_views",
    "pairwise":            "adjacent_pairs",
    "triplewise":          "adjacent_triples",
    "windowed_complete":   "every_split_at_each_index",
    "intersperse":         "between_each",
    "sieve":               "primes_up_to",
    "side_effect":         "with_side_effect",
    "iterate":             "iter_n_times",   # collides w/ toolz.iterate → handled below
    "consecutive_groups":  "consecutive_runs",
    "exactly_n":           "exactly_n_truthy",
    "is_unique":           "is_one_of_a_kind",
    "ilen":                "iter_length",
    "with_iter":           "context_iter",
    "iter_except":         "iter_until_exc",
    "locate":              "indices_where",
    "rlocate":             "indices_where_reverse",
    "replace":             "swap_subseq",
    "filter_value":        "keep_value",
    "lstrip_seqs":         "drop_leading_seqs",
    "duplicates_justseen": "adjacent_repeats",
}


# Some toolz vs more_itertools name collisions ("tail", "iterate", "flatten",
# "first_true", "consume", "powerset") — we route them via module-specific
# pre-rename in the iteration loop below. For collisions, the more_itertools
# version wins (more functions there).


def make_renamer(rename_map: Dict[str, str]):
    if not rename_map:
        return lambda s: s
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in rename_map.keys()) + r")\b"
    )
    return lambda s: pattern.sub(lambda m: rename_map[m.group(0)], s)


def extract_doctests(docstring: str) -> List[Dict[str, str]]:
    finder = doctest.DocTestParser()
    try:
        examples = finder.get_examples(docstring or "")
    except Exception:
        return []
    return [
        {"call": ex.source.rstrip("\n"), "expected": ex.want.rstrip("\n")}
        for ex in examples
    ]


def parse_signature(src: str, name: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node.args)
    return ""


def build_entry(orig_name: str, src: str, rename) -> Dict[str, Any]:
    """Build an entry from a function's source. Returns {} if unusable."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    func_node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == orig_name),
        None,
    )
    if func_node is None:
        return {}
    doc_orig = ast.get_docstring(func_node) or ""
    if not doc_orig.strip():
        return {}

    doctests = extract_doctests(rename(doc_orig))
    if not doctests:
        return {}

    renamed_name = rename(orig_name)
    return {
        "orig_name": orig_name,
        "name": renamed_name,
        "signature": rename(parse_signature(src, orig_name)),
        "docstring": rename(doc_orig),
        "impl_full_source": rename(src),
        "doctests": doctests,
    }


def collect(module, rename, names_to_pull: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for n in names_to_pull:
        obj = getattr(module, n, None)
        if obj is None:
            continue
        try:
            src = inspect.getsource(obj)
        except (TypeError, OSError):
            continue
        if not callable(obj):
            continue
        e = build_entry(n, src, rename)
        if e:
            entries.append(e)
    return entries


def main() -> int:
    # Resolve collisions: toolz names take precedence for unique items;
    # for shared names (tail, iterate), use a per-module renamer so each
    # version gets distinct identity in the corpus.
    toolz_unique = {k: v for k, v in RENAME_MAP.items()
                    if k not in ("tail", "iterate")}
    rename_toolz = make_renamer(toolz_unique)

    # more_itertools sees the full map plus collision-specific renames.
    mi_overrides = {
        "tail":    "yield_last",
        "iterate": "iter_n_times",
    }
    mi_map = {**RENAME_MAP, **mi_overrides}
    rename_mi = make_renamer(mi_map)

    toolz_funcs = [k for k in toolz_unique
                   if any(getattr(mod, k, None) is not None
                          for mod in (F, I, D))]
    mi_funcs = [k for k in mi_map
                if getattr(M, k, None) is not None]

    entries: List[Dict[str, Any]] = []
    for mod in (F, I, D):
        entries.extend(collect(mod, rename_toolz, toolz_funcs))
    entries.extend(collect(M, rename_mi, mi_funcs))

    # Deduplicate on renamed name (collisions should already be resolved).
    seen: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        if e["name"] not in seen:
            seen[e["name"]] = e
    entries = list(seen.values())
    print(f"  total entries with doctests: {len(entries)}")

    if len(entries) < 80:
        print("  WARN: corpus smaller than planned 80+ minimum.")

    random.seed(42)
    random.shuffle(entries)
    n_test = max(20, min(30, len(entries) // 5))
    n_train = len(entries) - n_test
    train_entries = entries[:n_train]
    test_entries_raw = entries[n_train:]

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
    print(f"  wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
