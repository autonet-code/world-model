#!/usr/bin/env python3
"""
Letter recognition via cascade equilibrium across a class hierarchy.

The flat classifier put all 26 letters on equal footing in one graph
and let graph-walk averaging discriminate among them. That worked at
46%.

This version organizes the letters as a hierarchy (clusters of similar
letters at internal nodes, individual letters at leaves) and classifies
by cascading: at each level, equilibrate among candidates, descend into
the winner's branch, recurse. Same engine, same operation; what changes
is that each equilibration is a smaller / clearer problem.

Two cascade strategies are tested:
  - HARD: pick the winner at each level, descend, no recovery from
    early misclassification.
  - SOFT: every path through the hierarchy accumulates probability.
    Final leaf prediction maximizes accumulated probability.

If the fractal-first claim is real, soft-cascade should outperform the
flat 46%. Hard-cascade may underperform if early-level errors are
unrecoverable.
"""

from __future__ import annotations

import random
from collections import Counter

from sklearn.datasets import fetch_openml

from world_model.analysis.classifier_wiring import fit_classifier_profile
from world_model.analysis.hierarchical_classifier import (
    build_hierarchy,
    classify_cascade_hard,
    classify_cascade_soft,
    classify_unified_with_hierarchy,
)


def main(test_fraction: float = 0.2, seed: int = 42, max_test_cases: int = 200) -> int:
    print()
    print("=" * 70)
    print("LETTER RECOGNITION VIA HIERARCHICAL CASCADE")
    print("=" * 70)

    print("\n  loading letter dataset...")
    data = fetch_openml('letter', version=1, as_frame=False, parser='auto')
    X = data.data.astype(float)
    y = data.target
    classes = sorted(set(y))
    n_classes = len(classes)

    rng = random.Random(seed)
    indices = list(range(len(y)))
    rng.shuffle(indices)
    split = int(len(y) * (1.0 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]
    if len(test_idx) > max_test_cases:
        test_idx = test_idx[:max_test_cases]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"  classes: {n_classes}, features: 16, train/test: {len(train_idx)}/{len(test_idx)}")

    profile = fit_classifier_profile(X_train, y_train, classes)
    hierarchy = build_hierarchy(profile)
    n_internal = sum(1 for n in hierarchy.nodes.values() if not n.is_leaf)
    print(f"  hierarchy: {n_internal} internal nodes + {n_classes} leaves "
          f"(max depth {max(n.depth for n in hierarchy.nodes.values())})")

    # ------------- Hard cascade -------------
    print(f"\n  HARD cascade: descend by winner at each level")
    correct_hard = 0
    confusion_hard: Counter = Counter()
    for i in range(len(test_idx)):
        prediction = classify_cascade_hard(X_test[i], profile, hierarchy)
        true_class = y_test[i]
        if prediction == true_class:
            correct_hard += 1
        confusion_hard[(true_class, prediction)] += 1
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(test_idx)}  running: {correct_hard/(i+1):.1%}")

    acc_hard = correct_hard / len(test_idx)
    print(f"\n  HARD accuracy: {correct_hard}/{len(test_idx)} = {acc_hard:.1%}")

    # ------------- Soft cascade -------------
    print(f"\n  SOFT cascade: accumulate probability along all paths")
    correct_soft = 0
    confusion_soft: Counter = Counter()
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()
    for i in range(len(test_idx)):
        prediction = classify_cascade_soft(X_test[i], profile, hierarchy)
        true_class = y_test[i]
        per_class_total[true_class] += 1
        if prediction == true_class:
            correct_soft += 1
            per_class_correct[true_class] += 1
        confusion_soft[(true_class, prediction)] += 1
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(test_idx)}  running: {correct_soft/(i+1):.1%}")

    acc_soft = correct_soft / len(test_idx)
    print(f"\n  SOFT accuracy: {correct_soft}/{len(test_idx)} = {acc_soft:.1%}")

    # ------------- Unified-with-hierarchy -------------
    print(f"\n  UNIFIED: leaves AND internal nodes in one equilibrium")
    correct_unified = 0
    confusion_unified: Counter = Counter()
    per_class_total_u: Counter = Counter()
    per_class_correct_u: Counter = Counter()
    for i in range(len(test_idx)):
        prediction = classify_unified_with_hierarchy(X_test[i], profile, hierarchy)
        true_class = y_test[i]
        per_class_total_u[true_class] += 1
        if prediction == true_class:
            correct_unified += 1
            per_class_correct_u[true_class] += 1
        confusion_unified[(true_class, prediction)] += 1
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(test_idx)}  running: {correct_unified/(i+1):.1%}")
    acc_unified = correct_unified / len(test_idx)
    print(f"\n  UNIFIED accuracy: {correct_unified}/{len(test_idx)} = {acc_unified:.1%}")

    # ------------- Summary -------------
    print(f"\n  baseline (random): {1/n_classes:.1%}")
    print(f"  flat-classifier reference: 46%")
    print(f"  HARD cascade:  {acc_hard:.1%}  ({(acc_hard-0.46)*100:+.1f} pts vs flat)")
    print(f"  SOFT cascade:  {acc_soft:.1%}  ({(acc_soft-0.46)*100:+.1f} pts vs flat)")
    print(f"  UNIFIED:       {acc_unified:.1%}  ({(acc_unified-0.46)*100:+.1f} pts vs flat)")

    print(f"\n  per-class soft accuracy (>=3 cases):")
    for c in classes:
        if per_class_total[c] >= 3:
            n = per_class_total[c]
            r = per_class_correct[c]
            bar = "#" * int(round(r / n * 20))
            print(f"    {c}: {r:>2d}/{n:>2d} = {r/n:>3.0%}  {bar}")

    miscls = [(t, p, c) for (t, p), c in confusion_soft.items() if t != p]
    miscls.sort(key=lambda x: -x[2])
    if miscls:
        print(f"\n  top SOFT confusions:")
        for t, p, c in miscls[:10]:
            print(f"    {t} -> {p}: {c}x")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
