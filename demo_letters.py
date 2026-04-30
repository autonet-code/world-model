#!/usr/bin/env python3
"""
Letter recognition via the shared classifier_wiring module.

26 classes, 16 features, 20000 cases. Random baseline ~3.8%.
The hardest of our classification demos to date.
"""

from __future__ import annotations

import random
from collections import Counter

from sklearn.datasets import fetch_openml

from world_model.analysis.classifier_wiring import (
    fit_classifier_profile,
    build_classifier_state,
    classify_case,
)


def main(test_fraction: float = 0.2, seed: int = 42, max_test_cases: int = 200) -> int:
    print()
    print("=" * 70)
    print("LETTER RECOGNITION VIA RESEED-AND-EQUILIBRATE")
    print("=" * 70)

    print("\n  loading letter dataset...")
    data = fetch_openml('letter', version=1, as_frame=False, parser='auto')
    X = data.data.astype(float)
    y = data.target
    n_total = len(y)
    classes = sorted(set(y))
    n_classes = len(classes)

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    split = int(n_total * (1.0 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]
    if len(test_idx) > max_test_cases:
        test_idx = test_idx[:max_test_cases]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"  letters: {n_total} cases, 16 features, 26 classes")
    print(f"  train/test: {len(train_idx)}/{len(test_idx)} (cap {max_test_cases}, seed={seed})")

    profile = fit_classifier_profile(X_train, y_train, classes)
    base_state = build_classifier_state(
        profile, class_prefix="letter_", feature_prefix="f"
    )

    n_tendencies = len(base_state.tendencies)
    n_edges = sum(len(adj) for adj in base_state.graph.weights.values()) // 2
    print(f"  state: {n_tendencies} tendencies, {n_edges} edges (contrast wiring)")

    print(f"\n  classifying {len(test_idx)} test cases...")
    correct = 0
    confusion: dict[tuple[str, str], int] = Counter()
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()
    for i in range(len(test_idx)):
        prediction = classify_case(
            X_test[i], profile, base_state,
            class_prefix="letter_", feature_prefix="f",
        )
        true_class = y_test[i]
        per_class_total[true_class] += 1
        if prediction == true_class:
            correct += 1
            per_class_correct[true_class] += 1
        confusion[(true_class, prediction)] += 1
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(test_idx)}  running: {correct/(i+1):.1%}")

    accuracy = correct / len(test_idx)
    print(f"\n  accuracy: {correct}/{len(test_idx)} = {accuracy:.1%}")
    print(f"  baseline (random): {1/n_classes:.1%}")

    print(f"\n  per-class accuracy (>=3 cases):")
    for c in classes:
        if per_class_total[c] >= 3:
            n = per_class_total[c]
            r = per_class_correct[c]
            bar = "#" * int(round(r / n * 20))
            print(f"    {c}: {r:>2d}/{n:>2d} = {r/n:>3.0%}  {bar}")

    miscls = [(t, p, c) for (t, p), c in confusion.items() if t != p]
    miscls.sort(key=lambda x: -x[2])
    if miscls:
        print(f"\n  top confusions (true -> predicted):")
        for t, p, c in miscls[:10]:
            print(f"    {t} -> {p}: {c}x")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
