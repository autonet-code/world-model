#!/usr/bin/env python3
"""
Iris classification via the shared classifier_wiring module.

3 species, 4 features, 150 cases. Random baseline 33%.
Same engine machinery as digits and letters. Only the dataset differs.
"""

from __future__ import annotations

import random
from collections import Counter

from sklearn.datasets import load_iris

from world_model.analysis.classifier_wiring import (
    fit_classifier_profile,
    build_classifier_state,
    classify_case,
)


def main(test_fraction: float = 0.3, seed: int = 42) -> int:
    print()
    print("=" * 70)
    print("IRIS CLASSIFICATION VIA RESEED-AND-EQUILIBRATE")
    print("=" * 70)

    iris = load_iris()
    X, y = iris.data, iris.target
    n = len(y)
    classes = list(range(len(iris.target_names)))
    class_names = list(iris.target_names)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * (1.0 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"\n  iris: {n} cases, 4 features, 3 species")
    print(f"  train/test: {len(train_idx)}/{len(test_idx)} (seed={seed})")

    profile = fit_classifier_profile(X_train, y_train, classes)
    base_state = build_classifier_state(
        profile, class_prefix="species_", feature_prefix="feat"
    )

    n_tendencies = len(base_state.tendencies)
    n_edges = sum(len(adj) for adj in base_state.graph.weights.values()) // 2
    print(f"  state: {n_tendencies} tendencies, {n_edges} edges (contrast wiring)")

    print(f"\n  classifying {len(test_idx)} test cases...")
    correct = 0
    confusion: dict[tuple[int, int], int] = Counter()
    for i in range(len(test_idx)):
        prediction = classify_case(
            X_test[i], profile, base_state,
            class_prefix="species_", feature_prefix="feat",
        )
        true_class = int(y_test[i])
        if prediction == true_class:
            correct += 1
        confusion[(true_class, prediction)] += 1

    accuracy = correct / len(test_idx)
    print(f"\n  accuracy: {correct}/{len(test_idx)} = {accuracy:.1%}")
    print(f"  baseline (random): 33.3%")

    print(f"\n  confusion (row=true, col=predicted):")
    print(f"    {'':12s} " + " ".join(f"{n:>11s}" for n in class_names))
    for i, name in enumerate(class_names):
        row_cells = " ".join(f"{confusion[(i, j)]:>11d}" for j in range(len(classes)))
        print(f"    {name:12s} {row_cells}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
