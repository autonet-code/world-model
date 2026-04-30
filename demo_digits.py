#!/usr/bin/env python3
"""
Handwritten digit classification via reseed-and-equilibrate.

Steps up from Iris: 10 classes (chance baseline 10%), 64 features
(8x8 pixel intensities), 1797 samples. The graph has 138 tendencies
(10 digit-class + 128 pixel-evidence) and 1280 edges, which is real
topology rather than the toy scale of Iris.

Same wiring strategy as iris -- digit-class tendencies are connected
to pixel-evidence tendencies (one "high" and one "low" per pixel)
with edge weights derived from per-class pixel-intensity centroids.
For each test case, substitute the pixel-evidence tendencies with
allocations reflecting the case's actual pixels and let the engine
propagate that evidence through the graph to reshape the digit-class
tendencies.

Honest-mode classification: the engine does the inference. No external
distance computation. propagate_via_graph=True.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from sklearn.datasets import load_digits

from world_model import (
    DefaultTendencyFactory,
    Lineage,
    PresentState,
    StakeWeightGraph,
    Substitution,
    Tendency,
    TendencySpec,
    reseed_and_equilibrate,
)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    """Per-class centroids and global feature ranges."""
    class_centroids: dict[int, list[float]]   # digit -> [mean per pixel]
    feature_min: list[float]
    feature_max: list[float]
    n_features: int


def fit_profile(X_train, y_train, n_classes: int) -> Profile:
    n_features = X_train.shape[1]
    centroids: dict[int, list[float]] = {}
    for c in range(n_classes):
        rows = [X_train[i] for i in range(len(y_train)) if y_train[i] == c]
        n = len(rows)
        if n == 0:
            centroids[c] = [0.0] * n_features
            continue
        centroids[c] = [
            sum(r[f] for r in rows) / n
            for f in range(n_features)
        ]
    feature_min = [float(X_train[:, f].min()) for f in range(n_features)]
    feature_max = [float(X_train[:, f].max()) for f in range(n_features)]
    return Profile(
        class_centroids=centroids,
        feature_min=feature_min,
        feature_max=feature_max,
        n_features=n_features,
    )


def normalize_feature(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# ---------------------------------------------------------------------------
# Build starting state
# ---------------------------------------------------------------------------

def build_starting_state(profile: Profile, n_classes: int) -> PresentState:
    factory = DefaultTendencyFactory()
    n_features = profile.n_features
    n_total = n_classes + 2 * n_features

    # Half the mass to class tendencies, half to pixel-evidence
    class_alloc = 0.5 / n_classes
    pixel_alloc = 0.5 / (2 * n_features)

    specs = []
    for c in range(n_classes):
        specs.append(TendencySpec(
            id=f"digit_{c}",
            initial_allocation=class_alloc,
            description=f"Belief that the case is digit {c}",
        ))
    for f in range(n_features):
        specs.append(TendencySpec(
            id=f"px{f:02d}_low",
            initial_allocation=pixel_alloc,
        ))
        specs.append(TendencySpec(
            id=f"px{f:02d}_high",
            initial_allocation=pixel_alloc,
        ))

    tendencies = factory.build_set(specs)

    # Build edges: each digit class connected to each pixel_high and
    # pixel_low. Weight depends on how high/low that pixel typically is
    # in cases of that class.
    graph = StakeWeightGraph()
    for c in range(n_classes):
        centroid = profile.class_centroids[c]
        for f in range(n_features):
            high_aff = normalize_feature(
                centroid[f], profile.feature_min[f], profile.feature_max[f]
            )
            low_aff = 1.0 - high_aff
            # Use a sharper mapping than iris -- digits centroids vary
            # more dramatically across pixels (some pixels are nearly
            # always 0 for a class, others nearly always max). A linear
            # squash to 0.05..0.95 keeps weights nontrivial.
            graph.add_edge(f"digit_{c}", f"px{f:02d}_high", 0.05 + 0.9 * high_aff)
            graph.add_edge(f"digit_{c}", f"px{f:02d}_low",  0.05 + 0.9 * low_aff)

    lineages = {tid: Lineage() for tid in tendencies.ids()}
    return PresentState(tendencies=tendencies, lineages=lineages, graph=graph)


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

def classify(case_features, profile: Profile, base_state: PresentState,
             n_classes: int) -> int:
    """Substitute pixel-evidence tendencies; let the engine propagate
    through the graph to reshape digit-class tendencies."""
    n_features = profile.n_features
    feature_budget = 0.5 / n_features

    substitutions = []
    for f in range(n_features):
        high_strength = normalize_feature(
            case_features[f], profile.feature_min[f], profile.feature_max[f]
        )
        low_strength = 1.0 - high_strength
        substitutions.append(Substitution(
            id=f"px{f:02d}_high",
            new_tendency=Tendency(
                id=f"px{f:02d}_high",
                allocation=feature_budget * high_strength,
            ),
        ))
        substitutions.append(Substitution(
            id=f"px{f:02d}_low",
            new_tendency=Tendency(
                id=f"px{f:02d}_low",
                allocation=feature_budget * low_strength,
            ),
        ))

    result = reseed_and_equilibrate(
        base_state,
        substitutions=substitutions,
        propagate_via_graph=True,
        learning_rate=0.3,
        tolerance=1e-5,
        max_iterations=300,
    )

    class_allocs = {
        c: result.state.tendencies.get(f"digit_{c}").allocation
        for c in range(n_classes)
    }
    return max(class_allocs, key=class_allocs.get)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_fraction: float = 0.3, seed: int = 42, max_test_cases: int = 200) -> int:
    print()
    print("=" * 70)
    print("HANDWRITTEN DIGITS CLASSIFICATION VIA RESEED-AND-EQUILIBRATE")
    print("=" * 70)

    digits = load_digits()
    X, y = digits.data, digits.target
    n_total = len(y)
    n_classes = len(digits.target_names)

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    split = int(n_total * (1.0 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]

    # Cap test cases for runtime control. Each classify call is non-trivial
    # because the calibration loop runs over 138 tendencies for up to 300 iter.
    if len(test_idx) > max_test_cases:
        test_idx = test_idx[:max_test_cases]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"\n  loaded digits: {n_total} cases, 64 pixels, 10 classes")
    print(f"  train/test split: {len(train_idx)}/{len(test_idx)} (cap {max_test_cases}, seed={seed})")

    profile = fit_profile(X_train, y_train, n_classes)
    print(f"\n  per-class centroids fitted (10 x 64-dim).")
    print(f"  feature ranges: pixel intensities span [0, 16].")

    base_state = build_starting_state(profile, n_classes)
    n_tendencies = len(base_state.tendencies)
    n_edges = sum(len(adj) for adj in base_state.graph.weights.values()) // 2
    print(f"\n  starting state: {n_tendencies} tendencies "
          f"({n_classes} digit + {2 * profile.n_features} pixel-evidence), "
          f"{n_edges} edges")

    print(f"\n  classifying {len(test_idx)} test cases...")
    correct = 0
    confusion: dict[tuple[int, int], int] = Counter()
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()

    for i in range(len(test_idx)):
        features = X_test[i]
        true_class = int(y_test[i])
        prediction = classify(features, profile, base_state, n_classes)
        per_class_total[true_class] += 1
        if prediction == true_class:
            correct += 1
            per_class_correct[true_class] += 1
        confusion[(true_class, prediction)] += 1

        if (i + 1) % 20 == 0:
            running = correct / (i + 1)
            print(f"    progress: {i+1}/{len(test_idx)}  running accuracy: {running:.1%}")

    accuracy = correct / len(test_idx)
    print(f"\n  accuracy: {correct}/{len(test_idx)} = {accuracy:.1%}")
    print(f"  baseline (random): {1 / n_classes:.1%}")

    # Per-class
    print(f"\n  per-class accuracy:")
    for c in range(n_classes):
        total = per_class_total[c]
        right = per_class_correct[c]
        if total == 0:
            print(f"    digit {c}: (no test cases)")
        else:
            print(f"    digit {c}: {right}/{total} = {right/total:.0%}")

    # Confusion matrix
    print(f"\n  confusion matrix (row = true, col = predicted):")
    print("        " + " ".join(f"{c:>4d}" for c in range(n_classes)))
    for true_c in range(n_classes):
        row = "  " + f"{true_c:>4d}: "
        for pred_c in range(n_classes):
            count = confusion[(true_c, pred_c)]
            cell = f"{count:>4d}" if count > 0 else "   ."
            row += " " + cell
        print(row)

    print()
    print("=" * 70)
    if accuracy >= 0.70:
        print(f"RESULT: engine recovered digit classifications at {accuracy:.0%} -- ")
        print(f"        well above the {1/n_classes:.0%} random baseline. Graph-mediated")
        print(f"        inference scales from 3 classes (iris) to 10 classes (digits)")
        print(f"        without algorithmic change.")
    elif accuracy >= 0.40:
        print(f"RESULT: engine accuracy {accuracy:.0%} substantially above {1/n_classes:.0%} random.")
        print(f"        Graph propagation works on this scale, though the wiring")
        print(f"        leaves room for refinement. Confusable digit pairs in the")
        print(f"        confusion matrix above show where the topology smooths over")
        print(f"        genuinely-similar cases.")
    elif accuracy >= 1 / n_classes + 0.10:
        print(f"RESULT: engine accuracy {accuracy:.0%} is above random ({1/n_classes:.0%}) but ")
        print(f"        only modestly. The graph wiring strategy that worked at iris")
        print(f"        scale needs refinement for 10-class problems.")
    else:
        print(f"RESULT: engine accuracy {accuracy:.0%} is near or below random.")
        print(f"        Either the graph wiring or the calibration parameters")
        print(f"        are wrong for this scale -- diagnose before claiming the")
        print(f"        engine generalizes across class counts.")
    print("=" * 70)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
