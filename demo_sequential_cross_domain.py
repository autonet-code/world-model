#!/usr/bin/env python3
"""
Sequential cross-domain test: one engine, multiple configurations via reseed.

The architecture's actual cross-domain claim is universality through
reseeding: the same engine instance handles any domain by being reseeded
into the appropriate configuration. We test whether this works without
residue from previous configurations.

Setup
-----

1. Build an engine instance.
2. Reseed it into IRIS configuration. Run classification. Record accuracy.
3. Reseed the SAME engine into DIGITS configuration (replacing all iris
   tendencies with digit tendencies via substitutions). Run classification.
   Record accuracy.
4. Compare to a fresh-engine baseline for each domain (no prior history).

If reseeding produces clean transitions, sequential accuracy matches
fresh-engine accuracy. If the engine carries residue, we'll see degradation
on the second domain.

This is the architecture's universality claim, falsifiable.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from sklearn.datasets import load_iris, load_digits

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
from world_model.analysis.classifier_wiring import (
    fit_classifier_profile,
    build_classifier_state,
    classify_case,
    contrast_to_edge_weight,
    case_feature_strength,
)


def banner(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def build_iris_state():
    """Iris configuration: 3 species + 8 feature-evidence."""
    iris = load_iris()
    X, y = iris.data, iris.target
    classes = list(range(len(iris.target_names)))

    rng = random.Random(42)
    indices = list(range(len(y)))
    rng.shuffle(indices)
    split = int(len(y) * 0.7)
    X_tr, y_tr = X[indices[:split]], y[indices[:split]]
    X_te, y_te = X[indices[split:]], y[indices[split:]]

    profile = fit_classifier_profile(X_tr, y_tr, classes)
    state = build_classifier_state(profile, class_prefix="species_", feature_prefix="feat")
    return profile, state, X_te, y_te


def build_digits_state():
    """Digits configuration: 10 classes + 128 feature-evidence."""
    digits = load_digits()
    X, y = digits.data, digits.target
    classes = list(range(10))

    rng = random.Random(42)
    indices = list(range(len(y)))
    rng.shuffle(indices)
    split = int(len(y) * 0.7)
    X_tr, y_tr = X[indices[:split]], y[indices[:split]]
    X_te, y_te = X[indices[split:split + 100]], y[indices[split:split + 100]]  # cap to 100 for runtime

    profile = fit_classifier_profile(X_tr, y_tr, classes)
    state = build_classifier_state(profile, class_prefix="digit_", feature_prefix="px")
    return profile, state, X_te, y_te


def reseed_into_target_config(source_state: PresentState, target_state: PresentState):
    """Reseed source_state into target_state's configuration via substitutions.

    Strategy: for every tendency in source_state that is NOT in target_state,
    remove it (substitute with None). For every tendency in target_state that
    is NOT in source_state, add it. For every tendency in both, replace with
    the target's version.

    This is the universal reseed operation -- a substitution per affected
    tendency, total replacement, equilibration via the same mechanism we
    use for any domain shift.
    """
    source_ids = set(source_state.tendencies.ids())
    target_ids = set(target_state.tendencies.ids())

    substitutions = []

    # Remove tendencies not in target
    for tid in source_ids - target_ids:
        substitutions.append(Substitution(id=tid, new_tendency=None))

    # Add or replace tendencies from target
    for tid in target_ids:
        target_t = target_state.tendencies.get(tid)
        # Get edges from target's graph
        target_edges = dict(target_state.graph.weights.get(tid, {}))
        substitutions.append(Substitution(
            id=tid,
            new_tendency=Tendency(
                id=tid,
                allocation=target_t.allocation,
                description=target_t.description,
            ),
            edges=target_edges if target_edges else None,
        ))

    result = reseed_and_equilibrate(
        source_state,
        substitutions=substitutions,
        propagate_via_graph=False,    # just settle the new structure
        learning_rate=0.5,
        tolerance=1e-4,
        max_iterations=200,
    )
    return result.state


def run_iris_classification(profile, state, X_te, y_te):
    """Classify iris test cases against the given state."""
    correct = 0
    for i in range(len(X_te)):
        pred = classify_case(
            X_te[i], profile, state,
            class_prefix="species_", feature_prefix="feat",
        )
        if pred == int(y_te[i]):
            correct += 1
    return correct / len(X_te)


def run_digits_classification(profile, state, X_te, y_te):
    """Classify digit test cases against the given state."""
    correct = 0
    for i in range(len(X_te)):
        pred = classify_case(
            X_te[i], profile, state,
            class_prefix="digit_", feature_prefix="px",
        )
        if pred == int(y_te[i]):
            correct += 1
    return correct / len(X_te)


def main():
    banner("SEQUENTIAL CROSS-DOMAIN TEST: one engine, multiple configurations")

    print("\n  Building IRIS configuration (fresh engine)...")
    iris_profile, iris_fresh_state, iris_X_te, iris_y_te = build_iris_state()
    print(f"    {len(iris_fresh_state.tendencies)} tendencies, "
          f"{sum(len(adj) for adj in iris_fresh_state.graph.weights.values()) // 2} edges")

    print("\n  Building DIGITS configuration (fresh engine)...")
    digits_profile, digits_fresh_state, digits_X_te, digits_y_te = build_digits_state()
    print(f"    {len(digits_fresh_state.tendencies)} tendencies, "
          f"{sum(len(adj) for adj in digits_fresh_state.graph.weights.values()) // 2} edges")

    # ---------- Baseline: each domain in its own fresh engine ----------
    banner("Baseline: each domain in its own fresh engine")
    iris_baseline = run_iris_classification(
        iris_profile, iris_fresh_state, iris_X_te, iris_y_te
    )
    print(f"\n  IRIS  fresh engine accuracy: {iris_baseline:.1%}")
    digits_baseline = run_digits_classification(
        digits_profile, digits_fresh_state, digits_X_te, digits_y_te
    )
    print(f"  DIGITS fresh engine accuracy: {digits_baseline:.1%}")

    # ---------- Sequential: same engine, reseed iris -> digits ----------
    banner("Sequential: iris first, then RESEED INTO digits")

    # Step 1: start with iris configuration
    print("\n  Step 1: classify iris cases on iris-configured engine")
    print("    (this matches the baseline above; just re-running for the trace)")
    seq_iris_acc = run_iris_classification(
        iris_profile, iris_fresh_state, iris_X_te, iris_y_te
    )
    print(f"    accuracy: {seq_iris_acc:.1%}")

    # Step 2: reseed into digits
    print("\n  Step 2: reseed engine from iris -> digits configuration")
    print("    (substitutions: remove all iris tendencies, add all digits tendencies)")
    reseeded_state = reseed_into_target_config(iris_fresh_state, digits_fresh_state)
    print(f"    after reseed: {len(reseeded_state.tendencies)} tendencies "
          f"(expected {len(digits_fresh_state.tendencies)})")

    # Step 3: classify digits on the reseeded engine
    print("\n  Step 3: classify digit cases on the reseeded engine")
    seq_digits_acc = run_digits_classification(
        digits_profile, reseeded_state, digits_X_te, digits_y_te
    )
    print(f"    accuracy: {seq_digits_acc:.1%}")

    # ---------- Reverse direction: digits -> iris ----------
    banner("Reverse: digits first, then RESEED INTO iris")

    print("\n  Step 1: classify digits on digits-configured engine")
    rev_digits_acc = run_digits_classification(
        digits_profile, digits_fresh_state, digits_X_te, digits_y_te
    )
    print(f"    accuracy: {rev_digits_acc:.1%}")

    print("\n  Step 2: reseed engine from digits -> iris configuration")
    rev_reseeded_state = reseed_into_target_config(digits_fresh_state, iris_fresh_state)
    print(f"    after reseed: {len(rev_reseeded_state.tendencies)} tendencies")

    print("\n  Step 3: classify iris on the reseeded engine")
    rev_iris_acc = run_iris_classification(
        iris_profile, rev_reseeded_state, iris_X_te, iris_y_te
    )
    print(f"    accuracy: {rev_iris_acc:.1%}")

    # ---------- Summary ----------
    banner("SUMMARY")
    print()
    print(f"                          fresh engine  after-reseed    delta")
    print(f"  iris (baseline -> reseeded-into):   "
          f"{iris_baseline:>5.1%}        {rev_iris_acc:>5.1%}     "
          f"{(rev_iris_acc - iris_baseline) * 100:+.1f} pts")
    print(f"  digits (baseline -> reseeded-into): "
          f"{digits_baseline:>5.1%}        {seq_digits_acc:>5.1%}     "
          f"{(seq_digits_acc - digits_baseline) * 100:+.1f} pts")

    print()
    iris_delta = rev_iris_acc - iris_baseline
    digits_delta = seq_digits_acc - digits_baseline
    if abs(iris_delta) < 0.05 and abs(digits_delta) < 0.05:
        print("  Reseeding preserves accuracy. The engine genuinely handles")
        print("  sequential domain switches via the universal reseed operation.")
        print("  The architecture's universality claim holds for sequential use.")
    elif iris_delta < -0.05 or digits_delta < -0.05:
        print("  Reseeding DEGRADES accuracy. The engine carries residue from")
        print("  previous configurations, likely through the lineage events that")
        print("  reseed_and_equilibrate emits.")
    else:
        print("  Reseeding produces small variation but not significant degradation.")
    print()


if __name__ == "__main__":
    main()
