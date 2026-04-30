#!/usr/bin/env python3
"""
Feature imputation: generate hidden features conditioned on visible ones.

The engine isn't just a discriminator; it's supposed to be a generative
substrate that produces coherent states from partial information. This
test exercises that capability.

Setup
-----

Take a labeled dataset (digits). For each test case:
  1. Hide some features (set to None / drop substitutions for them).
  2. Substitute only the visible features into the engine.
  3. Let the engine equilibrate.
  4. Read out the *predicted* allocations of the hidden feature-evidence
     tendencies.
  5. Convert back to predicted feature values.
  6. Compare to ground truth.

Two metrics:
  - Imputation error (how close are predicted feature values to actuals)
  - Downstream classification: does using the imputed features still
    let the engine classify correctly?

Why this matters architecturally
--------------------------------

The graph-walk relaxation we built is bidirectional: features pull on
classes, classes pull on features. We've only used the
features->classes direction (substitute features, read class). The
classes->features direction is the same engine, run with the same
operation, just reading out different tendencies.

If the engine cleanly generates plausible features for hidden
positions when given a class context (or partial features), it's
genuinely operating as a generative substrate, not a one-way classifier.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
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
from world_model.analysis.classifier_wiring import (
    fit_classifier_profile,
    build_classifier_state,
    classify_case,
    contrast_to_edge_weight,
    case_feature_strength,
)


def imput_and_classify(
    case_features,
    visible_indices: list,
    profile,
    base_state,
    class_prefix: str = "digit_",
    feature_prefix: str = "px",
):
    """Substitute only visible features; read out class allocations AND
    imputed feature values for hidden positions.

    Returns:
      predicted_class: id of the class with highest allocation
      imputed_features: dict {feature_index -> imputed_value}
                        for features NOT in visible_indices
    """
    n_features = profile.n_features
    visible_set = set(visible_indices)
    hidden_indices = [f for f in range(n_features) if f not in visible_set]

    # Substitute only visible features
    feature_budget = 0.5 / max(len(visible_indices), 1)

    substitutions = []
    for f in visible_indices:
        low_strength, high_strength = case_feature_strength(
            float(case_features[f]),
            profile.feature_pop_mean[f],
            profile.feature_pop_std[f],
        )
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_high",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_high",
                allocation=feature_budget * high_strength,
            ),
        ))
        substitutions.append(Substitution(
            id=f"{feature_prefix}{f:02d}_low",
            new_tendency=Tendency(
                id=f"{feature_prefix}{f:02d}_low",
                allocation=feature_budget * low_strength,
            ),
        ))

    # Hidden features: do NOT substitute -- let the engine equilibrate
    # them based on graph propagation.

    result = reseed_and_equilibrate(
        base_state,
        substitutions=substitutions,
        propagate_via_graph=True,
        learning_rate=0.3,
        tolerance=1e-5,
        max_iterations=300,
    )

    # Read out class allocations
    class_allocs = {
        c: result.state.tendencies.get(f"{class_prefix}{c}").allocation
        for c in profile.classes
    }
    predicted_class = max(class_allocs, key=class_allocs.get)

    # Read out imputed feature values
    imputed_features = {}
    for f in hidden_indices:
        high_alloc = result.state.tendencies.get(f"{feature_prefix}{f:02d}_high").allocation
        low_alloc = result.state.tendencies.get(f"{feature_prefix}{f:02d}_low").allocation

        # Convert relative high vs low allocations back to a feature
        # value via inverse of the tanh/z-score mapping. We use a simple
        # relative read: the ratio of high to (high+low) maps to feature
        # value position within [pop_mean - 2*std, pop_mean + 2*std].
        total = high_alloc + low_alloc
        if total > 0:
            high_share = high_alloc / total   # in [0, 1]
        else:
            high_share = 0.5
        # Inverse map: high_share=1 means very high z, =0 very low z
        # Use linear approx: z ~= (high_share - 0.5) * 4
        z = (high_share - 0.5) * 4
        imputed_value = profile.feature_pop_mean[f] + z * profile.feature_pop_std[f]
        imputed_features[f] = imputed_value

    return predicted_class, imputed_features


def main(test_fraction: float = 0.3, seed: int = 42, max_test_cases: int = 100,
         hide_fraction: float = 0.5):
    print()
    print("=" * 70)
    print("FEATURE IMPUTATION: generate hidden features from visible ones")
    print("=" * 70)

    digits = load_digits()
    X, y = digits.data, digits.target
    classes = list(range(10))

    rng = random.Random(seed)
    indices = list(range(len(y)))
    rng.shuffle(indices)
    split = int(len(y) * (1.0 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]
    if len(test_idx) > max_test_cases:
        test_idx = test_idx[:max_test_cases]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    profile = fit_classifier_profile(X_tr, y_tr, classes)
    base_state = build_classifier_state(
        profile, class_prefix="digit_", feature_prefix="px",
    )

    n_features = profile.n_features
    n_hidden = int(n_features * hide_fraction)
    n_visible = n_features - n_hidden

    print(f"\n  digits: 10 classes, {n_features} pixel features")
    print(f"  hide {hide_fraction:.0%} = {n_hidden} pixels per case at random")
    print(f"  visible: {n_visible} pixels; engine imputes the rest")
    print(f"  test cases: {len(X_te)}")

    # ---------- Baseline: full features classification ----------
    print("\n" + "-" * 70)
    print("Baseline: classification with all features visible")
    print("-" * 70)
    baseline_correct = 0
    for i in range(len(X_te)):
        pred = classify_case(
            X_te[i], profile, base_state,
            class_prefix="digit_", feature_prefix="px",
        )
        if pred == int(y_te[i]):
            baseline_correct += 1
    baseline_acc = baseline_correct / len(X_te)
    print(f"  full-feature accuracy: {baseline_acc:.1%}")

    # ---------- Imputation test ----------
    print("\n" + "-" * 70)
    print(f"Imputation: hide {hide_fraction:.0%} of features per case randomly")
    print("-" * 70)

    feature_rng = random.Random(seed + 1)

    impute_classify_correct = 0
    total_imputation_error = 0.0
    total_imputation_samples = 0
    per_visible_results = {}   # n_visible -> (correct, total)

    # Also test classification with feature values RANDOMLY filled in
    # for hidden positions (a simple baseline for "did imputation help?")
    naive_correct = 0

    for i in range(len(X_te)):
        # Randomly select visible feature indices for this case
        visible_indices = sorted(feature_rng.sample(range(n_features), n_visible))
        hidden_indices = [f for f in range(n_features) if f not in set(visible_indices)]

        # Engine imputes
        predicted_class, imputed_features = imput_and_classify(
            X_te[i], visible_indices, profile, base_state,
            class_prefix="digit_", feature_prefix="px",
        )

        # Compute imputation error (per hidden feature)
        for f in hidden_indices:
            actual = float(X_te[i][f])
            imputed = imputed_features[f]
            total_imputation_error += abs(actual - imputed)
            total_imputation_samples += 1

        # Did the engine still classify correctly with hidden features?
        if predicted_class == int(y_te[i]):
            impute_classify_correct += 1

        # Naive baseline: classify with hidden features replaced by population mean
        case_with_means = list(X_te[i])
        for f in hidden_indices:
            case_with_means[f] = profile.feature_pop_mean[f]
        naive_pred = classify_case(
            case_with_means, profile, base_state,
            class_prefix="digit_", feature_prefix="px",
        )
        if naive_pred == int(y_te[i]):
            naive_correct += 1

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(X_te)}: imputation accuracy {impute_classify_correct/(i+1):.1%}")

    impute_acc = impute_classify_correct / len(X_te)
    naive_acc = naive_correct / len(X_te)
    avg_imputation_error = (total_imputation_error / total_imputation_samples
                            if total_imputation_samples > 0 else 0)
    # Approximate feature range as 4x average std (covers ~95% of values)
    avg_std = sum(profile.feature_pop_std) / len(profile.feature_pop_std)
    feature_range = 4 * avg_std
    relative_error = avg_imputation_error / max(feature_range, 1e-9)

    print(f"\n  imputation classification accuracy: {impute_acc:.1%}")
    print(f"  naive (mean-fill) classification:   {naive_acc:.1%}")
    print(f"  full-feature baseline:              {baseline_acc:.1%}")
    print(f"\n  average imputation error: {avg_imputation_error:.3f}")
    print(f"  feature range: {feature_range:.1f}")
    print(f"  relative error: {relative_error:.1%}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  full features baseline:        {baseline_acc:.1%}")
    print(f"  engine imputes hidden:         {impute_acc:.1%}  "
          f"(degradation: {(impute_acc - baseline_acc) * 100:+.1f} pts)")
    print(f"  naive mean-fill classification: {naive_acc:.1%}  "
          f"(degradation: {(naive_acc - baseline_acc) * 100:+.1f} pts)")
    print()
    print(f"  imputation MAE: {avg_imputation_error:.3f} ({relative_error:.1%} of feature range)")
    print()

    if impute_acc >= naive_acc - 0.02:
        if impute_acc >= baseline_acc - 0.05:
            print("  Engine handles imputation as well as the naive baseline AND")
            print("  classification accuracy holds up. The graph relaxation is")
            print("  productively reasoning about hidden features.")
        else:
            print("  Engine imputation matches naive mean-fill but classification")
            print("  degrades. The graph propagation IS bidirectional but the")
            print("  imputed values aren't more useful than population means.")
    else:
        print("  Engine imputation UNDERPERFORMS naive mean-fill. Hidden-feature")
        print("  predictions through graph relaxation are noisier than just")
        print("  using population averages. Architecture isn't generative here.")
    print()


if __name__ == "__main__":
    main()
