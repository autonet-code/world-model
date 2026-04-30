#!/usr/bin/env python3
"""
Iris classification via reseed-and-equilibrate.

Goal: validate that the engine can actually do something useful end-to-end
on a small labeled dataset. Iris is famously trivial -- a logistic
regression handles it in five lines. The point here isn't to beat that.
The point is that *the same engine that ran our fictional-world demo*
can be pointed at a classification task and produce a meaningful
readout, with no domain-specific code in the engine itself.

Mapping
-------

  - 3 species tendencies (setosa, versicolor, virginica). Allocation =
    current belief that the case-under-test belongs to this species.

  - 8 feature-evidence tendencies. For each of the 4 features, two
    positional tendencies: "<feature>_low" and "<feature>_high". These
    are the engine's "axis markers."

  - The stake-weight graph encodes per-species feature profiles. From
    a training set, we compute each species' mean feature values; the
    edge weight between species S and feature-tendency F_high is large
    when S's mean for F is at the high end of the range, small when at
    the low end. This is "the engine's prior."

  - To classify a test case: substitute the feature-evidence tendencies
    with allocations reflecting the case's actual feature values, hold
    the species tendencies open, run reseed_and_equilibrate. The species
    tendency whose allocation grows the most is the prediction.

This wiring is intentionally simple. We're not trying to engineer a
strong classifier; we're checking that the engine can absorb evidence
through substitution and produce a coherent post-equilibrium readout.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from sklearn.datasets import load_iris

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
# Configuration
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

SPECIES = ["setosa", "versicolor", "virginica"]


# ---------------------------------------------------------------------------
# Build the engine's prior from training data
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    """Per-species feature statistics derived from the training set."""
    species_centroids: dict[str, list[float]]   # species -> [mean per feature]
    feature_min: list[float]
    feature_max: list[float]


def fit_profile(X_train, y_train) -> Profile:
    """Compute species centroids and feature ranges. No actual ML training --
    we just gather per-species means and global feature ranges."""
    centroids: dict[str, list[float]] = {}
    for s_idx, species in enumerate(SPECIES):
        rows = [X_train[i] for i in range(len(y_train)) if y_train[i] == s_idx]
        n = len(rows)
        if n == 0:
            centroids[species] = [0.0] * X_train.shape[1]
            continue
        centroids[species] = [
            sum(r[f] for r in rows) / n
            for f in range(X_train.shape[1])
        ]
    feature_min = [float(X_train[:, f].min()) for f in range(X_train.shape[1])]
    feature_max = [float(X_train[:, f].max()) for f in range(X_train.shape[1])]
    return Profile(
        species_centroids=centroids,
        feature_min=feature_min,
        feature_max=feature_max,
    )


def normalize_feature(value: float, lo: float, hi: float) -> float:
    """Map a raw value into [0, 1] within the global feature range."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# ---------------------------------------------------------------------------
# Build a starting present state
# ---------------------------------------------------------------------------

def build_starting_state(profile: Profile) -> PresentState:
    """Construct an 11-tendency state: 3 species + 8 feature-evidence.

    The feature-evidence tendencies start at neutral allocation. The
    species tendencies start with equal allocation. The stake-graph
    edges encode the engine's prior: per-species centroid affinity
    with feature-evidence axis markers.
    """
    factory = DefaultTendencyFactory()

    # Equal allocation across species (3) + 8 feature-evidence tendencies.
    # Each species starts at ~equal belief; feature tendencies at smaller
    # neutral values that get reset per case.
    n_total = len(SPECIES) + 2 * len(FEATURE_NAMES)
    species_alloc_each = 0.5 / len(SPECIES)        # species share half the mass
    feature_alloc_each = 0.5 / (2 * len(FEATURE_NAMES))   # other half spread across features

    specs = []
    for sp in SPECIES:
        specs.append(TendencySpec(
            id=f"species_{sp}",
            initial_allocation=species_alloc_each,
            description=f"Belief that the case is {sp}",
        ))
    for f in FEATURE_NAMES:
        specs.append(TendencySpec(
            id=f"{f}_low",
            initial_allocation=feature_alloc_each,
            description=f"Evidence that {f} is at the low end of its range",
        ))
        specs.append(TendencySpec(
            id=f"{f}_high",
            initial_allocation=feature_alloc_each,
            description=f"Evidence that {f} is at the high end of its range",
        ))

    tendencies = factory.build_set(specs)

    # Build the stake-weight graph. Edge weight between species S and
    # feature_high tendency = (centroid_S[feature] - feature_min) / range.
    # Edge weight to feature_low = 1 - that. Higher edge weight = stronger
    # association.
    graph = StakeWeightGraph()
    for sp in SPECIES:
        centroid = profile.species_centroids[sp]
        for f_idx, fname in enumerate(FEATURE_NAMES):
            high_aff = normalize_feature(
                centroid[f_idx], profile.feature_min[f_idx], profile.feature_max[f_idx]
            )
            low_aff = 1.0 - high_aff
            # Squash to a useful range so weights aren't 0 or 1.
            graph.add_edge(f"species_{sp}", f"{fname}_high", 0.1 + 0.8 * high_aff)
            graph.add_edge(f"species_{sp}", f"{fname}_low",  0.1 + 0.8 * low_aff)

    lineages = {tid: Lineage() for tid in tendencies.ids()}
    return PresentState(tendencies=tendencies, lineages=lineages, graph=graph)


# ---------------------------------------------------------------------------
# Classify a single case
# ---------------------------------------------------------------------------

def classify(case_features, profile: Profile, base_state: PresentState) -> str:
    """Substitute feature-evidence tendencies for this case and let the engine
    propagate that evidence through the stake-graph to reshape the species
    tendencies' allocations. The species tendency whose allocation lands
    highest after calibration is the prediction.

    No species-side computation in this function. The engine does the work.
    """
    substitutions = []
    for f_idx, fname in enumerate(FEATURE_NAMES):
        high_strength = normalize_feature(
            case_features[f_idx],
            profile.feature_min[f_idx],
            profile.feature_max[f_idx],
        )
        low_strength = 1.0 - high_strength

        # Each feature gets 0.5/4 = 0.125 of the total mass, split into
        # high/low components proportional to the case's evidence.
        feature_budget = 0.5 / len(FEATURE_NAMES)
        substitutions.append(Substitution(
            id=f"{fname}_high",
            new_tendency=Tendency(
                id=f"{fname}_high",
                allocation=feature_budget * high_strength,
            ),
        ))
        substitutions.append(Substitution(
            id=f"{fname}_low",
            new_tendency=Tendency(
                id=f"{fname}_low",
                allocation=feature_budget * low_strength,
            ),
        ))

    # The species tendencies are NOT substituted. Their targets are
    # computed each calibration iteration as a weighted average of
    # their stake-graph neighbors -- which are the feature-evidence
    # tendencies we just substituted. Species with stake-edges to
    # feature-evidence tendencies that received high allocations will
    # rise; species connected to feature-evidence tendencies that
    # received low allocations will fall.
    result = reseed_and_equilibrate(
        base_state,
        substitutions=substitutions,
        propagate_via_graph=True,
        learning_rate=0.3,
        tolerance=1e-5,
        max_iterations=300,
    )

    species_allocs = {
        sp: result.state.tendencies.get(f"species_{sp}").allocation
        for sp in SPECIES
    }
    return max(species_allocs, key=species_allocs.get)


# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------

def main(test_fraction: float = 0.3, seed: int = 42) -> int:
    print()
    print("=" * 70)
    print("IRIS CLASSIFICATION VIA RESEED-AND-EQUILIBRATE")
    print("=" * 70)

    iris = load_iris()
    X, y = iris.data, iris.target
    n = len(y)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * (1.0 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"\n  loaded iris: {n} cases, 4 features, 3 species")
    print(f"  train/test split: {len(train_idx)}/{len(test_idx)} (seed={seed})")

    # Fit per-species centroids (the engine's prior)
    profile = fit_profile(X_train, y_train)
    print(f"\n  per-species centroids learned from training set:")
    for sp, centroid in profile.species_centroids.items():
        formatted = "  ".join(f"{v:.2f}" for v in centroid)
        print(f"    {sp:11s}  [{formatted}]")

    # Build the starting state once; reuse for each test case.
    base_state = build_starting_state(profile)
    print(f"\n  starting state: {len(base_state.tendencies)} tendencies, "
          f"{sum(len(adj) for adj in base_state.graph.weights.values()) // 2} edges")

    # Classify each test case
    print(f"\n  classifying {len(test_idx)} test cases...")
    correct = 0
    confusion: dict[tuple[int, int], int] = Counter()
    for i, case_idx in enumerate(test_idx):
        features = X_test[i]
        true_label = SPECIES[y_test[i]]
        prediction = classify(features, profile, base_state)
        if prediction == true_label:
            correct += 1
        confusion[(y_test[i], SPECIES.index(prediction))] += 1

    accuracy = correct / len(test_idx)
    print(f"\n  accuracy: {correct}/{len(test_idx)} = {accuracy:.1%}")
    print(f"  baseline (random): {1 / len(SPECIES):.1%}")

    print(f"\n  confusion matrix (row = true, col = predicted):")
    print(f"             {'  '.join(f'{s:>11s}' for s in SPECIES)}")
    for i, sp in enumerate(SPECIES):
        cells = []
        for j in range(len(SPECIES)):
            count = confusion[(i, j)]
            cells.append(f"{count:>11d}")
        print(f"    {sp:11s}{'  '.join(cells)}")

    print()
    print("=" * 70)
    if accuracy >= 0.85:
        print(f"RESULT: engine recovered iris classifications at {accuracy:.0%} accuracy.")
        print("        This validates that reseed-and-equilibrate, given an")
        print("        appropriate stake-graph wiring, can absorb evidence")
        print("        and produce a coherent equilibrium readout.")
    elif accuracy >= 1 / len(SPECIES) + 0.15:
        print(f"RESULT: engine accuracy {accuracy:.0%} is well above random ({1/len(SPECIES):.0%}).")
        print("        Engine is doing real work but the wiring needs refinement.")
    else:
        print(f"RESULT: engine accuracy {accuracy:.0%} is near or below random.")
        print("        The classification mapping or calibration is broken --")
        print("        diagnose before trusting the engine on richer domains.")
    print("=" * 70)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
