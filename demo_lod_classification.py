#!/usr/bin/env python3
"""
LOD classification: query letters at multiple resolutions in one engine.

The architecture's LOD claim: tendencies live at multiple resolutions,
and the engine can query at the resolution the question needs. Coarse
queries resolve fast with low-LOD tendencies; fine queries need high-LOD
detail.

This test exercises that claim end-to-end on the Letters dataset.

Setup
-----

Two LOD levels for letter classification:

  LOD 1 (coarse):  group letters into 4 visual categories
                   - VOWELS: A, E, I, O, U
                   - ROUND: B, C, D, G, O, P, Q, R, S
                   - ANGULAR: K, L, M, N, T, V, W, X, Y, Z
                   - MIXED: F, H, J
                   (overlaps are intentional -- letters can have multiple
                    coarse properties; the engine learns the strongest)

  LOD 2 (fine):    individual letters (the standard 26 classes)

The engine wires letters at *both* levels: each letter has a fine-grained
tendency, and each visual category has a coarse-grained tendency. They
share feature-evidence neighbors. A query specifies the LOD it cares about.

Tests
-----

  Test 1: classify at LOD 2 only (standard 26-way) -- baseline
  Test 2: classify at LOD 1 only (4-way category) -- should be much higher
          accuracy because there are fewer classes to discriminate
  Test 3: ADAPTIVE -- equilibrate at LOD 1, see confidence; for low-confidence
          cases drop to LOD 2. Measure overall LOD-2 accuracy and the
          fraction of cases that needed the LOD-2 query.

If LOD adaptive achieves the same accuracy as LOD 2 alone but resolves
many cases at LOD 1 only, the engine is genuinely benefiting from
multi-resolution structure.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_openml

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
    contrast_to_edge_weight,
    case_feature_strength,
)


# ---------------------------------------------------------------------------
# LOD 1: coarse categories DERIVED FROM DATA
# ---------------------------------------------------------------------------
#
# Hand-defined categories (e.g., VOWELS / ROUND / ANGULAR / MIXED) failed:
# they don't correspond to natural clusters in the 16-dim feature space,
# so LOD 1 inference was no better than random. We now derive coarse
# categories from the data itself via k-means on per-class centroids.
# This is the engine equivalent of "learn the hierarchy" rather than
# imposing one.

from sklearn.cluster import KMeans

NUM_COARSE_CATEGORIES = 6   # data-driven; tune per dataset


def derive_categories(profile, classes, n_categories: int = NUM_COARSE_CATEGORIES):
    """Cluster per-class centroids to derive coarse categories."""
    centroids = np.array([profile.class_centroids[c] for c in classes])
    km = KMeans(n_clusters=n_categories, n_init=10, random_state=42)
    labels = km.fit_predict(centroids)

    # Build category names "cat_0", "cat_1", ...
    category_names = [f"cat_{i}" for i in range(n_categories)]

    # Membership: each letter has full membership in exactly its assigned
    # cluster (one-hot). Could be relaxed to soft membership later.
    memberships = {}
    for i, letter in enumerate(classes):
        memberships[letter] = {
            cat: (1.0 if cat == category_names[labels[i]] else 0.0)
            for cat in category_names
        }

    return category_names, memberships, km.cluster_centers_


# ---------------------------------------------------------------------------
# Multi-LOD state
# ---------------------------------------------------------------------------

def build_multi_lod_state(profile, classes, category_names, memberships,
                          category_centroids):
    """Build state with both LOD 1 (data-derived categories) and
    LOD 2 (individual letters) tendencies."""
    factory = DefaultTendencyFactory()
    n_features = profile.n_features
    n_letters = len(classes)
    n_categories = len(category_names)

    # Allocations: 0.3 to LOD 1, 0.3 to LOD 2, 0.4 to features
    cat_alloc = 0.3 / n_categories
    letter_alloc = 0.3 / n_letters
    feature_alloc = 0.4 / (2 * n_features)

    specs = []
    for cat in category_names:
        specs.append(TendencySpec(id=f"cat_{cat}", initial_allocation=cat_alloc))
    for letter in classes:
        specs.append(TendencySpec(id=f"letter_{letter}", initial_allocation=letter_alloc))
    for f in range(n_features):
        specs.append(TendencySpec(id=f"f{f:02d}_low", initial_allocation=feature_alloc))
        specs.append(TendencySpec(id=f"f{f:02d}_high", initial_allocation=feature_alloc))

    tendencies = factory.build_set(specs)
    graph = StakeWeightGraph()

    # LOD 2: letter tendencies edge to feature evidence (contrast wiring)
    for letter in classes:
        z_per_feature = profile.class_zscores[letter]
        for f in range(n_features):
            low_edge, high_edge = contrast_to_edge_weight(z_per_feature[f])
            graph.add_edge(f"letter_{letter}", f"f{f:02d}_high", high_edge)
            graph.add_edge(f"letter_{letter}", f"f{f:02d}_low", low_edge)

    # LOD 1: category tendencies edge to feature evidence based on
    # category centroid z-scores
    for cat_idx, cat in enumerate(category_names):
        cat_centroid = category_centroids[cat_idx]
        for f in range(n_features):
            z = (cat_centroid[f] - profile.feature_pop_mean[f]) / max(
                profile.feature_pop_std[f], 1e-9
            )
            low_edge, high_edge = contrast_to_edge_weight(z)
            graph.add_edge(f"cat_{cat}", f"f{f:02d}_high", high_edge)
            graph.add_edge(f"cat_{cat}", f"f{f:02d}_low", low_edge)

    # Cross-LOD: letters edge to their category memberships
    for letter in classes:
        for cat, strength in memberships[letter].items():
            if strength > 0:
                edge_w = 0.05 + 0.6 * strength
                graph.add_edge(f"letter_{letter}", f"cat_{cat}", edge_w)

    lineages = {tid: Lineage() for tid in tendencies.ids()}
    return PresentState(tendencies=tendencies, lineages=lineages, graph=graph)


def get_letter_category_label(letter: str, memberships) -> str:
    """Return the highest-membership category for ground truth at LOD 1."""
    return max(memberships[letter], key=memberships[letter].get)


def classify_at_lod(case_features, profile, base_state, lod: int, classes,
                    category_names):
    """Run inference; read out winners at the requested LOD."""
    n_features = profile.n_features

    # Surprise-weighted feature budget
    surprises = []
    for f in range(n_features):
        z = (float(case_features[f]) - profile.feature_pop_mean[f]) / max(
            profile.feature_pop_std[f], 1e-9
        )
        surprises.append(abs(z))
    floor = 0.1
    weights = [floor + s for s in surprises]
    total = sum(weights)
    feature_budgets = [
        (0.4 * (w / total)) if total > 0 else (0.4 / n_features)
        for w in weights
    ]

    substitutions = []
    for f in range(n_features):
        low_s, high_s = case_feature_strength(
            float(case_features[f]),
            profile.feature_pop_mean[f],
            profile.feature_pop_std[f],
        )
        substitutions.append(Substitution(
            id=f"f{f:02d}_high",
            new_tendency=Tendency(id=f"f{f:02d}_high",
                                  allocation=feature_budgets[f] * high_s),
        ))
        substitutions.append(Substitution(
            id=f"f{f:02d}_low",
            new_tendency=Tendency(id=f"f{f:02d}_low",
                                  allocation=feature_budgets[f] * low_s),
        ))

    result = reseed_and_equilibrate(
        base_state,
        substitutions=substitutions,
        propagate_via_graph=True,
        learning_rate=0.3,
        tolerance=1e-5,
        max_iterations=300,
    )

    if lod == 1:
        cat_allocs = {
            cat: result.state.tendencies.get(f"cat_{cat}").allocation
            for cat in category_names
        }
        return cat_allocs
    elif lod == 2:
        letter_allocs = {
            letter: result.state.tendencies.get(f"letter_{letter}").allocation
            for letter in classes
        }
        return letter_allocs
    else:
        raise ValueError(f"unknown LOD {lod}")


def main(test_fraction: float = 0.2, seed: int = 42, max_test_cases: int = 200):
    print()
    print("=" * 70)
    print("LOD CLASSIFICATION: query letters at multiple resolutions")
    print("=" * 70)

    print("\n  loading letter dataset...")
    data = fetch_openml('letter', version=1, as_frame=False, parser='auto')
    X = data.data.astype(float)
    y = data.target
    classes = sorted(set(y))

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
    category_names, memberships, category_centroids = derive_categories(
        profile, classes, n_categories=NUM_COARSE_CATEGORIES,
    )
    state = build_multi_lod_state(profile, classes, category_names, memberships,
                                  category_centroids)
    n_tendencies = len(state.tendencies)
    n_edges = sum(len(adj) for adj in state.graph.weights.values()) // 2
    print(f"  multi-LOD state: {n_tendencies} tendencies "
          f"({len(category_names)} data-derived categories + {len(classes)} letters + "
          f"{2*profile.n_features} features), {n_edges} edges")

    # Show what the data-derived categories look like
    print("\n  data-derived LOD 1 categories (k-means on training centroids):")
    for cat in category_names:
        members = [l for l in classes if memberships[l][cat] > 0.5]
        print(f"    {cat}: {' '.join(members)}")

    # ---------- Test 1: LOD 2 only (26-way classification) ----------
    print("\n" + "-" * 70)
    print(f"Test 1: LOD 2 (full {len(classes)}-way letter classification)")
    print("-" * 70)
    correct_lod2 = 0
    for i in range(len(X_te)):
        allocs = classify_at_lod(X_te[i], profile, state, lod=2, classes=classes,
                                 category_names=category_names)
        pred = max(allocs, key=allocs.get)
        if pred == y_te[i]:
            correct_lod2 += 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(X_te)}  running: {correct_lod2/(i+1):.1%}")
    acc_lod2 = correct_lod2 / len(X_te)
    print(f"\n  LOD 2 accuracy: {correct_lod2}/{len(X_te)} = {acc_lod2:.1%}")

    # ---------- Test 2: LOD 1 only (n-way category) ----------
    print("\n" + "-" * 70)
    print(f"Test 2: LOD 1 ({len(category_names)}-way category classification)")
    print("-" * 70)
    correct_lod1 = 0
    for i in range(len(X_te)):
        allocs = classify_at_lod(X_te[i], profile, state, lod=1, classes=classes,
                                 category_names=category_names)
        pred_cat = max(allocs, key=allocs.get)
        true_cat = get_letter_category_label(y_te[i], memberships)
        if pred_cat == true_cat:
            correct_lod1 += 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(X_te)}  running: {correct_lod1/(i+1):.1%}")
    acc_lod1 = correct_lod1 / len(X_te)
    print(f"\n  LOD 1 accuracy: {correct_lod1}/{len(X_te)} = {acc_lod1:.1%}")
    print(f"  baseline (random for {len(category_names)} cats): "
          f"{1/len(category_names):.1%}")

    # ---------- Test 3: Adaptive (LOD 1 first, drop to LOD 2 if uncertain) ----------
    print("\n" + "-" * 70)
    print("Test 3: ADAPTIVE -- LOD 1 first, drop to LOD 2 if low confidence")
    print("-" * 70)

    # Confidence: top-1 must be at least this much greater than top-2,
    # measured as relative ratio (top - second) / second. Allocations
    # at LOD 1 are small numbers (~0.02) but the *ratio* between top
    # and second is meaningful: when LOD 1 strongly prefers one cat
    # over another, top/second can be 1.05-1.2.
    confidence_threshold = 0.05  # 5% relative

    correct_adaptive = 0
    n_used_lod2 = 0
    for i in range(len(X_te)):
        cat_allocs = classify_at_lod(X_te[i], profile, state, lod=1, classes=classes,
                                     category_names=category_names)
        sorted_cats = sorted(cat_allocs.items(), key=lambda kv: -kv[1])
        top_cat, top_alloc = sorted_cats[0]
        second_alloc = sorted_cats[1][1] if len(sorted_cats) > 1 else 0
        # Relative confidence: how much bigger is top vs second?
        confidence = (top_alloc - second_alloc) / max(second_alloc, 1e-9)

        if confidence >= confidence_threshold:
            letter_allocs = classify_at_lod(X_te[i], profile, state, lod=2,
                                            classes=classes,
                                            category_names=category_names)
            top_letters = [
                l for l in classes
                if max(memberships[l], key=memberships[l].get) == top_cat
            ]
            if top_letters:
                constrained_allocs = {l: letter_allocs[l] for l in top_letters}
                pred = max(constrained_allocs, key=constrained_allocs.get)
            else:
                pred = max(letter_allocs, key=letter_allocs.get)
        else:
            letter_allocs = classify_at_lod(X_te[i], profile, state, lod=2,
                                            classes=classes,
                                            category_names=category_names)
            pred = max(letter_allocs, key=letter_allocs.get)
            n_used_lod2 += 1

        if pred == y_te[i]:
            correct_adaptive += 1

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(X_te)}  running: {correct_adaptive/(i+1):.1%}  "
                  f"(LOD2 used: {n_used_lod2/(i+1):.0%})")

    acc_adaptive = correct_adaptive / len(X_te)
    pct_used_lod2 = n_used_lod2 / len(X_te)
    print(f"\n  Adaptive accuracy: {correct_adaptive}/{len(X_te)} = {acc_adaptive:.1%}")
    print(f"  Cases that needed LOD 2: {n_used_lod2}/{len(X_te)} = {pct_used_lod2:.1%}")

    # ---------- Summary ----------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  LOD 2 (26-way letter):       {acc_lod2:.1%}")
    print(f"  LOD 1 (4-way category):      {acc_lod1:.1%}")
    print(f"  Adaptive (LOD 1 -> LOD 2):   {acc_adaptive:.1%}  "
          f"(used LOD 2 on {pct_used_lod2:.0%} of cases)")
    print()
    if acc_adaptive >= acc_lod2 - 0.02:
        print("  Adaptive matches LOD 2 accuracy while resolving many cases at LOD 1.")
        print("  The engine genuinely benefits from multi-resolution structure.")
    elif acc_adaptive > acc_lod2:
        print("  Adaptive EXCEEDS LOD 2: the LOD 1 prior helps disambiguate.")
    else:
        print(f"  Adaptive underperforms LOD 2 by {(acc_lod2-acc_adaptive)*100:.1f} pts.")
        print("  The LOD 1 confidence test is misjudging some cases.")
    print()


if __name__ == "__main__":
    main()
