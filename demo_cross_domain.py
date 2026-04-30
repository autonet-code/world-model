#!/usr/bin/env python3
"""
Cross-domain shared-engine test.

The fractal/universal claim says one engine handles multiple domains
through a single substrate, with shared tendencies bridging them.
This test constructs two synthetic domains that share genuine
structural properties and runs them through one engine.

Setup
-----

Two synthetic domains, A and B. Each has 5 classes, each defined by a
4-dim centroid in its own feature space. The two domains share three
abstract structural properties:

  COMPACT     = how "tight" the class profile is (low spread)
  EXTENDED    = how "stretched" the class profile is (high spread on one axis)
  ASYMMETRIC  = how unbalanced the class profile is

Each class in domain A and domain B has a known signature on these
abstract properties. Class A_i and B_i are *defined* to share the same
abstract signature, so they should activate the same bridge tendencies.

The graph has:
  - 5 A-class tendencies + 8 A-feature-evidence tendencies
  - 5 B-class tendencies + 8 B-feature-evidence tendencies
  - 3 bridge tendencies (COMPACT, EXTENDED, ASYMMETRIC)

A-class tendencies edge to A-feature-evidence tendencies AND to bridge
tendencies. Same for B.

Test
----

  Test 1: classify A cases through engine with only A side + bridges.
  Test 2: classify B cases through engine with only B side + bridges.
  Test 3: classify A and B cases through engine with BOTH sides + bridges,
          one case from each domain at a time, simultaneously.

If accuracy in Test 3 matches Test 1 and Test 2, the engine handles
cross-domain without contamination. If Test 3 *exceeds* the controls,
the bridges help: cross-domain evidence flows through and improves
classification on each side.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

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
# Synthetic domain construction
# ---------------------------------------------------------------------------

# Bridge property signatures per class index. Both domains share these.
# Indices: 0=class_0, 1=class_1, ..., 4=class_4
# Properties: COMPACT, EXTENDED, ASYMMETRIC (each in [0, 1])
SHARED_SIGNATURES = {
    0: {"COMPACT": 0.9, "EXTENDED": 0.1, "ASYMMETRIC": 0.2},  # very compact, balanced
    1: {"COMPACT": 0.1, "EXTENDED": 0.9, "ASYMMETRIC": 0.3},  # very extended
    2: {"COMPACT": 0.5, "EXTENDED": 0.4, "ASYMMETRIC": 0.9},  # asymmetric
    3: {"COMPACT": 0.7, "EXTENDED": 0.3, "ASYMMETRIC": 0.5},  # mixed
    4: {"COMPACT": 0.3, "EXTENDED": 0.7, "ASYMMETRIC": 0.7},  # extended + asymmetric
}

BRIDGE_PROPERTIES = ["COMPACT", "EXTENDED", "ASYMMETRIC"]


def synthesize_domain(domain_name: str, n_features: int = 4, n_per_class: int = 50,
                      noise: float = 0.4, seed: int = 0):
    """Generate cases for 5 classes, each consistent with the shared signatures.

    Each domain has its own 4-dim feature space. The mapping from
    abstract signatures to feature centroids is domain-specific
    (different 4D directions encode COMPACT for A vs B), but the
    abstract signature is the same.

    Returns: X (n_cases, n_features), y (n_cases,), centroids (5, n_features)
    """
    rng = np.random.RandomState(seed)
    n_classes = 5

    # For each domain, pick a random orthonormal basis: each abstract
    # property maps to a different direction in the domain's feature space.
    direction_for_property = {}
    available_dims = list(range(n_features))
    rng.shuffle(available_dims)
    for i, prop in enumerate(BRIDGE_PROPERTIES):
        direction_for_property[prop] = available_dims[i]

    # Build per-class centroids in the domain's feature space
    centroids = np.zeros((n_classes, n_features))
    for class_idx in range(n_classes):
        sig = SHARED_SIGNATURES[class_idx]
        for prop, strength in sig.items():
            dim = direction_for_property[prop]
            centroids[class_idx, dim] = strength
        # Add a small amount of domain-specific structure on the
        # remaining dimension to ensure 5 distinct classes even when
        # signatures are similar.
        unused_dim = available_dims[len(BRIDGE_PROPERTIES)]
        centroids[class_idx, unused_dim] = class_idx / n_classes

    # Sample cases around each centroid with noise
    X = []
    y = []
    for class_idx in range(n_classes):
        for _ in range(n_per_class):
            sample = centroids[class_idx] + rng.randn(n_features) * noise
            X.append(sample)
            y.append(class_idx)
    return np.array(X), np.array(y), centroids


# ---------------------------------------------------------------------------
# Engine state construction
# ---------------------------------------------------------------------------

@dataclass
class DomainProfile:
    """Per-class statistics for one domain."""
    domain: str
    n_features: int
    classes: list
    class_centroids: dict
    feature_pop_mean: list
    feature_pop_std: list


def fit_profile(X_train, y_train, classes, domain: str) -> DomainProfile:
    n_features = X_train.shape[1]
    centroids = {}
    for c in classes:
        rows = [X_train[i] for i in range(len(y_train)) if y_train[i] == c]
        centroids[c] = list(np.mean(rows, axis=0)) if rows else [0.0] * n_features

    pop_mean = [float(np.mean(X_train[:, f])) for f in range(n_features)]
    pop_std = [max(float(np.std(X_train[:, f])), 1e-9) for f in range(n_features)]

    return DomainProfile(
        domain=domain,
        n_features=n_features,
        classes=list(classes),
        class_centroids=centroids,
        feature_pop_mean=pop_mean,
        feature_pop_std=pop_std,
    )


def contrast_to_edge_weight(z: float) -> tuple[float, float]:
    signed = math.tanh(z / 2.0)
    high = 0.5 + 0.5 * signed
    low = 0.5 - 0.5 * signed
    return 0.05 + 0.9 * low, 0.05 + 0.9 * high


def case_feature_strength(value: float, pop_mean: float, pop_std: float) -> tuple[float, float]:
    z = (value - pop_mean) / max(pop_std, 1e-9)
    signed = math.tanh(z / 2.0)
    high = 0.5 + 0.5 * signed
    low = 0.5 - 0.5 * signed
    return low, high


def build_state(
    domains: list,                  # list of (DomainProfile, profile_name) tuples to include
    include_bridges: bool = True,
) -> PresentState:
    """Build engine state with one or more domain sides plus optional bridges.

    Each domain contributes:
      - n_classes class tendencies (e.g., A_class_0, B_class_3)
      - 2 * n_features feature-evidence tendencies (e.g., A_f00_low)

    Bridges contribute 3 tendencies (COMPACT, EXTENDED, ASYMMETRIC)
    that connect to all domains' class tendencies via signature-based
    edges.
    """
    factory = DefaultTendencyFactory()
    specs = []

    # Count for budget allocation
    total_class_tendencies = sum(len(p.classes) for p, _ in domains)
    total_feature_tendencies = sum(2 * p.n_features for p, _ in domains)
    n_bridges = len(BRIDGE_PROPERTIES) if include_bridges else 0
    total = total_class_tendencies + total_feature_tendencies + n_bridges

    # Allocations: half to classes, half to features, small slice to bridges
    class_alloc_each = 0.4 / total_class_tendencies if total_class_tendencies else 0
    feature_alloc_each = 0.4 / total_feature_tendencies if total_feature_tendencies else 0
    bridge_alloc_each = 0.2 / n_bridges if n_bridges else 0

    for profile, prefix in domains:
        for c in profile.classes:
            specs.append(TendencySpec(
                id=f"{prefix}_class_{c}",
                initial_allocation=class_alloc_each,
            ))
        for f in range(profile.n_features):
            specs.append(TendencySpec(
                id=f"{prefix}_f{f:02d}_low",
                initial_allocation=feature_alloc_each,
            ))
            specs.append(TendencySpec(
                id=f"{prefix}_f{f:02d}_high",
                initial_allocation=feature_alloc_each,
            ))

    if include_bridges:
        for prop in BRIDGE_PROPERTIES:
            specs.append(TendencySpec(
                id=f"bridge_{prop}",
                initial_allocation=bridge_alloc_each,
            ))

    tendencies = factory.build_set(specs)
    graph = StakeWeightGraph()

    # Domain-internal edges: each class to its feature-evidence tendencies
    for profile, prefix in domains:
        for c in profile.classes:
            centroid = profile.class_centroids[c]
            for f in range(profile.n_features):
                z = (centroid[f] - profile.feature_pop_mean[f]) / max(
                    profile.feature_pop_std[f], 1e-9
                )
                low_edge, high_edge = contrast_to_edge_weight(z)
                graph.add_edge(f"{prefix}_class_{c}", f"{prefix}_f{f:02d}_high", high_edge)
                graph.add_edge(f"{prefix}_class_{c}", f"{prefix}_f{f:02d}_low", low_edge)

    # Bridge edges: class tendencies to bridge tendencies based on the
    # SHARED signature (same for both domains)
    if include_bridges:
        for profile, prefix in domains:
            for c in profile.classes:
                sig = SHARED_SIGNATURES[c]
                for prop, strength in sig.items():
                    # Edge weight reflects how strongly this class
                    # exhibits this abstract property
                    edge_weight = 0.1 + 0.7 * strength
                    graph.add_edge(
                        f"{prefix}_class_{c}",
                        f"bridge_{prop}",
                        edge_weight,
                    )

    lineages = {tid: Lineage() for tid in tendencies.ids()}
    return PresentState(tendencies=tendencies, lineages=lineages, graph=graph)


def classify(case_features, profile: DomainProfile, prefix: str,
             base_state: PresentState,
             paired_case=None, paired_profile=None, paired_prefix=None):
    """Classify a single case (or pair of cases from two domains).

    paired_* args allow simultaneous classification of two cases from
    different domains in one engine call.
    """
    n_features = profile.n_features

    # Compute surprise-weighted feature budgets
    feature_surprises = []
    for f in range(n_features):
        z = (float(case_features[f]) - profile.feature_pop_mean[f]) / max(
            profile.feature_pop_std[f], 1e-9
        )
        feature_surprises.append(abs(z))
    floor = 0.1
    weights = [floor + s for s in feature_surprises]
    total_w = sum(weights)
    feature_budgets = [
        (0.4 * (w / total_w)) if total_w > 0 else (0.4 / n_features)
        for w in weights
    ]

    substitutions = []
    for f in range(n_features):
        low_strength, high_strength = case_feature_strength(
            float(case_features[f]),
            profile.feature_pop_mean[f],
            profile.feature_pop_std[f],
        )
        substitutions.append(Substitution(
            id=f"{prefix}_f{f:02d}_high",
            new_tendency=Tendency(
                id=f"{prefix}_f{f:02d}_high",
                allocation=feature_budgets[f] * high_strength,
            ),
        ))
        substitutions.append(Substitution(
            id=f"{prefix}_f{f:02d}_low",
            new_tendency=Tendency(
                id=f"{prefix}_f{f:02d}_low",
                allocation=feature_budgets[f] * low_strength,
            ),
        ))

    # Add paired-domain substitutions if provided
    if paired_case is not None and paired_profile is not None and paired_prefix is not None:
        paired_n_features = paired_profile.n_features
        paired_surprises = []
        for f in range(paired_n_features):
            z = (float(paired_case[f]) - paired_profile.feature_pop_mean[f]) / max(
                paired_profile.feature_pop_std[f], 1e-9
            )
            paired_surprises.append(abs(z))
        paired_weights = [floor + s for s in paired_surprises]
        paired_total = sum(paired_weights)
        paired_budgets = [
            (0.4 * (w / paired_total)) if paired_total > 0 else (0.4 / paired_n_features)
            for w in paired_weights
        ]
        for f in range(paired_n_features):
            low_s, high_s = case_feature_strength(
                float(paired_case[f]),
                paired_profile.feature_pop_mean[f],
                paired_profile.feature_pop_std[f],
            )
            substitutions.append(Substitution(
                id=f"{paired_prefix}_f{f:02d}_high",
                new_tendency=Tendency(
                    id=f"{paired_prefix}_f{f:02d}_high",
                    allocation=paired_budgets[f] * high_s,
                ),
            ))
            substitutions.append(Substitution(
                id=f"{paired_prefix}_f{f:02d}_low",
                new_tendency=Tendency(
                    id=f"{paired_prefix}_f{f:02d}_low",
                    allocation=paired_budgets[f] * low_s,
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

    # Read out winning class for primary domain
    primary_allocs = {
        c: result.state.tendencies.get(f"{prefix}_class_{c}").allocation
        for c in profile.classes
    }
    primary_winner = max(primary_allocs, key=primary_allocs.get)

    # Optionally read out paired domain winner
    paired_winner = None
    if paired_profile is not None and paired_prefix is not None:
        paired_allocs = {
            c: result.state.tendencies.get(f"{paired_prefix}_class_{c}").allocation
            for c in paired_profile.classes
        }
        paired_winner = max(paired_allocs, key=paired_allocs.get)

    return primary_winner, paired_winner


# ---------------------------------------------------------------------------
# Main: run the three tests
# ---------------------------------------------------------------------------

def main(seed: int = 42):
    print()
    print("=" * 70)
    print("CROSS-DOMAIN SHARED-ENGINE TEST")
    print("=" * 70)

    # Generate two domains with shared signatures
    print("\n  Generating two synthetic domains with shared abstract structure...")
    X_a, y_a, centroids_a = synthesize_domain("A", seed=10)
    X_b, y_b, centroids_b = synthesize_domain("B", seed=20)
    print(f"    Domain A: 5 classes x 50 cases = {len(y_a)} total")
    print(f"    Domain B: 5 classes x 50 cases = {len(y_b)} total")
    print(f"    Each class has shared abstract signature: COMPACT, EXTENDED, ASYMMETRIC")
    print(f"    Domains differ in feature-space layout but share abstract structure.")

    # Train/test split per domain
    rng = np.random.RandomState(seed)
    def split(X, y, frac=0.7):
        n = len(y)
        idx = list(range(n))
        rng.shuffle(idx)
        sp = int(n * frac)
        return X[idx[:sp]], y[idx[:sp]], X[idx[sp:]], y[idx[sp:]]

    Xa_tr, ya_tr, Xa_te, ya_te = split(X_a, y_a)
    Xb_tr, yb_tr, Xb_te, yb_te = split(X_b, y_b)

    classes = sorted(set(y_a))   # 0..4
    profile_a = fit_profile(Xa_tr, ya_tr, classes, domain="A")
    profile_b = fit_profile(Xb_tr, yb_tr, classes, domain="B")

    # ---------- Test 1: Domain A solo (with bridges) ----------
    print("\n" + "-" * 70)
    print("Test 1: classify Domain A cases (A side + bridges only)")
    print("-" * 70)
    state_a = build_state([(profile_a, "A")], include_bridges=True)
    correct_a_solo = 0
    n_test = min(100, len(ya_te))
    for i in range(n_test):
        winner, _ = classify(Xa_te[i], profile_a, "A", state_a)
        if winner == ya_te[i]:
            correct_a_solo += 1
    acc_a_solo = correct_a_solo / n_test
    print(f"  accuracy: {correct_a_solo}/{n_test} = {acc_a_solo:.1%}")

    # ---------- Test 2: Domain B solo (with bridges) ----------
    print("\n" + "-" * 70)
    print("Test 2: classify Domain B cases (B side + bridges only)")
    print("-" * 70)
    state_b = build_state([(profile_b, "B")], include_bridges=True)
    correct_b_solo = 0
    for i in range(n_test):
        winner, _ = classify(Xb_te[i], profile_b, "B", state_b)
        if winner == yb_te[i]:
            correct_b_solo += 1
    acc_b_solo = correct_b_solo / n_test
    print(f"  accuracy: {correct_b_solo}/{n_test} = {acc_b_solo:.1%}")

    # ---------- Test 3: Cross-domain (A and B in one engine) ----------
    print("\n" + "-" * 70)
    print("Test 3: cross-domain (A and B in one engine, paired cases)")
    print("-" * 70)
    state_ab = build_state([(profile_a, "A"), (profile_b, "B")], include_bridges=True)

    correct_a_xd = 0
    correct_b_xd = 0
    for i in range(n_test):
        # Pair an A case with a B case at the same index
        a_winner, b_winner = classify(
            Xa_te[i], profile_a, "A", state_ab,
            paired_case=Xb_te[i], paired_profile=profile_b, paired_prefix="B",
        )
        if a_winner == ya_te[i]:
            correct_a_xd += 1
        if b_winner == yb_te[i]:
            correct_b_xd += 1

    acc_a_xd = correct_a_xd / n_test
    acc_b_xd = correct_b_xd / n_test
    print(f"  Domain A accuracy: {correct_a_xd}/{n_test} = {acc_a_xd:.1%}")
    print(f"  Domain B accuracy: {correct_b_xd}/{n_test} = {acc_b_xd:.1%}")

    # ---------- Test 4 (control): Cross-domain WITHOUT bridges ----------
    print("\n" + "-" * 70)
    print("Test 4 (control): cross-domain WITHOUT bridges")
    print("-" * 70)
    state_ab_nobridge = build_state(
        [(profile_a, "A"), (profile_b, "B")], include_bridges=False
    )
    correct_a_nb = 0
    correct_b_nb = 0
    for i in range(n_test):
        a_w, b_w = classify(
            Xa_te[i], profile_a, "A", state_ab_nobridge,
            paired_case=Xb_te[i], paired_profile=profile_b, paired_prefix="B",
        )
        if a_w == ya_te[i]:
            correct_a_nb += 1
        if b_w == yb_te[i]:
            correct_b_nb += 1
    acc_a_nb = correct_a_nb / n_test
    acc_b_nb = correct_b_nb / n_test
    print(f"  Domain A accuracy: {correct_a_nb}/{n_test} = {acc_a_nb:.1%}")
    print(f"  Domain B accuracy: {correct_b_nb}/{n_test} = {acc_b_nb:.1%}")

    # ---------- Summary ----------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Test               Domain A    Domain B")
    print(f"  Solo + bridges     {acc_a_solo:>7.1%}    {acc_b_solo:>7.1%}")
    print(f"  Cross-domain       {acc_a_xd:>7.1%}    {acc_b_xd:>7.1%}")
    print(f"  Cross, no bridge   {acc_a_nb:>7.1%}    {acc_b_nb:>7.1%}")
    print()

    # Interpretation
    contamination_a = acc_a_solo - acc_a_xd
    contamination_b = acc_b_solo - acc_b_xd
    bridge_lift_a = acc_a_xd - acc_a_nb
    bridge_lift_b = acc_b_xd - acc_b_nb

    print(f"  Cross-domain delta vs solo:  A: {contamination_a*100:+.1f} pts  "
          f"B: {contamination_b*100:+.1f} pts")
    print(f"  Bridge lift (vs no-bridge):  A: {bridge_lift_a*100:+.1f} pts  "
          f"B: {bridge_lift_b*100:+.1f} pts")

    print()
    if abs(contamination_a) < 0.05 and abs(contamination_b) < 0.05:
        print("  Cross-domain processing does not contaminate either side. The")
        print("  engine handles two domains in one substrate without interference.")
    elif contamination_a > 0.05 or contamination_b > 0.05:
        print("  Cross-domain processing CONTAMINATES at least one side. The")
        print("  engine is not isolating domain-specific evidence properly.")
    else:
        print("  Cross-domain processing actually IMPROVES one or both domains.")
        print("  The bridge tendencies are letting evidence flow productively.")

    if bridge_lift_a > 0.02 or bridge_lift_b > 0.02:
        print("  Bridges contribute positively: removing them measurably hurts.")
    elif bridge_lift_a < -0.02 or bridge_lift_b < -0.02:
        print("  Bridges contribute NEGATIVELY: removing them measurably helps.")
    else:
        print("  Bridges have negligible effect either way.")
    print()


if __name__ == "__main__":
    main()
