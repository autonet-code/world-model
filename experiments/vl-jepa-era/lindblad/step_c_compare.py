"""Step C — compare classical equilibrate vs equilibrate_continuous.

Runs the same observation schedule through both equilibration kernels
and reports per-root alpha trajectories side-by-side.

Three test scenarios:

  1. Single root, all-PRO observations. Classical produces linear-in-n
     alpha climb; continuous produces exponential-in-t saturation.
     Expected divergence: continuous saturates earlier.

  2. Single root, mixed PRO/CON observations creating tension. Classical
     has no representation for tension; continuous's transverse field
     resists committing to either pole.

  3. Two roots with shared coordinates (J coupling). Classical has the
     locality rule via cross-staking; continuous has J coupling.
     Compare how strongly observations on root a propagate to root b.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"C:\code\autonet")
sys.path.insert(0, r"C:\code\world-model")
sys.path.insert(0, str(HERE.parent))

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World, equilibrate,
)
from lindblad.equilibrate_continuous import (  # type: ignore
    alpha_from_substrate_score, equilibrate_continuous, extract_subclaims,
)


# ---------------------------------------------------------------------------
# World builders
# ---------------------------------------------------------------------------


def build_one_root_world(dim: int = 2) -> World:
    world = World()
    anchor = tuple([1.0] + [0.0] * (dim - 1))
    world.add_tendency(GeneralizedTendency(
        id="r",
        thesis="The thing.",
        anchor=anchor,
        polarity_axis=anchor,
        budget=1.0,
        bandwidth=0.5,
        smooth_promotion=True,  # capacity accumulation needed for Step D
    ))
    return world


def build_two_root_world(dim: int = 2) -> World:
    """Two roots whose anchors share the same coordinate region (so the
    cross-coupling J is non-trivial)."""
    world = World()
    a1 = tuple([1.0] + [0.0] * (dim - 1))
    a2 = tuple([0.9] + [0.0] * (dim - 1))   # nearby anchor
    world.add_tendency(GeneralizedTendency(
        id="a", thesis="root a.",
        anchor=a1, polarity_axis=a1,
        budget=1.0, bandwidth=0.5, smooth_promotion=True,
    ))
    world.add_tendency(GeneralizedTendency(
        id="b", thesis="root b.",
        anchor=a2, polarity_axis=a2,
        budget=1.0, bandwidth=0.5, smooth_promotion=True,
    ))
    return world


def make_obs(seq: int, kind: str, dim: int = 2, root_idx: int = 0) -> Observation:
    """PRO obs aligns with axis 0; CON obs anti-aligns. root_idx selects
    target dimension if you want disjoint observations.
    """
    sign = 1.0 if kind == "PRO" else -1.0
    coords = [0.0] * dim
    coords[root_idx] = sign
    return Observation(id=f"{kind.lower()}_{seq}_{root_idx}",
                       coords=tuple(coords), label=f"{kind}_{seq}")


# ---------------------------------------------------------------------------
# Schedule runner — classical
# ---------------------------------------------------------------------------


def run_classical_schedule(
    world_builder, schedule: List[str], rounds_per_step: int = 4,
) -> List[dict]:
    world = world_builder()
    trace = []
    for tid in sorted(world.tendencies.keys()):
        a = alpha_from_substrate_score(world.tendencies[tid].tree.score)
        trace.append({"step": 0, "kind": None, "root": tid, "alpha": a})
    for i, kind in enumerate(schedule, 1):
        obs = make_obs(i, kind)
        world.add_observation(obs)
        equilibrate(world, max_rounds=rounds_per_step, tolerance=1e-4)
        for tid in sorted(world.tendencies.keys()):
            a = alpha_from_substrate_score(world.tendencies[tid].tree.score)
            trace.append({"step": i, "kind": kind, "root": tid, "alpha": a})
    return trace


# ---------------------------------------------------------------------------
# Schedule runner — continuous
# ---------------------------------------------------------------------------


def run_continuous_schedule(
    world_builder, schedule: List[str], dt_per_obs: float = 1.0,
) -> List[dict]:
    """Drip observations one at a time; between each pair we evolve for
    dt_per_obs of simulated time. The substrate's score trees grow
    discretely (sub-claims sprout via the substrate's own act-pass)
    but per-root alpha is governed by Lindblad evolution.

    To mirror the discrete substrate's structure, we do:
      1. Add observation to world.
      2. Run one round of substrate.act + apply_stakes so sub-claim
         children sprout (the GRAPH STRUCTURE update).
      3. Run equilibrate_continuous for dt_per_obs (the SCORE update
         under the new structure).
      4. Clear observations so they aren't double-counted.
    """
    world = world_builder()
    trace = []
    for tid in sorted(world.tendencies.keys()):
        a = alpha_from_substrate_score(world.tendencies[tid].tree.score)
        trace.append({"step": 0, "kind": None, "root": tid, "alpha": a})
    for i, kind in enumerate(schedule, 1):
        obs = make_obs(i, kind)
        world.add_observation(obs)
        # Step 1-2: let the substrate sprout sub-claim children for this obs.
        # We use one round of act + apply_stakes (NOT a full equilibrate)
        # so the graph structure updates without the discrete score
        # accumulation overwhelming our continuous evolution.
        for tendency in world.tendencies.values():
            tendency.act(world)
        world.apply_stakes()
        # Step 3: continuous evolution.
        equilibrate_continuous(
            world, t_total=dt_per_obs, dt=1e-3,
            bandwidth=0.5, kappa=1.0, lam=1.0, mu=1.0, base_gamma=0.5,
            write_back=True,
            mode="stepD", W_scale=0.05,  # match substrate's typical capacity scale
        )
        for tid in sorted(world.tendencies.keys()):
            a = alpha_from_substrate_score(world.tendencies[tid].tree.score)
            trace.append({"step": i, "kind": kind, "root": tid, "alpha": a})
        # Don't clear observations -- the substrate keeps them so cross-tendency
        # acts work. But the jump operators only fire for ones we haven't
        # yet incorporated... actually for simplicity, we clear after each step.
        world.clear_observations()
    return trace


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def compare_traces(scenario_name: str, classical: List[dict],
                   continuous: List[dict]) -> dict:
    print("=" * 72)
    print(f"Scenario: {scenario_name}")
    print("=" * 72)

    # Group by root id
    roots = sorted(set(r["root"] for r in classical))

    out = {"scenario": scenario_name, "by_root": {}}
    for root in roots:
        c_alphas = [r["alpha"] for r in classical if r["root"] == root]
        q_alphas = [r["alpha"] for r in continuous if r["root"] == root]
        kinds = [r["kind"] for r in classical if r["root"] == root]
        n = min(len(c_alphas), len(q_alphas))

        print(f"\nRoot '{root}':")
        print(f"  step  kind     classical    continuous  diff")
        for i in range(n):
            kind = kinds[i] or "-"
            diff = q_alphas[i] - c_alphas[i]
            print(f"  {i:>4}  {kind:>4}  {c_alphas[i]:>10.4f}  {q_alphas[i]:>10.4f}  {diff:+.4f}")

        out["by_root"][root] = {
            "classical_alpha": c_alphas[:n],
            "continuous_alpha": q_alphas[:n],
            "kinds": [k for k in kinds[:n]],
        }
    return out


# ---------------------------------------------------------------------------
# Three scenarios
# ---------------------------------------------------------------------------


def scenario_1_pure_pro():
    schedule = ["PRO"] * 10
    classical = run_classical_schedule(build_one_root_world, schedule)
    continuous = run_continuous_schedule(build_one_root_world, schedule)
    return compare_traces("S1: 10 PRO observations on single root", classical, continuous)


def scenario_2_mixed_tension():
    # Alternate PRO and CON to maintain tension
    schedule = ["PRO", "CON"] * 5
    classical = run_classical_schedule(build_one_root_world, schedule)
    continuous = run_continuous_schedule(build_one_root_world, schedule)
    return compare_traces("S2: alternating PRO/CON on single root", classical, continuous)


def scenario_3_two_roots_shared():
    schedule = ["PRO"] * 5 + ["CON"] * 5
    classical = run_classical_schedule(build_two_root_world, schedule)
    continuous = run_continuous_schedule(build_two_root_world, schedule)
    return compare_traces("S3: 5 PRO then 5 CON on shared-coordinate roots",
                          classical, continuous)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_comparison(results: list, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plot)")
        return
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.5 * len(results)))
    if len(results) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, results):
        for root, data in scenario["by_root"].items():
            steps = list(range(len(data["classical_alpha"])))
            ax.plot(steps, data["classical_alpha"], "o-",
                    label=f"{root} classical", alpha=0.7)
            ax.plot(steps, data["continuous_alpha"], "x--",
                    label=f"{root} continuous", alpha=0.9)
        ax.set_title(scenario["scenario"])
        ax.set_xlabel("step")
        ax.set_ylabel("alpha")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nplot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Step C — classical equilibrate vs equilibrate_continuous\n")
    results = [
        scenario_1_pure_pro(),
        scenario_2_mixed_tension(),
        scenario_3_two_roots_shared(),
    ]
    out_json = HERE / "step_c_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {out_json}")
    plot_comparison(results, HERE / "step_c_comparison.png")


if __name__ == "__main__":
    main()
