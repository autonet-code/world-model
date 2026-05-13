"""Phase 1: re-run the substrate-to-Lindblad bridge with rho_0 built
from (score, n) jointly, instead of score alone.

Three arms per scenario:
  A. Classical equilibrate (the substrate's own discrete update).
  B. Continuous bridge with rho_0 diagonal-only (Step C/D behavior).
  C. Continuous bridge with rho_0 = (alpha from score, c from node.n).

The Phase 1 hypothesis: arm C should track classical (arm A) more
faithfully than arm B did, because arm B was throwing away half the
substrate's state by setting c=0 always.

We don't claim C will perfectly match A — the substrate's discrete
update rule and the Lindblad ODE are different processes. But if the
bridge is going to work at all, having the right initial coherence
is necessary.
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
    alpha_from_substrate_score, equilibrate_continuous,
    extract_subclaims, _root_coherence_from_n,
)


def build_one_root_world() -> World:
    world = World()
    world.add_tendency(GeneralizedTendency(
        id="r",
        thesis="The thing.",
        anchor=(1.0, 0.0),
        polarity_axis=(1.0, 0.0),
        budget=1.0,
        bandwidth=0.5,
        smooth_promotion=True,
    ))
    return world


def make_obs(seq: int, kind: str) -> Observation:
    sign = 1.0 if kind == "PRO" else -1.0
    return Observation(id=f"{kind.lower()}_{seq}", coords=(sign, 0.0),
                       label=f"{kind}_{seq}")


# ---------------------------------------------------------------------------
# Schedule runners — three arms
# ---------------------------------------------------------------------------


def run_classical(world_builder, schedule: List[str]) -> List[float]:
    """Arm A: substrate's own discrete equilibrate."""
    world = world_builder()
    out = [alpha_from_substrate_score(world.tendencies["r"].tree.score)]
    for i, kind in enumerate(schedule, 1):
        world.add_observation(make_obs(i, kind))
        equilibrate(world, max_rounds=4, tolerance=1e-4)
        out.append(alpha_from_substrate_score(world.tendencies["r"].tree.score))
    return out


def run_continuous(
    world_builder, schedule: List[str], use_novelty: bool,
) -> List[float]:
    """Arm B (use_novelty=False) or Arm C (use_novelty=True).

    Each step:
      1. Add observation.
      2. Run substrate's act+apply_stakes (lets sub-claims sprout).
      3. Update novelty (the substrate's per-round update).
      4. Run continuous equilibration with rho_0 derived per use_novelty flag.
    """
    world = world_builder()
    out = [alpha_from_substrate_score(world.tendencies["r"].tree.score)]
    for i, kind in enumerate(schedule, 1):
        world.add_observation(make_obs(i, kind))
        # Let the substrate sprout sub-claims for this obs.
        for tendency in world.tendencies.values():
            tendency.act(world)
        world.apply_stakes()
        # Update n (this is the new continuous-novelty machinery).
        for tendency in world.tendencies.values():
            tendency.update_novelty(dt=1.0)
        # Now run the bridge with the chosen rho_0 form.
        equilibrate_continuous(
            world, t_total=1.0, dt=1e-3,
            bandwidth=0.5, kappa=1.0, lam=1.0, mu=1.0, base_gamma=0.5,
            write_back=True,
            mode="stepD", W_scale=0.05,
            use_novelty_in_rho0=use_novelty,
        )
        out.append(alpha_from_substrate_score(world.tendencies["r"].tree.score))
        world.clear_observations()
    return out


# ---------------------------------------------------------------------------
# Diagnostic: report per-step n_root for the novelty arm
# ---------------------------------------------------------------------------


def run_continuous_with_n_trace(
    world_builder, schedule: List[str],
) -> tuple[List[float], List[float]]:
    """Like run_continuous(use_novelty=True) but also records per-step
    n_root (avg n across direct sub-claims). Useful for understanding
    why arm C diverges or converges with arm A.
    """
    world = world_builder()
    alphas = [alpha_from_substrate_score(world.tendencies["r"].tree.score)]
    n_root_trace = [0.0]  # no sub-claims yet at step 0
    for i, kind in enumerate(schedule, 1):
        world.add_observation(make_obs(i, kind))
        for tendency in world.tendencies.values():
            tendency.act(world)
        world.apply_stakes()
        for tendency in world.tendencies.values():
            tendency.update_novelty(dt=1.0)
        n_root = _root_coherence_from_n(world, "r")
        equilibrate_continuous(
            world, t_total=1.0, dt=1e-3,
            bandwidth=0.5, kappa=1.0, lam=1.0, mu=1.0, base_gamma=0.5,
            write_back=True,
            mode="stepD", W_scale=0.05,
            use_novelty_in_rho0=True,
        )
        alphas.append(alpha_from_substrate_score(world.tendencies["r"].tree.score))
        n_root_trace.append(n_root)
        world.clear_observations()
    return alphas, n_root_trace


# ---------------------------------------------------------------------------
# Compare three arms
# ---------------------------------------------------------------------------


def compare_three_arms(scenario_name: str, schedule: List[str]) -> dict:
    print("=" * 76)
    print(f"Scenario: {scenario_name}")
    print("=" * 76)

    a_classical = run_classical(build_one_root_world, schedule)
    b_no_novelty = run_continuous(build_one_root_world, schedule, use_novelty=False)
    c_with_novelty, c_n_root = run_continuous_with_n_trace(build_one_root_world, schedule)

    # Trim to common length
    n = min(len(a_classical), len(b_no_novelty), len(c_with_novelty))

    print(f"{'step':>4} {'kind':>4}  "
          f"{'A=classical':>12} {'B=no novelty':>13} {'C=with novelty':>16}  "
          f"{'B-A':>7} {'C-A':>7}  {'n_root':>7}")
    kinds = [None] + schedule
    for i in range(n):
        ka = kinds[i] if i < len(kinds) else "-"
        kind = ka or "-"
        print(f"{i:>4} {kind:>4}  "
              f"{a_classical[i]:>12.4f} {b_no_novelty[i]:>13.4f} {c_with_novelty[i]:>16.4f}  "
              f"{b_no_novelty[i] - a_classical[i]:>+7.3f} "
              f"{c_with_novelty[i] - a_classical[i]:>+7.3f}  "
              f"{c_n_root[i]:>7.3f}")

    # Compute residuals: how much closer is C to A than B is?
    rmse_b = math.sqrt(sum((b_no_novelty[i] - a_classical[i]) ** 2 for i in range(n)) / n)
    rmse_c = math.sqrt(sum((c_with_novelty[i] - a_classical[i]) ** 2 for i in range(n)) / n)
    improvement = rmse_b - rmse_c
    print()
    print(f"RMSE vs classical (A):")
    print(f"  arm B (no novelty in rho_0):   {rmse_b:.4f}")
    print(f"  arm C (with novelty in rho_0): {rmse_c:.4f}")
    print(f"  improvement (B - C):           {improvement:+.4f}")
    if improvement > 0.01:
        print("  -> C is closer to classical (novelty helps the bridge)")
    elif improvement < -0.01:
        print("  -> C is further from classical (novelty hurts the bridge)")
    else:
        print("  -> no meaningful difference")
    return {
        "scenario": scenario_name,
        "schedule": schedule,
        "classical": a_classical,
        "no_novelty": b_no_novelty,
        "with_novelty": c_with_novelty,
        "n_root_trace": c_n_root,
        "rmse_no_novelty": rmse_b,
        "rmse_with_novelty": rmse_c,
        "improvement": improvement,
    }


# ---------------------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------------------


def plot_three_arms(results: list, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plot)")
        return
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]
    for ax_pair, scenario in zip(axes, results):
        ax_alpha, ax_n = ax_pair
        steps = list(range(len(scenario["classical"])))
        ax_alpha.plot(steps, scenario["classical"], "o-", label="A: classical", alpha=0.85, linewidth=2)
        ax_alpha.plot(steps, scenario["no_novelty"], "s--", label="B: no novelty", alpha=0.7)
        ax_alpha.plot(steps, scenario["with_novelty"], "x:", label="C: with novelty", alpha=0.85)
        ax_alpha.set_title(scenario["scenario"])
        ax_alpha.set_xlabel("step")
        ax_alpha.set_ylabel("alpha")
        ax_alpha.set_ylim(-0.05, 1.05)
        ax_alpha.grid(True, alpha=0.3)
        ax_alpha.legend(loc="best", fontsize=8)
        ax_n.plot(steps, scenario["n_root_trace"], ".-", color="tab:purple")
        ax_n.set_title("n_root (avg n across sub-claims)")
        ax_n.set_xlabel("step")
        ax_n.set_ylabel("n_root")
        ax_n.set_ylim(-0.05, 1.05)
        ax_n.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nplot saved to {out_path}")


# ---------------------------------------------------------------------------
# Three scenarios (mirror Step C)
# ---------------------------------------------------------------------------


def main():
    print("Phase 1: bridge with novelty-as-coherence in rho_0\n")
    scenarios = [
        ("S1: 10 PRO observations", ["PRO"] * 10),
        ("S2: alternating PRO/CON (5+5)", ["PRO", "CON"] * 5),
        ("S3: 5 PRO then 5 CON", ["PRO"] * 5 + ["CON"] * 5),
    ]
    results = [compare_three_arms(name, schedule) for name, schedule in scenarios]
    out_json = HERE / "phase_1_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {out_json}")
    plot_three_arms(results, HERE / "phase_1_comparison.png")

    # Summary
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    for r in results:
        print(f"  {r['scenario']:35s}  "
              f"RMSE B={r['rmse_no_novelty']:.4f}  "
              f"RMSE C={r['rmse_with_novelty']:.4f}  "
              f"improvement={r['improvement']:+.4f}")


if __name__ == "__main__":
    main()
