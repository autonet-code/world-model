"""Stage 3 — empirical check: does substrate alpha(t) trajectory match
a Lindblad evolution?

Setup:
  - One tendency, one root.
  - Drip-feed observations one at a time, equilibrating between each.
  - Record alpha (= net_score / something normalized to [0,1]) per step.
  - Fit a Lindblad model to the trajectory and report residuals.

We don't need to recover all internal substrate state — we just need
the score trajectory and the per-step observation type (PRO or CON)
to compute the predicted Lindblad dynamics.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Substrate imports
sys.path.insert(0, r"C:\code\autonet")
sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)

# Lindblad kernel
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lindblad.lindblad_kernel import (  # type: ignore
    RAISE_TO_PRO,
    LOWER_TO_CON,
    evolve,
    maximally_mixed,
    population_pro,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_world(dim: int = 2) -> World:
    """Single-root world with one tendency at origin, polarity along axis 0."""
    world = World()
    anchor = tuple([1.0] + [0.0] * (dim - 1))
    world.add_tendency(GeneralizedTendency(
        id="r",
        thesis="The thing.",
        anchor=anchor,
        polarity_axis=anchor,
        budget=1.0,
        bandwidth=0.5,
        smooth_promotion=False,  # turn off to keep dynamics simple
    ))
    return world


def alpha_from_world(world: World, root_id: str = "r") -> float:
    """Map substrate net_score to alpha in [0,1].

    net_score in the substrate is unbounded. We use a soft scaling:
    alpha = sigmoid(net_score) so that 0 stake -> 0.5, large positive -> ~1,
    large negative -> ~0. This is the natural mapping when treating
    net_score as a log-odds and alpha as a probability.
    """
    s = world.tendencies[root_id].tree.score
    return 1.0 / (1.0 + math.exp(-s))


def make_pro_observation(seq: int, dim: int = 2) -> Observation:
    """Observation aligned with the tendency's polarity (drives PRO)."""
    coords = tuple([1.0] + [0.0] * (dim - 1))
    return Observation(id=f"pro_{seq}", coords=coords, label=f"pro_{seq}")


def make_con_observation(seq: int, dim: int = 2) -> Observation:
    """Observation anti-aligned with the tendency's polarity (drives CON)."""
    coords = tuple([-1.0] + [0.0] * (dim - 1))
    return Observation(id=f"con_{seq}", coords=coords, label=f"con_{seq}")


# ---------------------------------------------------------------------------
# Substrate-side trace
# ---------------------------------------------------------------------------


def run_substrate_trace(
    obs_schedule: list[str],
    dim: int = 2,
    rounds_per_step: int = 4,
) -> list[dict]:
    """Drip-feed observations and record (step, alpha, n_pro_obs, n_con_obs)
    after equilibrate.
    """
    world = build_world(dim=dim)
    trace = []
    n_pro = 0
    n_con = 0
    # Initial state (no observations)
    trace.append({
        "step": 0,
        "alpha": alpha_from_world(world),
        "n_pro": 0,
        "n_con": 0,
        "obs_kind": None,
    })
    for i, kind in enumerate(obs_schedule, 1):
        if kind == "PRO":
            obs = make_pro_observation(i, dim=dim)
            n_pro += 1
        elif kind == "CON":
            obs = make_con_observation(i, dim=dim)
            n_con += 1
        else:
            raise ValueError(kind)
        world.add_observation(obs)
        equilibrate(world, max_rounds=rounds_per_step, tolerance=1e-4)
        trace.append({
            "step": i,
            "alpha": alpha_from_world(world),
            "n_pro": n_pro,
            "n_con": n_con,
            "obs_kind": kind,
        })
    return trace


# ---------------------------------------------------------------------------
# Lindblad-side trace
# ---------------------------------------------------------------------------


def run_lindblad_trace(
    obs_schedule: list[str],
    gamma_pro: float,
    gamma_con: float,
    dt_per_obs: float = 1.0,
    int_dt: float = 1e-3,
) -> list[dict]:
    """Apply the same observation schedule to a Lindblad-evolved single
    qubit. Each observation = run amplitude-damping channel toward PRO
    or CON for time dt_per_obs, with rate gamma_*.
    """
    H = np.zeros((2, 2), dtype=complex)
    rho = maximally_mixed()
    trace = [{"step": 0, "alpha": population_pro(rho), "obs_kind": None}]
    for i, kind in enumerate(obs_schedule, 1):
        if kind == "PRO":
            jump_ops = [(RAISE_TO_PRO, gamma_pro)]
        elif kind == "CON":
            jump_ops = [(LOWER_TO_CON, gamma_con)]
        else:
            raise ValueError(kind)
        rho, _ = evolve(rho, H, jump_ops=jump_ops, t_total=dt_per_obs, dt=int_dt)
        trace.append({
            "step": i,
            "alpha": population_pro(rho),
            "obs_kind": kind,
        })
    return trace


# ---------------------------------------------------------------------------
# Fit gamma to observed substrate trajectory
# ---------------------------------------------------------------------------


def lindblad_predict_alpha(
    obs_schedule: list[str],
    alpha_0: float,
    gamma_pro: float,
    gamma_con: float,
    dt_per_obs: float = 1.0,
) -> list[float]:
    """Closed-form alpha trajectory.

    For amplitude damping toward PRO with rate g over time t:
      alpha(t) = 1 - (1 - alpha_0) * exp(-g * t)
    For damping toward CON:
      alpha(t) = alpha_0 * exp(-g * t)
    """
    alphas = [alpha_0]
    a = alpha_0
    for kind in obs_schedule:
        if kind == "PRO":
            a = 1.0 - (1.0 - a) * math.exp(-gamma_pro * dt_per_obs)
        elif kind == "CON":
            a = a * math.exp(-gamma_con * dt_per_obs)
        alphas.append(a)
    return alphas


def fit_gammas(
    obs_schedule: list[str],
    observed_alphas: list[float],
    dt_per_obs: float = 1.0,
) -> Tuple[float, float, float]:
    """Grid-search gamma_pro, gamma_con minimizing MSE between observed
    alpha and Lindblad-predicted alpha. Returns (gamma_pro, gamma_con, mse).
    """
    best = (None, None, float("inf"))
    grid = np.geomspace(0.05, 5.0, 60)
    alpha_0 = observed_alphas[0]
    for gp in grid:
        for gc in grid:
            pred = lindblad_predict_alpha(obs_schedule, alpha_0, gp, gc, dt_per_obs)
            mse = sum((p - o) ** 2 for p, o in zip(pred, observed_alphas)) / len(observed_alphas)
            if mse < best[2]:
                best = (float(gp), float(gc), mse)
    return best


# ---------------------------------------------------------------------------
# Plot to file
# ---------------------------------------------------------------------------


def plot_traces(
    obs_schedule: list[str],
    sub_trace: list[dict],
    fitted_alphas: list[float],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping plot)")
        return
    steps = [r["step"] for r in sub_trace]
    sub_alphas = [r["alpha"] for r in sub_trace]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, sub_alphas, "o-", label="substrate alpha", linewidth=2)
    ax.plot(steps, fitted_alphas, "x--", label="lindblad fit", linewidth=1)
    ax.set_xlabel("step (observation #)")
    ax.set_ylabel("alpha = sigmoid(net_score)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("substrate alpha(t) vs Lindblad fit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Mark PRO/CON observations on the x-axis
    for i, kind in enumerate(obs_schedule, 1):
        color = "tab:green" if kind == "PRO" else "tab:red"
        ax.axvline(i, color=color, alpha=0.1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Schedule: drip 10 PRO, then 5 CON, then 5 PRO. Asymmetric so we can
    # see relaxation in both directions.
    schedule = ["PRO"] * 10 + ["CON"] * 5 + ["PRO"] * 5

    print("=" * 72)
    print("Stage 3: substrate alpha(t) vs Lindblad fit")
    print("=" * 72)
    print(f"observation schedule: {len(schedule)} steps")
    pro_count = schedule.count("PRO")
    con_count = schedule.count("CON")
    print(f"  PRO: {pro_count}, CON: {con_count}")
    print()

    print("running substrate trace…")
    sub_trace = run_substrate_trace(schedule, rounds_per_step=4)
    sub_alphas = [r["alpha"] for r in sub_trace]

    print(f"  alpha trajectory (first/middle/last):")
    print(f"    {sub_alphas[0]:.4f}, …, {sub_alphas[len(sub_alphas)//2]:.4f}, …, {sub_alphas[-1]:.4f}")
    print()

    print("fitting Lindblad gammas…")
    gp, gc, mse = fit_gammas(schedule, sub_alphas, dt_per_obs=1.0)
    print(f"  best fit: gamma_pro={gp:.4f}  gamma_con={gc:.4f}  mse={mse:.6f}")
    print(f"  rmse = {math.sqrt(mse):.4f} (typical residual per point)")
    print()

    fitted_alphas = lindblad_predict_alpha(schedule, sub_alphas[0], gp, gc, dt_per_obs=1.0)
    print(f"{'step':>4} {'kind':>4} {'sub_alpha':>10} {'fit_alpha':>10} {'residual':>10}")
    for r, f in zip(sub_trace, fitted_alphas):
        kind = r.get("obs_kind") or "-"
        print(f"{r['step']:>4} {kind:>4} {r['alpha']:>10.4f} {f:>10.4f} {(r['alpha']-f):>+10.4f}")

    plot_path = HERE / "stage3_alpha_trajectory.png"
    plot_traces(schedule, sub_trace, fitted_alphas, plot_path)

    print()
    out_json = HERE / "stage3_results.json"
    import json
    out_json.write_text(json.dumps({
        "schedule": schedule,
        "substrate_alpha": sub_alphas,
        "fitted_alpha": fitted_alphas,
        "gamma_pro": gp,
        "gamma_con": gc,
        "mse": mse,
        "rmse": math.sqrt(mse),
    }, indent=2), encoding="utf-8")
    print(f"  results saved to {out_json}")


if __name__ == "__main__":
    main()
