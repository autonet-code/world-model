#!/usr/bin/env python3
"""Tier 2: resist-then-yield-decisively at consensus scale.

Sweeps N x split x temporal x {discrete, continuous} kernel,
measures settle time + final alpha + decisiveness gain + RMSE
between kernels, checks 6 predictions.

No LLM. Pure substrate dynamics.

See TIER2_SPEC.md for the predictions and rationale.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, r"C:\code\world-model")

from world_model.generalized import (  # type: ignore
    GeneralizedTendency, Observation, World,
)
from world_model.generalized.equilibrate import (  # type: ignore
    equilibrate_continuous,
)
from world_model.models.tree import Position  # type: ignore


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "tier2_results.json"
STATUS_PATH = HERE / "tier2_status.json"
PLOTS_DIR = HERE / "tier2_plots"


# ---------------------------------------------------------------------------
# Substrate setup
# ---------------------------------------------------------------------------


def build_world() -> Tuple[World, str, str]:
    """Single tendency, single sub-claim. Returns (world, root_id, sub_claim_id)."""
    world = World()
    correctness = GeneralizedTendency(
        id="correctness",
        thesis="The contested claim.",
        anchor=(1.0,),
        polarity_axis=(1.0,),
        bandwidth=0.7,
        veto_shaped=False,
        novelty_gamma_pro=1.0,
        novelty_gamma_con=1.0,
        novelty_drift=0.01,
    )
    world.add_tendency(correctness)
    root_id = correctness.tree.root_node.id
    # Pre-sprout the contested sub-claim at coords (1,)
    sub = correctness.sprout_child(
        parent_node_id=root_id,
        position=Position.PRO,
        anchor=(1.0,),
        polarity_axis=(1.0,),
        content="contested_subclaim",
    )
    return world, root_id, sub.id


# ---------------------------------------------------------------------------
# Schedule generators
# ---------------------------------------------------------------------------


def schedule_pulse(N_pro: int, N_con: int, T: int) -> Dict[int, Tuple[int, int]]:
    """All agents post in round 1; quiet rounds 2..T."""
    out = {}
    for r in range(1, T + 1):
        out[r] = (N_pro, 0) if r == 1 else (0, 0)
        # but we need both PRO and CON in round 1
    out[1] = (N_pro, N_con)
    return out


def schedule_drip(N_pro: int, N_con: int, T: int) -> Dict[int, Tuple[int, int]]:
    """Spread the N_pro + N_con posts evenly across rounds 1..T."""
    total = N_pro + N_con
    out = {r: (0, 0) for r in range(1, T + 1)}
    if total == 0:
        return out
    pro_per = N_pro / T
    con_per = N_con / T
    pro_carry = 0.0
    con_carry = 0.0
    for r in range(1, T + 1):
        pro_carry += pro_per
        con_carry += con_per
        p = int(pro_carry)
        c = int(con_carry)
        pro_carry -= p
        con_carry -= c
        out[r] = (p, c)
    # Make sure totals match
    actual_pro = sum(p for p, _ in out.values())
    actual_con = sum(c for _, c in out.values())
    out[T] = (out[T][0] + (N_pro - actual_pro), out[T][1] + (N_con - actual_con))
    return out


def schedule_async(N_pro: int, N_con: int, T: int, seed: int = 42) -> Dict[int, Tuple[int, int]]:
    """Each agent posts in a uniformly-random round in [1, T]."""
    rng = random.Random(seed)
    out = {r: [0, 0] for r in range(1, T + 1)}
    for _ in range(N_pro):
        r = rng.randint(1, T)
        out[r][0] += 1
    for _ in range(N_con):
        r = rng.randint(1, T)
        out[r][1] += 1
    return {r: tuple(v) for r, v in out.items()}


SCHEDULES = {"pulse": schedule_pulse, "drip": schedule_drip, "async": schedule_async}


# ---------------------------------------------------------------------------
# Discrete kernel run
# ---------------------------------------------------------------------------


def make_obs(round_idx: int, agent_idx: int, sign: int) -> Observation:
    """An observation aimed at coords (sign,) — sign = +1 (PRO) or -1 (CON)."""
    return Observation(
        id=f"r{round_idx}_a{agent_idx}_{'p' if sign > 0 else 'c'}",
        coords=(float(sign),),
        label=f"r{round_idx}/a{agent_idx}/{'PRO' if sign > 0 else 'CON'}",
    )


def _direct_post_round(world: World, sub_id: str, root_id: str,
                       n_pro: int, n_con: int, agent_offset: int) -> None:
    """Each agent posts directly on the contested sub-claim with a
    unique agent_id. PRO agents post on the sub-claim; CON agents
    post on a CON sibling under the root (sprouting it lazily).

    This bypasses tendency.act() because that path collapses N
    same-coord observations into a single post per tendency. For the
    consensus experiment we want each agent's vote to count
    independently.
    """
    correctness = world.tendencies["correctness"]
    sub = correctness.tree.get_node(sub_id)
    if sub is None:
        return
    # CON agents need a CON sibling to post on so their evidence
    # actually lowers the sub-claim's score. Sprout it lazily once.
    con_sib_id = None
    for child in sub.con_children:
        con_sib_id = child.id
        break
    if con_sib_id is None and n_con > 0:
        con_node = correctness.sprout_child(
            parent_node_id=sub_id,
            position=Position.CON,
            anchor=(-1.0,),
            polarity_axis=(-1.0,),
            content="con_sibling",
        )
        con_sib_id = con_node.id
    for i in range(n_pro):
        sub.add_post(agent_id=f"agent_pro_{agent_offset + i}")
    if con_sib_id is not None:
        con_node = correctness.tree.get_node(con_sib_id)
        for i in range(n_con):
            con_node.add_post(agent_id=f"agent_con_{agent_offset + n_pro + i}")


def _wipe_round_contributions(world: World, sub_id: str) -> None:
    """Posts accumulate on the sub-claim across rounds. We don't
    wipe per-round (unlike apply_stakes) because the consensus
    experiment wants total-evidence to drive the verdict. n_val and
    score reflect the running cumulative state.
    """
    pass


def _snapshot(world: World, root_id: str, sub_id: str, r: int,
              n_pro: int, n_con: int) -> Dict:
    """Per-round snapshot. We read the ROOT's score because the
    continuous kernel evolves root populations via Lindblad
    (writes back to the root's stakes, not to sub-claims). The
    sub-claim's score, n, and post count are tracked for context.
    """
    correctness = world.tendencies["correctness"]
    root = correctness.tree.root_node
    sub = correctness.tree.get_node(sub_id)
    if sub is None:
        sub_score = 0.0
        sub_n = 0.0
        sub_posts = 0
    else:
        sub_score = sub.net_score
        sub_n = sub.n
        sub_posts = len(sub.stakes)
    root_score = root.net_score
    alpha = _alpha_from_score(root_score)
    return {
        "round": r,
        "n_pro": n_pro,
        "n_con": n_con,
        "n_val": sub_n,
        "net_score": root_score,
        "sub_score": sub_score,
        "alpha": alpha,
        "posts": sub_posts,
    }


def run_discrete(schedule: Dict[int, Tuple[int, int]], T: int) -> List[Dict]:
    """Direct-post discrete kernel: each round, agents post directly
    on the contested sub-claim (PRO) or its CON sibling. Posts
    accumulate across rounds. update_novelty runs each round to
    evolve n_val. Alpha is read from the ROOT's score (where the
    continuous Lindblad kernel writes back).
    """
    world, root_id, sub_id = build_world()
    correctness = world.tendencies["correctness"]
    trace: List[Dict] = []
    agent_counter = 0
    for r in range(1, T + 1):
        n_pro, n_con = schedule.get(r, (0, 0))
        if n_pro + n_con > 0:
            _direct_post_round(world, sub_id, root_id, n_pro, n_con, agent_counter)
            agent_counter += n_pro + n_con
        for tendency in world.tendencies.values():
            tendency.update_novelty(dt=1.0)
        trace.append(_snapshot(world, root_id, sub_id, r, n_pro, n_con))
    return trace


# ---------------------------------------------------------------------------
# Continuous kernel run (Lindblad)
# ---------------------------------------------------------------------------


def run_continuous(schedule: Dict[int, Tuple[int, int]], T: int) -> List[Dict]:
    """Direct-post continuous kernel: each round, agents post
    directly on the contested sub-claim AND add per-agent
    observations to the world so the Lindblad kernel can build
    jump operators from them. This drives the Lindblad evolution
    differently from a pure Hamiltonian-only run; jump operators
    push population toward PRO or CON depending on the agent's
    stance.
    """
    world, root_id, sub_id = build_world()
    correctness = world.tendencies["correctness"]
    trace: List[Dict] = []
    agent_counter = 0
    for r in range(1, T + 1):
        n_pro, n_con = schedule.get(r, (0, 0))
        if n_pro + n_con > 0:
            _direct_post_round(world, sub_id, root_id, n_pro, n_con, agent_counter)
            # Also feed observations to the world so Lindblad jump
            # operators see this round's evidence. PRO obs at coords
            # (1,) give RAISE_TO_PRO operator; CON obs at coords
            # (-1,) give LOWER_TO_CON.
            for i in range(n_pro):
                world.add_observation(make_obs(r, agent_counter + i, +1))
            for i in range(n_con):
                world.add_observation(make_obs(r, agent_counter + n_pro + i, -1))
            agent_counter += n_pro + n_con
        try:
            equilibrate_continuous(
                world, t_total=1.0, dt=0.05,
                bandwidth=0.7, base_gamma=0.5,
                use_novelty_in_rho0=True, write_back=True,
            )
        except Exception:
            pass
        world.clear_observations()
        for tendency in world.tendencies.values():
            tendency.update_novelty(dt=1.0)
        trace.append(_snapshot(world, root_id, sub_id, r, n_pro, n_con))
    return trace


def _alpha_from_score(score: float) -> float:
    if score > 30:
        return 1.0
    if score < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def settle_time(trace: List[Dict], window: int = 5,
                threshold: float = 0.001) -> int:
    """First round where alpha is stable for `window` consecutive rounds
    (max abs delta within window < threshold). Returns T_total + 1 if
    never settled.
    """
    alphas = [t["alpha"] for t in trace]
    for i in range(len(alphas) - window + 1):
        deltas = [abs(alphas[i+j+1] - alphas[i+j]) for j in range(window - 1)]
        if all(d < threshold for d in deltas):
            return trace[i]["round"]
    return len(trace) + 1


def final_alpha(trace: List[Dict]) -> float:
    return trace[-1]["alpha"] if trace else 0.5


def transient_max_amplitude(trace: List[Dict]) -> float:
    fa = final_alpha(trace)
    return max(abs(t["alpha"] - fa) for t in trace) if trace else 0.0


def decisiveness_gain(trace: List[Dict], n_pro: int, n_con: int) -> float:
    """gain = |final_alpha - 0.5| / |input_ratio - 0.5|, where
    input_ratio = n_pro / (n_pro + n_con). Returns gain or 0 if input
    is exactly 50/50.
    """
    total = n_pro + n_con
    if total == 0:
        return 0.0
    input_ratio = n_pro / total
    input_gap = abs(input_ratio - 0.5)
    if input_gap < 1e-6:
        return 0.0  # 50/50; no tilt to amplify
    output_gap = abs(final_alpha(trace) - 0.5)
    return output_gap / input_gap


def rmse_traces(trace_a: List[Dict], trace_b: List[Dict]) -> float:
    if not trace_a or not trace_b:
        return 0.0
    n = min(len(trace_a), len(trace_b))
    return math.sqrt(sum((trace_a[i]["alpha"] - trace_b[i]["alpha"])**2
                         for i in range(n)) / n)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


T_TOTAL = 100

CONFIGS = []
for n_total in [10, 100, 1000]:
    for split_label, ratio in [("50-50", 0.5), ("60-40", 0.6), ("80-20", 0.8)]:
        n_pro = int(n_total * ratio)
        n_con = n_total - n_pro
        for temporal in ["pulse", "drip", "async"]:
            CONFIGS.append({
                "N": n_total, "split": split_label,
                "n_pro": n_pro, "n_con": n_con,
                "temporal": temporal,
            })


def write_status(status: dict) -> None:
    status["last_update"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_one_config(cfg: Dict, status: Dict) -> Dict:
    """Run both kernels on one configuration; collect metrics."""
    n_pro, n_con = cfg["n_pro"], cfg["n_con"]
    temporal = cfg["temporal"]
    if temporal == "async":
        schedule = SCHEDULES["async"](n_pro, n_con, T_TOTAL, seed=42)
    else:
        schedule = SCHEDULES[temporal](n_pro, n_con, T_TOTAL)

    status["sub_phase"] = "discrete"
    write_status(status)
    started = time.time()
    trace_d = run_discrete(schedule, T_TOTAL)
    elapsed_d = time.time() - started

    status["sub_phase"] = "continuous"
    write_status(status)
    started = time.time()
    trace_c = run_continuous(schedule, T_TOTAL)
    elapsed_c = time.time() - started

    return {
        "config": cfg,
        "schedule_summary": {
            "first_round": schedule.get(1, (0, 0)),
            "total_pro_scheduled": sum(p for p, _ in schedule.values()),
            "total_con_scheduled": sum(c for _, c in schedule.values()),
        },
        "discrete": {
            "trace": trace_d,
            "elapsed_s": elapsed_d,
            "settle_time": settle_time(trace_d),
            "final_alpha": final_alpha(trace_d),
            "transient_max": transient_max_amplitude(trace_d),
            "decisiveness_gain": decisiveness_gain(trace_d, n_pro, n_con),
        },
        "continuous": {
            "trace": trace_c,
            "elapsed_s": elapsed_c,
            "settle_time": settle_time(trace_c),
            "final_alpha": final_alpha(trace_c),
            "transient_max": transient_max_amplitude(trace_c),
            "decisiveness_gain": decisiveness_gain(trace_c, n_pro, n_con),
        },
        "rmse_continuous_vs_discrete": rmse_traces(trace_c, trace_d),
    }


def evaluate_predictions(results: List[Dict]) -> Tuple[int, int, List[Dict]]:
    pass_count = 0
    fail_count = 0
    predictions: List[Dict] = []

    def check(label: str, condition: bool, detail: str) -> None:
        nonlocal pass_count, fail_count
        ok = "PASS" if condition else "FAIL"
        if condition:
            pass_count += 1
        else:
            fail_count += 1
        print(f"  [{ok}] {label}: {detail}")
        predictions.append({"label": label, "condition": bool(condition), "detail": detail})

    # C1: every config settles within 80 rounds (continuous kernel).
    c1_failures = [r for r in results
                   if r["continuous"]["settle_time"] >= 80]
    check("C1 (continuous settles within 80 rounds)",
          len(c1_failures) == 0,
          (f"all {len(results)} configs settled" if not c1_failures
           else f"{len(c1_failures)} configs did not settle: " +
           ", ".join(f"{r['config']['N']}/{r['config']['split']}/{r['config']['temporal']}"
                     f"@{r['continuous']['settle_time']}"
                     for r in c1_failures[:3])))

    # C2: settle time scales sub-linearly in N for fixed split/temporal.
    # Compare N=1000 to N=10 (drip+50-50 as canonical).
    canonical_N10 = next((r for r in results
                          if r["config"]["N"] == 10
                          and r["config"]["split"] == "50-50"
                          and r["config"]["temporal"] == "drip"), None)
    canonical_N1000 = next((r for r in results
                            if r["config"]["N"] == 1000
                            and r["config"]["split"] == "50-50"
                            and r["config"]["temporal"] == "drip"), None)
    if canonical_N10 and canonical_N1000:
        s10 = canonical_N10["continuous"]["settle_time"]
        s1k = canonical_N1000["continuous"]["settle_time"]
        c2_ok = s1k < 10 * s10 if s10 > 0 else False
        check("C2 (settle time sub-linear in N)", c2_ok,
              f"settle@N=10 {s10}, settle@N=1000 {s1k}, "
              f"ratio {s1k/max(s10,1):.2f} (expected < 10)")
    else:
        check("C2 (settle time sub-linear in N)", False,
              "missing canonical N=10 or N=1000 50/50 drip configs")

    # C3: final alpha tracks input ratio (no flips). For each config,
    # final_alpha is on the same side of 0.5 as input_ratio.
    c3_failures = []
    for r in results:
        n_pro = r["config"]["n_pro"]
        n_con = r["config"]["n_con"]
        if n_pro + n_con == 0:
            continue
        input_ratio = n_pro / (n_pro + n_con)
        fa = r["continuous"]["final_alpha"]
        if abs(input_ratio - 0.5) < 1e-6:
            continue  # 50/50 has no defined "side"
        if (input_ratio > 0.5) != (fa > 0.5):
            c3_failures.append(
                f"{r['config']['N']}/{r['config']['split']}/{r['config']['temporal']}"
                f": input {input_ratio:.2f} -> final α {fa:.3f}")
    check("C3 (final alpha tracks input ratio)", len(c3_failures) == 0,
          (f"all non-50/50 configs tracked" if not c3_failures
           else f"{len(c3_failures)} flips: {c3_failures[:3]}"))

    # C4: decisiveness gain >= 1.5 on 60/40.
    sixty_forty = [r for r in results if r["config"]["split"] == "60-40"]
    gains = [r["continuous"]["decisiveness_gain"] for r in sixty_forty]
    c4_ok = all(g >= 1.5 for g in gains) if gains else False
    check("C4 (60/40 decisiveness gain >= 1.5)", c4_ok,
          (f"min gain {min(gains):.2f}, max {max(gains):.2f}" if gains
           else "no 60/40 configs"))

    # C5: 50/50 lands in [0.4, 0.6].
    fifty = [r for r in results if r["config"]["split"] == "50-50"]
    c5_failures = [r for r in fifty
                   if not (0.4 <= r["continuous"]["final_alpha"] <= 0.6)]
    check("C5 (50/50 lands near 0.5)", len(c5_failures) == 0,
          (f"all {len(fifty)} 50/50 configs landed in [0.4, 0.6]"
           if not c5_failures
           else f"{len(c5_failures)} outside band: " +
           ", ".join(f"{r['config']['N']}/{r['config']['temporal']}@{r['continuous']['final_alpha']:.3f}"
                     for r in c5_failures[:3])))

    # C6: continuous kernel diverges from discrete on tilted cases.
    tilted = [r for r in results
              if r["config"]["split"] in ("60-40", "80-20")]
    rmses = [r["rmse_continuous_vs_discrete"] for r in tilted]
    c6_ok = all(r > 0.05 for r in rmses) if rmses else False
    check("C6 (continuous != discrete on tilted)", c6_ok,
          (f"all tilted RMSE > 0.05; min {min(rmses):.3f}, max {max(rmses):.3f}"
           if rmses
           else "no tilted configs"))

    return pass_count, fail_count, predictions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_per_config(results: List[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable; skipping plots)")
        return
    PLOTS_DIR.mkdir(exist_ok=True)
    # Group by (N, split): one plot per group, three temporal lines × two kernels.
    groups: Dict[Tuple[int, str], List[Dict]] = {}
    for r in results:
        key = (r["config"]["N"], r["config"]["split"])
        groups.setdefault(key, []).append(r)
    for (N, split), rs in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, kernel in zip(axes, ["discrete", "continuous"]):
            for r in rs:
                trace = r[kernel]["trace"]
                xs = [t["round"] for t in trace]
                ys = [t["alpha"] for t in trace]
                ax.plot(xs, ys, "-", label=r["config"]["temporal"], linewidth=1.5)
                st = r[kernel]["settle_time"]
                if st < len(trace):
                    ax.axvline(st, alpha=0.2, linestyle="--")
            ax.set_title(f"{kernel}")
            ax.set_xlabel("round")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="black", linewidth=0.3)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        axes[0].set_ylabel("alpha (population in PRO state)")
        plt.suptitle(f"Tier 2: N={N}, split={split}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"tier2_N{N}_{split}.png", dpi=120)
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"Tier 2: {len(CONFIGS)} configs x 2 kernels = {len(CONFIGS)*2} runs")
    print(f"  T_total = {T_TOTAL}")
    print()

    status = {
        "started_at": time.time(),
        "phase": "running",
        "n_configs": len(CONFIGS),
        "current_idx": 0,
        "current_config": None,
        "sub_phase": None,
        "last_update": time.time(),
    }
    write_status(status)

    results: List[Dict] = []
    for idx, cfg in enumerate(CONFIGS, 1):
        status["current_idx"] = idx
        status["current_config"] = cfg
        write_status(status)
        print(f"  [{idx}/{len(CONFIGS)}] N={cfg['N']:>4} "
              f"split={cfg['split']:>5} temporal={cfg['temporal']:>5}")
        r = run_one_config(cfg, status)
        d = r["discrete"]
        c = r["continuous"]
        print(f"    discrete  : settle@{d['settle_time']:>3}  alpha_f {d['final_alpha']:.3f}  "
              f"gain {d['decisiveness_gain']:.2f}  ({d['elapsed_s']:.1f}s)")
        print(f"    continuous: settle@{c['settle_time']:>3}  alpha_f {c['final_alpha']:.3f}  "
              f"gain {c['decisiveness_gain']:.2f}  ({c['elapsed_s']:.1f}s)")
        print(f"    RMSE c-vs-d: {r['rmse_continuous_vs_discrete']:.3f}")
        results.append(r)

    print()
    print("=" * 76)
    print("Tier 2 predictions:")
    print("=" * 76)
    pass_count, fail_count, predictions = evaluate_predictions(results)
    print()
    print(f"  {pass_count}/{pass_count+fail_count} predictions passed")

    print()
    print("plotting per-(N, split) trajectories...")
    plot_per_config(results)
    print(f"  plots saved to {PLOTS_DIR}")

    # Strip per-round traces from the JSON output to keep file size sane;
    # write per-config metrics + first/last 5 rounds of each trace as
    # a sample.
    serializable_results = []
    for r in results:
        sr = {
            "config": r["config"],
            "schedule_summary": r["schedule_summary"],
            "rmse_continuous_vs_discrete": r["rmse_continuous_vs_discrete"],
        }
        for k in ("discrete", "continuous"):
            sr[k] = {
                "elapsed_s": r[k]["elapsed_s"],
                "settle_time": r[k]["settle_time"],
                "final_alpha": r[k]["final_alpha"],
                "transient_max": r[k]["transient_max"],
                "decisiveness_gain": r[k]["decisiveness_gain"],
                "trace_head": r[k]["trace"][:5],
                "trace_tail": r[k]["trace"][-5:],
            }
        serializable_results.append(sr)

    RESULTS_PATH.write_text(json.dumps({
        "configs_run": len(CONFIGS),
        "T_total": T_TOTAL,
        "results": serializable_results,
        "predictions": predictions,
        "pass_count": pass_count,
        "fail_count": fail_count,
    }, indent=2), encoding="utf-8")
    print(f"  results saved to {RESULTS_PATH}")

    status["phase"] = "complete"
    status["pass_count"] = pass_count
    status["fail_count"] = fail_count
    write_status(status)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
