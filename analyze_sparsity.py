#!/usr/bin/env python3
"""
Analyze sparsity of stake-weight distributions in fictional-world equilibria.

The architectural hypothesis: stake-weight is concentrated on a small
fraction of edges (heavy-tailed distribution), making localized inference
viable. We now test the sharper, more informative version:

    Sparsity is a function of world-coherence.
    Coherent worlds produce heavy-tailed stake distributions.
    Incoherent worlds produce flat ones.

The flatness observed in the early personality data is consistent with
either insufficient coherence in the modelled subject or insufficient
arena settlement. Rather than fit the engine to that legacy data, we
develop the engine against parametric fictional worlds where coherence
is a knob we control.

Stages run by default:

  1. Synthetic controls       (analyzer self-check on uniform vs power-law)
  2. Coherence sweep          (the substantive experiment)

Usage:
    python analyze_sparsity.py                  # default sweep
    python analyze_sparsity.py --controls-only  # just analyzer self-check
    python analyze_sparsity.py --legacy         # also run on saved JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the sparsity module directly via importlib to avoid triggering
# the top-level world_model package __init__, which currently imports a
# missing attention.curves module. Pre-existing breakage, unrelated.
import importlib.util
def _load_module(name: str, relpath: list[str]):
    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).parent.joinpath(*relpath),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required for @dataclass to find module
    spec.loader.exec_module(mod)
    return mod


# Load sparsity first (fictional_worlds depends on it via relative import,
# but we're sidestepping the package, so we wire that dependency by hand).
_sparsity = _load_module(
    "world_model.analysis.sparsity",
    ["world_model", "analysis", "sparsity.py"],
)

# Make the sparsity module accessible to fictional_worlds' relative import.
# We have to register it under a name fictional_worlds will look up.
sys.modules.setdefault("world_model", type(sys)("world_model"))
sys.modules.setdefault("world_model.analysis", type(sys)("world_model.analysis"))
sys.modules["world_model.analysis.sparsity"] = _sparsity

_fictional = _load_module(
    "world_model.analysis.fictional_worlds",
    ["world_model", "analysis", "fictional_worlds.py"],
)

compute_sparsity_metrics = _sparsity.compute_sparsity_metrics
extract_stake_edges_from_dict = _sparsity.extract_stake_edges_from_dict
synthetic_powerlaw_edges = _sparsity.synthetic_powerlaw_edges
synthetic_uniform_edges = _sparsity.synthetic_uniform_edges
coherence_sweep = _fictional.coherence_sweep
format_sweep_table = _fictional.format_sweep_table


def banner(text: str) -> None:
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def run_synthetic_controls() -> bool:
    """Validate the analyzer on known-shape synthetic data.

    Returns True iff both controls behave as expected:
      - uniform: hypothesis NOT supported
      - power-law: hypothesis supported
    """
    banner("CONTROL 1: synthetic uniform edges (expect: NOT supported)")
    uniform = synthetic_uniform_edges(n_edges=500)
    r1 = compute_sparsity_metrics(uniform)
    print(r1.summary())

    banner("CONTROL 2: synthetic power-law edges (expect: SUPPORTED)")
    powerlaw = synthetic_powerlaw_edges(n_edges=500, alpha=2.5)
    r2 = compute_sparsity_metrics(powerlaw)
    print(r2.summary())

    uniform_ok = not r1.hypothesis_supported
    powerlaw_ok = r2.hypothesis_supported

    banner("CONTROL VERDICT")
    print(f"uniform behaves correctly:    {uniform_ok}  (hypothesis_supported = {r1.hypothesis_supported})")
    print(f"power-law behaves correctly:  {powerlaw_ok}  (hypothesis_supported = {r2.hypothesis_supported})")
    print()
    if uniform_ok and powerlaw_ok:
        print("Analyzer passes its own controls.")
    else:
        print("ANALYZER BUG: at least one control failed. Real-data results are not trustworthy.")
    return uniform_ok and powerlaw_ok


def run_on_file(path: Path) -> None:
    if not path.exists():
        print(f"  skipped: {path} not found")
        return
    banner(f"REAL DATA: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    edges = extract_stake_edges_from_dict(data)
    print(f"extracted {len(edges)} stake-edges from {path.name}")
    if not edges:
        print("(no stake-edges found in this file -- nothing to analyze)")
        return
    report = compute_sparsity_metrics(edges)
    print(report.summary())

    print()
    print("agent share (sorted by weight):")
    for agent, share in sorted(report.agent_share.items(), key=lambda kv: -kv[1]):
        print(f"  {agent:14s} {share:6.1%}")

    if report.tree_share:
        print()
        print(f"tree share ({len(report.tree_share)} trees, sorted by weight):")
        for i, (tree, share) in enumerate(
            sorted(report.tree_share.items(), key=lambda kv: -kv[1])
        ):
            if i >= 10:
                print(f"  ... and {len(report.tree_share) - 10} more")
                break
            print(f"  {tree[:8]}...  {share:6.1%}")


def run_coherence_sweep() -> None:
    banner("COHERENCE SWEEP: sparsity vs world-coherence")
    print("Sweeping coherence in [0, 1]. 7 agents, 200 observations, 5 trees,")
    print("stake density 0.4, averaged over 5 random seeds per point.\n")
    points = coherence_sweep()
    print(format_sweep_table(points))

    print()
    print("Reading the table:")
    print("  - gini:    0 = uniform weights, 1 = total concentration")
    print("  - top10%:  fraction of total weight on the heaviest 10% of edges")
    print("  - alpha:   power-law tail exponent (None when no plausible tail)")
    print("  - pl_ok:   majority-of-seeds plausible power-law fit")
    print()
    if points:
        first, last = points[0], points[-1]
        gini_delta = last.gini - first.gini
        top10_delta = last.top_10pct_share - first.top_10pct_share
        print(f"At coherence={first.coherence:.2f}:  gini={first.gini:.3f}  top10={first.top_10pct_share:.1%}")
        print(f"At coherence={last.coherence:.2f}:  gini={last.gini:.3f}  top10={last.top_10pct_share:.1%}")
        print(f"Delta:                   gini +{gini_delta:.3f}  top10 +{top10_delta:.1%}")
        if gini_delta > 0.2 and top10_delta > 0.1:
            print()
            print("Sparsity grows with coherence as predicted.")
        elif gini_delta < 0.05 and top10_delta < 0.05:
            print()
            print("Sparsity does NOT track coherence -- architectural concern.")
        else:
            print()
            print("Mixed signal -- partial support, worth deeper investigation.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controls-only", action="store_true",
        help="run only the analyzer self-check on synthetic uniform/power-law data",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="also run on saved personality equilibria for reference",
    )
    parser.add_argument(
        "--file", type=Path, action="append", default=None,
        help="specific legacy JSON file to analyze (implies --legacy)",
    )
    args = parser.parse_args()

    controls_ok = run_synthetic_controls()

    if args.controls_only:
        return 0 if controls_ok else 1

    run_coherence_sweep()

    if args.legacy or args.file:
        files = args.file or [
            Path("data/andrei_adversarial.json"),
            Path("data/andrei_world_model.json"),
        ]
        for path in files:
            run_on_file(path)

    return 0 if controls_ok else 1


if __name__ == "__main__":
    sys.exit(main())
