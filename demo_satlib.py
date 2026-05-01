#!/usr/bin/env python3
"""
SATLIB uf20-91: 20-variable random 3-SAT, near phase transition.

We feed the architecture standard 3-SAT instances from SATLIB and
measure how often it finds a satisfying assignment.

Encoding
--------

Each variable v_i becomes a tendency pair (v_i_T at +1 on dim i,
v_i_F at -1 on dim i) in 20-dim coordinate space.

Each 3-SAT clause `(L1 OR L2 OR L3)` is encoded as a single
observation in 20-dim space. For each literal:
  - positive literal  v_i  -> coord[i] = +1
  - negated literal  -v_i  -> coord[i] = -1
  - other dimensions  -> coord[i] = 0

The clause-observation is satisfied if the assignment puts at least
one of its literals on the matching side. Architecture should find
this via equilibration.

Reading the answer
------------------

For each variable v_i, compare root scores of v_i_T and v_i_F. The
higher score is the architecture's answer. Verify against the clauses.

Note: SATLIB uf20-91 is at the phase transition (clause/var ratio
~4.55), so it's a hard regime. Pass rate of even 50% would be a
real result. Modern SAT solvers reach 100% but they use DPLL with
backtracking; we have one-shot equilibration.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
)


def parse_cnf(path: str) -> tuple[int, list[list[int]]]:
    """Parse DIMACS CNF. Returns (n_vars, clauses) where each clause is
    a list of signed integers (positive = positive literal, negative =
    negated literal). Variables are 1-indexed.
    """
    n_vars = 0
    clauses: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c") or line.startswith("%"):
                continue
            if line.startswith("p"):
                parts = line.split()
                n_vars = int(parts[2])
                continue
            if line == "0":
                continue
            tokens = line.split()
            literals = [int(t) for t in tokens if t != "0"]
            if literals:
                clauses.append(literals)
    return n_vars, clauses


def build_world(n_vars: int, clauses: list[list[int]]) -> World:
    world = World()
    # Tendencies: one T and one F per variable
    for i in range(n_vars):
        pos = tuple(1.0 if j == i else 0.0 for j in range(n_vars))
        neg = tuple(-1.0 if j == i else 0.0 for j in range(n_vars))
        world.add_tendency(GeneralizedTendency(
            id=f"v{i+1}_T",
            thesis=f"v{i+1}=T",
            anchor=pos,
            polarity_axis=pos,
            budget=1.0,
            bandwidth=0.7,
        ))
        world.add_tendency(GeneralizedTendency(
            id=f"v{i+1}_F",
            thesis=f"v{i+1}=F",
            anchor=neg,
            polarity_axis=neg,
            budget=1.0,
            bandwidth=0.7,
        ))
    # Observations: one per clause
    for c_idx, clause in enumerate(clauses):
        coords = [0.0] * n_vars
        for lit in clause:
            i = abs(lit) - 1
            coords[i] = 1.0 if lit > 0 else -1.0
        world.add_observation(Observation(
            id=f"c{c_idx}",
            coords=tuple(coords),
            label=f"clause {clause}",
        ))
    return world


def read_assignment(world: World, n_vars: int) -> dict[int, bool]:
    """Return {var_index_1based: bool}."""
    scores = world.root_scores()
    out: dict[int, bool] = {}
    for i in range(1, n_vars + 1):
        st = scores[f"v{i}_T"]
        sf = scores[f"v{i}_F"]
        out[i] = st > sf
    return out


def evaluate_assignment(assignment: dict[int, bool],
                        clauses: list[list[int]]) -> tuple[int, int]:
    """Return (n_satisfied, n_total) clauses."""
    sat = 0
    for clause in clauses:
        if any(
            (assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)])
            for lit in clause
        ):
            sat += 1
    return sat, len(clauses)


def run_instance(path: str) -> tuple[bool, int, int, float, int]:
    """Run one CNF. Returns (all_satisfied, n_satisfied, n_total,
    elapsed_seconds, rounds)."""
    n_vars, clauses = parse_cnf(path)
    world = build_world(n_vars, clauses)
    t0 = time.time()
    rounds = equilibrate(world, max_rounds=10, tolerance=1e-3)
    elapsed = time.time() - t0
    assignment = read_assignment(world, n_vars)
    sat, total = evaluate_assignment(assignment, clauses)
    return sat == total, sat, total, elapsed, rounds


def main(n_instances: int = 20):
    data_dir = Path("data/uf20-91")
    cnf_files = sorted(data_dir.glob("uf20-*.cnf"))[:n_instances]
    print()
    print("=" * 70)
    print(f"SATLIB uf20-91: running {len(cnf_files)} instances")
    print("=" * 70)
    print(f"  20 variables, ~91 clauses each, near phase transition.")
    print(f"  Architecture: single equilibration per instance, no backtracking.\n")

    n_full = 0
    total_sat = 0
    total_clauses = 0
    total_time = 0.0
    rates: list[float] = []
    for i, path in enumerate(cnf_files):
        all_sat, sat, total, elapsed, rounds = run_instance(str(path))
        rate = sat / total
        rates.append(rate)
        if all_sat:
            n_full += 1
        total_sat += sat
        total_clauses += total
        total_time += elapsed
        if (i + 1) % 5 == 0 or i < 5:
            status = "FULL SAT" if all_sat else f"{sat}/{total} ({rate:.0%})"
            print(f"  [{i+1}/{len(cnf_files)}] {path.name}: {status} in {elapsed:.2f}s ({rounds} rounds)")
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Full-SAT rate:       {n_full}/{len(cnf_files)} = {n_full/len(cnf_files):.1%}")
    print(f"  Avg clause rate:     {total_sat}/{total_clauses} = {total_sat/total_clauses:.1%}")
    print(f"  Avg time/instance:   {total_time/len(cnf_files):.2f}s")
    print(f"  Best instance rate:  {max(rates):.1%}")
    print(f"  Worst instance rate: {min(rates):.1%}")
    print()
    if n_full / len(cnf_files) > 0.10:
        print("  Architecture handles 3-SAT at phase transition with measurable")
        print("  pass rate. Single-equilibration result; with backtracking it'd")
        print("  be higher.")
    else:
        print("  Architecture finds partial assignments but rarely full-SAT on")
        print("  phase-transition instances. The constraint structure is too")
        print("  intricate for one-shot equilibration to resolve fully.")
    print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
