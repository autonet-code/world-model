#!/usr/bin/env python3
"""
Boolean constraint satisfaction via tendency equilibration.

Setup
-----

Three boolean variables x, y, z. Each is two tendencies on a 1D axis:
  - x_true:  thesis "x is true",  anchor +1, polarity +1
  - x_false: thesis "x is false", anchor -1, polarity -1
(same for y, z).

Constraints are encoded as OBSERVATIONS that each constrained pair
absorbs PRO or contradicts. For example, the constraint "x AND y"
generates these observations into the world:
  - one observation at the x_true anchor (PRO for x_true, CON for x_false)
  - one observation at the y_true anchor (PRO for y_true, CON for y_false)

For "x XOR y", we get:
  - observation pair encoding (x=true, y=false) and (x=false, y=true)
  ... the structure varies per constraint.

For our test we use a 3-var case with a known unique satisfying
assignment:

  Constraints:
    x AND y      -> requires x=T, y=T
    NOT z        -> requires z=F

  Unique solution: x=T, y=T, z=F.

After equilibration, we read off each variable's bit by which
tendency (true vs false) has the higher root score. Test passes if
the architecture finds the unique solution.

Why this matters
----------------

This is a multi-tendency cross-staking pattern that REQUIRES the
adversarial dynamics to work. If x_true and y_true don't reinforce
each other (both supported by the AND constraint), we won't converge.
If z_true and z_false don't separate (with z_false winning under the
NOT constraint), we won't converge.

It's also the simplest reasoning-shaped test: the answer isn't a
classification, it's a SATISFYING ASSIGNMENT discovered by the
equilibrium of competing claims.
"""

from __future__ import annotations

from world_model.generalized import (
    GeneralizedTendency,
    Observation,
    World,
    equilibrate,
    equilibrate_with_growth,
)


def banner(s: str) -> None:
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def make_var(name: str, world: World) -> tuple[GeneralizedTendency, GeneralizedTendency]:
    """Add a boolean variable to the world as a (true, false) tendency pair.

    Both tendencies sit on the same 1D axis (one per variable). To
    keep the variables on independent axes (x, y, z each their own
    direction), the *anchor* is in a 3D space where each variable
    occupies one of the three axes.
    """
    t = GeneralizedTendency(
        id=f"{name}_T",
        thesis=f"{name}=T",
        anchor={"x": (+1.0, 0.0, 0.0), "y": (0.0, +1.0, 0.0), "z": (0.0, 0.0, +1.0)}[name],
        polarity_axis={"x": (+1.0, 0.0, 0.0), "y": (0.0, +1.0, 0.0), "z": (0.0, 0.0, +1.0)}[name],
        budget=1.0,
        bandwidth=0.7,   # narrow so x's claims don't interfere with y's
    )
    f = GeneralizedTendency(
        id=f"{name}_F",
        thesis=f"{name}=F",
        anchor={"x": (-1.0, 0.0, 0.0), "y": (0.0, -1.0, 0.0), "z": (0.0, 0.0, -1.0)}[name],
        polarity_axis={"x": (-1.0, 0.0, 0.0), "y": (0.0, -1.0, 0.0), "z": (0.0, 0.0, -1.0)}[name],
        budget=1.0,
        bandwidth=0.7,
    )
    world.add_tendency(t)
    world.add_tendency(f)
    return t, f


def run_case(name: str, constraints: list[Observation],
             expected: dict[str, str]) -> bool:
    """Run one SAT case. Returns True if architecture found the expected
    assignment (or, for UNSAT cases with expected=None for some vars,
    if those vars came out ambiguous).
    """
    banner(f"CASE: {name}")
    world = World()
    make_var("x", world)
    make_var("y", world)
    make_var("z", world)
    for c in constraints:
        world.add_observation(c)
    rounds = equilibrate(world, max_rounds=20, tolerance=1e-3)
    print(f"\n  Equilibrated in {rounds} rounds")
    scores = world.root_scores()
    print(f"\n  Tendency root scores:")
    success = True
    for vid in ["x", "y", "z"]:
        s_T = scores[f"{vid}_T"]
        s_F = scores[f"{vid}_F"]
        winner = "T" if s_T > s_F else "F"
        gap = abs(s_T - s_F)
        exp = expected.get(vid)
        if exp is None:
            marker = f"ambiguous (gap={gap:.3f})"
            ok = gap < 0.1
        else:
            ok = winner == exp
            marker = "OK" if ok else f"-- expected {exp}"
        if not ok:
            success = False
        print(f"    {vid}: T={s_T:+.3f}, F={s_F:+.3f}  -> {winner}  ({marker})")
    return success


def main():
    results = []

    # Case 1: x AND y, NOT z. Unique sol T, T, F.
    results.append(("x AND y; NOT z", run_case(
        "x AND y; NOT z. Unique solution: T, T, F.",
        [
            Observation(id="c_xT", coords=(+1.0, 0.0, 0.0), label="x must be T"),
            Observation(id="c_yT", coords=(0.0, +1.0, 0.0), label="y must be T"),
            Observation(id="c_zF", coords=(0.0, 0.0, -1.0), label="z must be F"),
        ],
        {"x": "T", "y": "T", "z": "F"},
    )))

    # Case 2: NOT x, y, z. Unique sol F, T, T.
    results.append(("NOT x; y; z", run_case(
        "NOT x; y; z. Unique solution: F, T, T.",
        [
            Observation(id="c_xF", coords=(-1.0, 0.0, 0.0), label="x must be F"),
            Observation(id="c_yT", coords=(0.0, +1.0, 0.0), label="y must be T"),
            Observation(id="c_zT", coords=(0.0, 0.0, +1.0), label="z must be T"),
        ],
        {"x": "F", "y": "T", "z": "T"},
    )))

    # Case 3: contradictory constraints on x. Should be ambiguous on x.
    results.append(("x AND NOT x (contradictory); y; z", run_case(
        "x AND NOT x; y; z. UNSAT for x; satisfiable for y, z.",
        [
            Observation(id="c_xT", coords=(+0.99, 0.0, 0.0), label="x must be T"),
            Observation(id="c_xF", coords=(-0.99, 0.0, 0.0), label="x must be F"),
            Observation(id="c_yT", coords=(0.0, +1.0, 0.0), label="y must be T"),
            Observation(id="c_zT", coords=(0.0, 0.0, +1.0), label="z must be T"),
        ],
        {"x": None, "y": "T", "z": "T"},   # x ambiguous, y,z resolved
    )))

    # Case 4: just one constraint. Should resolve x, leave y/z ambiguous.
    results.append(("x; (y, z unconstrained)", run_case(
        "x. Only x is constrained; y and z should be ambiguous (low gap).",
        [
            Observation(id="c_xT", coords=(+1.0, 0.0, 0.0), label="x must be T"),
        ],
        {"x": "T", "y": None, "z": None},
    )))

    banner("OVERALL")
    for name, ok in results:
        print(f"  {'OK' if ok else '--'}  {name}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n  {n_ok}/{len(results)} cases passed")
    print()


if __name__ == "__main__":
    main()
