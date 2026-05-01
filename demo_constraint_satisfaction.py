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


def make_var(name: str, world: World, n_vars: int = 3) -> tuple[GeneralizedTendency, GeneralizedTendency]:
    """Add a boolean variable to the world as a (true, false) tendency pair.

    Each variable occupies its own axis in an n_vars-dimensional space.
    """
    var_index = {chr(ord("u") + i): i for i in range(n_vars)}
    var_index.update({"x": 0, "y": 1, "z": 2, "w": 3, "v": 4})
    idx = var_index[name]
    pos = tuple(1.0 if i == idx else 0.0 for i in range(n_vars))
    neg = tuple(-1.0 if i == idx else 0.0 for i in range(n_vars))
    t = GeneralizedTendency(
        id=f"{name}_T", thesis=f"{name}=T",
        anchor=pos, polarity_axis=pos,
        budget=1.0, bandwidth=0.7,
    )
    f = GeneralizedTendency(
        id=f"{name}_F", thesis=f"{name}=F",
        anchor=neg, polarity_axis=neg,
        budget=1.0, bandwidth=0.7,
    )
    world.add_tendency(t)
    world.add_tendency(f)
    return t, f


def run_case(name: str, constraints: list[Observation],
             expected: dict[str, str], use_growth: bool = False,
             vars: list[str] = None) -> bool:
    """Run one SAT case. Returns True if architecture found the expected
    assignment (or, for UNSAT cases with expected=None for some vars,
    if those vars came out ambiguous).
    """
    banner(f"CASE: {name}")
    var_list = vars if vars else ["x", "y", "z"]
    world = World()
    for v in var_list:
        make_var(v, world, n_vars=len(var_list))
    for c in constraints:
        world.add_observation(c)
    if use_growth:
        rounds, new_nodes = equilibrate_with_growth(
            world, max_outer=4, max_rounds=20, tolerance=1e-3,
            contention_threshold=0.1, offset=0.4,
        )
        print(f"\n  Equilibrated in {rounds} rounds; sprouted {new_nodes} new nodes")
    else:
        rounds = equilibrate(world, max_rounds=20, tolerance=1e-3)
        print(f"\n  Equilibrated in {rounds} rounds")
    scores = world.root_scores()
    print(f"\n  Tendency root scores:")
    success = True
    for vid in var_list:
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

    # Case 5: implication x -> y. Observe x=T. Architecture should
    # infer y=T via cross-staking propagation.
    # We encode "x -> y" as a vector observation that simultaneously
    # supports x_F (anti-x) OR y_T (pro-y). In coord space, an
    # observation at (-1, +1, 0) is on the F-side of x's axis and
    # T-side of y's axis. It's PRO to x_F AND PRO to y_T. Combined
    # with an observation forcing x=T, the equilibrium has to land
    # on y=T (since x=T forces the implication's other branch).
    results.append(("x -> y; x is T (infer y=T)", run_case(
        "x -> y; x=T. Architecture should infer y=T from x.",
        [
            Observation(id="c_imp", coords=(-1.0, +1.0, 0.0),
                        label="x->y: either x is F or y is T"),
            Observation(id="c_xT", coords=(+1.0, 0.0, 0.0), label="x must be T"),
        ],
        {"x": "T", "y": "T", "z": None},   # y inferred via implication
    )))

    # Case 6: x OR y; x is F. Architecture should infer y=T.
    # "x OR y" = an observation supporting x_T OR y_T. Encoded at
    # (+1, +1, 0). With x=F forced, only the y=T branch remains.
    results.append(("x OR y; x is F (infer y=T)", run_case(
        "x OR y; x=F. Architecture should infer y=T.",
        [
            Observation(id="c_or", coords=(+1.0, +1.0, 0.0),
                        label="x OR y: at least one is T"),
            Observation(id="c_xF", coords=(-1.0, 0.0, 0.0), label="x must be F"),
        ],
        {"x": "F", "y": "T", "z": None},
    )))

    # Case 7: chained implications. x -> y, y -> z. Observe x=T.
    # Two-hop reasoning: must conclude y=T (from x=T and x->y), then
    # z=T (from y=T and y->z).
    # x->y encoded at (-1, +1, 0): x is F or y is T.
    # y->z encoded at (0, -1, +1): y is F or z is T.
    results.append(("x->y; y->z; x=T (two-hop infer z=T)", run_case(
        "x->y; y->z; x=T. Two-hop reasoning: y=T then z=T.",
        [
            Observation(id="c_imp1", coords=(-1.0, +1.0, 0.0),
                        label="x->y"),
            Observation(id="c_imp2", coords=(0.0, -1.0, +1.0),
                        label="y->z"),
            Observation(id="c_xT", coords=(+1.0, 0.0, 0.0), label="x=T"),
        ],
        {"x": "T", "y": "T", "z": "T"},
    )))

    # Case 8: same as 7 but with growth enabled. The contended y
    # should sprout sub-claims that resolve the conditional structure.
    results.append(("x->y; y->z; x=T WITH GROWTH", run_case(
        "x->y; y->z; x=T WITH GROWTH. Test if depth-on-demand resolves it.",
        [
            Observation(id="c_imp1", coords=(-1.0, +1.0, 0.0),
                        label="x->y"),
            Observation(id="c_imp2", coords=(0.0, -1.0, +1.0),
                        label="y->z"),
            Observation(id="c_xT", coords=(+1.0, 0.0, 0.0), label="x=T"),
        ],
        {"x": "T", "y": "T", "z": "T"},
        use_growth=True,
    )))

    # Case 9: 4-variable chain. x -> y; y -> z; z -> w. x=T.
    # Three-hop chained inference. Expect T, T, T, T.
    results.append(("4-var chain: x->y; y->z; z->w; x=T", run_case(
        "x->y; y->z; z->w; x=T. Three-hop chain. Expect all T.",
        [
            Observation(id="c1", coords=(-1.0, +1.0, 0.0, 0.0), label="x->y"),
            Observation(id="c2", coords=(0.0, -1.0, +1.0, 0.0), label="y->z"),
            Observation(id="c3", coords=(0.0, 0.0, -1.0, +1.0), label="z->w"),
            Observation(id="c_xT", coords=(+1.0, 0.0, 0.0, 0.0), label="x=T"),
        ],
        {"x": "T", "y": "T", "z": "T", "w": "T"},
        vars=["x", "y", "z", "w"],
    )))

    # Case 10: 4-var with mixed constraints.
    # x AND y; NOT z; w iff (x AND z). With x=T, y=T, z=F: w iff F = F.
    # Expected: T, T, F, F.
    results.append(("4-var mixed: x AND y; NOT z; w iff (x AND z)", run_case(
        "x AND y; NOT z; w<->(x AND z). Expect T, T, F, F.",
        [
            Observation(id="c_xT", coords=(+1.0, 0.0, 0.0, 0.0), label="x=T"),
            Observation(id="c_yT", coords=(0.0, +1.0, 0.0, 0.0), label="y=T"),
            Observation(id="c_zF", coords=(0.0, 0.0, -1.0, 0.0), label="z=F"),
            # w iff (x AND z): when x=T and z=T, w=T. When either is F, w=F.
            # Encode: an obs at (-1, 0, -1, +1) says "if x=F or z=F or w=T".
            # Combined with z=F, this is satisfied (z is F), so w stays free
            # unless we add the other direction. For simplicity, use just
            # one direction: "w must be F if z is F" -> obs at (0, 0, +1, -1).
            Observation(id="c_wzF", coords=(0.0, 0.0, +1.0, -1.0),
                        label="z=T or w=F (so if z=F, w must be F is forced when... actually no)"),
            # Wait, the simpler one: if z=F then w=F. Obs at (0, 0, +1, -1).
            # This says z is on T-side OR w is on F-side. Since z=F (not T),
            # w must be F.
        ],
        {"x": "T", "y": "T", "z": "F", "w": "F"},
        vars=["x", "y", "z", "w"],
    )))

    banner("OVERALL")
    for name, ok in results:
        print(f"  {'OK' if ok else '--'}  {name}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n  {n_ok}/{len(results)} cases passed")
    print()


if __name__ == "__main__":
    main()
