"""
Reseed-and-equilibrate: the universal operation.

Reseeding is the only way the engine generates non-present configurations.
Past, future, alternate-present, cosmological-time-slice -- all produced
by the same call: substitute the tendencies that differ for the target
configuration, hold the rest fixed, run calibration until allocations
settle, return the new state.

The operation is a pure function over present state. Inputs are deep-
copied at the boundary; outputs are a fresh state object. No side
effects on the inputs.

Substitution interface
======================

A substitution is a total replacement: ``substitute(id, new_tendency)``.
The engine does not reconcile fields. The caller builds the replacement
to look however the target configuration requires -- mostly identical
to the original, completely different, or anywhere in between. The
engine's job is composition, not construction.

Substitutions can also add (id not yet present) or remove (replacement
is None / sentinel) tendencies. After substitution, allocations
renormalize and calibration re-equilibrates the modified set.

Convergence
===========

The calibration loop applies the existing contraction-map reallocation
iteratively until the largest absolute allocation delta falls below
``tolerance``, capped at ``max_iterations``. The contraction property
guarantees convergence in principle; the cap is a sanity belt.

Each completed reseed emits one ALLOCATION_SHIFTED event per affected
tendency, reporting the *net* shift over the whole reseed (not per
inner iteration). This keeps event counts bounded and avoids
spuriously inflating engine-time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from ..models.tendency import Tendency, TendencySet
from ..models.lineage import (
    EngineClock,
    EventType,
    Lineage,
    LineageRecorder,
    StakeWeightGraph,
)


# ---------------------------------------------------------------------------
# Present-state aggregate
# ---------------------------------------------------------------------------


@dataclass
class PresentState:
    """The full present state: tendencies + lineage + stake-weight graph.

    This is the unit reseed-and-equilibrate operates on. It is
    intentionally a value-type aggregate -- nothing here references
    the live arena, so deep-copy is straightforward.

    Reseed never mutates an input PresentState; it returns a new one.
    """

    tendencies: TendencySet
    lineages: dict[str, Lineage] = field(default_factory=dict)
    graph: StakeWeightGraph = field(default_factory=StakeWeightGraph)

    def copy(self) -> "PresentState":
        """Deep-copy the entire state. Safe to mutate the result."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Substitutions and result
# ---------------------------------------------------------------------------


@dataclass
class Substitution:
    """One total replacement.

    - ``new_tendency`` populated -> add or replace.
    - ``new_tendency`` is None    -> remove the tendency at ``id``.

    Optionally, the caller can supply replacement edges for the
    stake-weight graph. If ``edges`` is provided, the new tendency's
    edges are set to exactly those weights (any prior edges from this
    tendency are dropped first). If ``edges`` is None, the existing
    edges are preserved -- the caller is reusing the original
    tendency's neighborhood.
    """

    id: str
    new_tendency: Optional[Tendency] = None
    edges: Optional[dict[str, float]] = None    # other_id -> weight

    @property
    def is_removal(self) -> bool:
        return self.new_tendency is None


@dataclass
class ReseedResult:
    """What a reseed produced."""

    state: PresentState
    iterations: int
    converged: bool                  # True if delta fell below tolerance before cap
    final_max_delta: float           # the largest |delta| at termination
    affected_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def reseed_and_equilibrate(
    state: PresentState,
    substitutions: list[Substitution],
    *,
    tolerance: float = 1e-4,
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    recorder: Optional[LineageRecorder] = None,
) -> ReseedResult:
    """Apply substitutions, calibrate to a fixpoint, return the new state.

    Parameters
    ----------
    state:
        Input state. Deep-copied; not mutated.
    substitutions:
        List of total replacements (or removals). Order matters only when
        substitutions interact -- e.g., adding A and removing A in the
        same call applies them in order.
    tolerance:
        Calibration loop terminates when the largest |allocation delta|
        across one iteration falls below this.
    max_iterations:
        Hard cap. If hit, ``ReseedResult.converged`` is False but the
        operation still returns the partially-settled state.
    learning_rate:
        Per-iteration blend factor toward the substitution-implied target
        distribution. The contraction property requires 0 < lr < 1.
    recorder:
        If provided, emits one ALLOCATION_SHIFTED event per affected
        tendency at the end of calibration, reporting the net shift.
        If None, no events are emitted (still pure-function from the
        engine's perspective; caller chose to discard event provenance).

    Returns
    -------
    ReseedResult with the new state, iteration count, convergence
    status, the final max delta, and the list of tendency ids whose
    allocations changed by more than tolerance from their input values.
    """
    if not (0.0 < learning_rate < 1.0):
        raise ValueError(
            f"learning_rate must be in (0, 1) for the contraction map to "
            f"converge; got {learning_rate}"
        )
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0; got {tolerance}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0; got {max_iterations}")

    new_state = state.copy()

    # Snapshot of pre-reseed allocations so we can report net deltas at the end.
    pre_alloc: dict[str, float] = {
        t.id: t.allocation for t in new_state.tendencies.all()
    }

    # 1. Apply substitutions.
    target_allocs = _apply_substitutions(new_state, substitutions)

    # 2. Renormalize post-substitution.
    new_state.tendencies.normalize()

    # 3. Calibrate: blend allocations toward the target distribution
    #    until the iteration delta is below tolerance.
    iterations, converged, final_delta = _calibrate(
        new_state.tendencies,
        target_allocs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
    )

    # 4. Compute affected ids (those whose allocation moved by > tolerance).
    affected: list[str] = []
    for t in new_state.tendencies.all():
        before = pre_alloc.get(t.id, 0.0)
        if abs(t.allocation - before) > tolerance:
            affected.append(t.id)

    # 5. Emit one ALLOCATION_SHIFTED per affected tendency, if a recorder
    #    was supplied.
    if recorder is not None:
        for tid in affected:
            t = new_state.tendencies.get(tid)
            recorder.emit_allocation_shift(
                origin_id=tid,
                new_allocation=t.allocation,
                delta=t.allocation - pre_alloc.get(tid, 0.0),
            )

    return ReseedResult(
        state=new_state,
        iterations=iterations,
        converged=converged,
        final_max_delta=final_delta,
        affected_ids=affected,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_substitutions(
    state: PresentState,
    substitutions: list[Substitution],
) -> dict[str, float]:
    """Apply each substitution in order. Returns the implied target allocations.

    The caller's specified ``new_tendency.allocation`` becomes the *target*
    for calibration -- the value the tendency should land on at the end of
    the reseed -- but it is *not* written directly into the live tendency
    set. Calibration drives the live allocation toward that target. This
    keeps the calibration loop the only thing that moves allocations,
    regardless of whether a substitution implies a small or large shift.

    Removals drop the tendency from the set, the lineage dict, and the
    graph. Additions inject a new tendency at allocation 0; calibration
    raises it. Replacements preserve the existing live allocation as the
    starting point so calibration has work to do.
    """
    targets: dict[str, float] = {
        t.id: t.allocation for t in state.tendencies.all()
    }

    for sub in substitutions:
        if sub.is_removal:
            state.tendencies.remove(sub.id)
            state.lineages.pop(sub.id, None)
            state.graph.remove_tendency(sub.id)
            targets.pop(sub.id, None)
            continue

        # Addition or replacement
        new_t = copy.deepcopy(sub.new_tendency)
        target_alloc = new_t.allocation

        if state.tendencies.has(new_t.id):
            # Replacement: keep the live allocation as the starting point so
            # calibration drives it toward the new target.
            existing = state.tendencies.get(new_t.id)
            new_t.allocation = existing.allocation
        else:
            # Addition: start at 0 and let calibration raise it.
            new_t.allocation = 0.0

        state.tendencies.add(new_t)
        state.lineages.setdefault(new_t.id, Lineage())
        state.graph.add_tendency(new_t.id)

        if sub.edges is not None:
            # Replace neighborhood: drop existing edges, install new ones.
            state.graph.remove_tendency(new_t.id)
            state.graph.add_tendency(new_t.id)
            for other_id, w in sub.edges.items():
                state.graph.add_edge(new_t.id, other_id, w)

        targets[new_t.id] = target_alloc

    # Normalize targets so they sum to 1 (caller may not have done this).
    total = sum(max(0.0, v) for v in targets.values())
    if total > 0:
        targets = {k: max(0.0, v) / total for k, v in targets.items()}
    elif targets:
        n = len(targets)
        targets = {k: 1.0 / n for k in targets}

    return targets


def _calibrate(
    tendencies: TendencySet,
    target_allocs: dict[str, float],
    *,
    tolerance: float,
    max_iterations: int,
    learning_rate: float,
) -> tuple[int, bool, float]:
    """Iterate the contraction map until allocations settle.

    At each step, every allocation moves a fraction ``learning_rate`` of
    the way toward its target. After each step we renormalize (the
    target distribution sums to 1; small numerical drift is corrected).

    Returns (iterations_run, converged?, final_max_delta).
    """
    last_max_delta = 0.0
    for it in range(1, max_iterations + 1):
        max_delta = 0.0
        for t in tendencies.all():
            target = target_allocs.get(t.id, t.allocation)
            delta = (target - t.allocation) * learning_rate
            t.allocation += delta
            if t.allocation < 0.0:
                t.allocation = 0.0
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        tendencies.normalize()
        last_max_delta = max_delta
        if max_delta < tolerance:
            return it, True, max_delta
    return max_iterations, False, last_max_delta
