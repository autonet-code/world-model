# Post-autonet findings: equilibration scaling is the launch blocker

**Update for the substrate agent after several months of integration work
into autonet. The Tier 3B / 6-root architecture is intact and validated.
But integrating it into a live daemon surfaced a scaling problem the
toy experiments couldn't see: equilibration cost grows fast enough with
world size that a few hundred events makes the daemon's per-event budget
infeasible.**

This isn't a regression in substrate correctness. It's an asymptotic
problem in the implementation that the small-N experiments hid.

## What integration looks like in autonet

The substrate runs inside the daemon as a `WorldService` (autonet
repo, `nodes/common/world_service.py`). Public API mirrors the
experiment scripts:

  - `submit_observation(obs, agent_id, ...)` — add one event, apply
    it, equilibrate
  - `submit_work_units([(problem, resolution, outcome), ...])` —
    batch convenience wrapper
  - `coords_for_text(...)`, `probe_inference(...)` — read paths

Everything is persisted to `~/.atn/world/<rpb_address>/` with a
WAL-style event log and periodic score snapshots. Crash-safe; resumes
by replaying events from disk on construction.

The 6-root charter from Tier 3B is the live charter:
`life_precious, self_preservation, promotion_of_intelligence,
evolution, correctness, simplicity`. Charter head is 6 dims; embedding
tail is 1024 dims (sentence-transformers MiniLM via the local
usefulness embedder). Total per-event coord vector is 1030 dims.

LLM-binary-flag adapter (Tier 3A swap-in) is wired but not yet on the
default path — `submit_work_units` zeros the charter head, only the
tail carries signal. The LLM path exists for explicit-charter scoring
when the cost is justified (e.g. one-shot seed runs against the Claude
Max bridge), not for every live training event.

## The snag — empirical

We tried to seed the live substrate from extracted Claude Code work
units (the same kind of corpus used in the tier3a/3b experiments,
just much larger — ~9000 work units from real sessions). Heuristic
path (no LLM):

  - **Run 1**: world started at 62 events on disk (from prior daemon
    activity). Added 50 work units → 100 new events. Wall time:
    **843 seconds (~14 min)**.  Ending world size: 162 events.
  - **Run 2**: world at 162. Added 100 units, ~50 effective new (the
    other 50 deduped against Run 1). New events applied:
    138. Wall time: **3766 seconds (~63 min)**. Ending world size:
    300 events.

So adding 50 units took ~14 minutes when starting from 62 events,
and ~63 minutes when starting from 162 events. The per-event cost
grows roughly with current world size — consistent with O(N) work
per added event, i.e. **O(N²) total** to seed a world of size N.

At this trajectory:

  - Reaching 1,000 events: hours
  - Reaching 10,000 events: days
  - Live operation past a few hundred events: epoch close blocks
    long enough that new gossip events queue faster than they apply

The wall this hits is the daemon's own viability, not just the seed
script's speed.

## What we think is happening

The current `equilibrate` loop appears to settle every node against
the whole world after each new addition. The cost dominates because:

  1. The full-world settle runs even when the new node lands in a
     small local neighborhood and would only meaningfully affect a
     few existing nodes.
  2. embedding_dim=1024 makes every pairwise distance computation
     expensive in the constant factor — fine for one settle pass,
     painful when the pass touches every existing node.
  3. There's no spatial index, so each settle is a brute O(N) scan
     of the existing world.

(Phase 10.8 in the autonet backlog flags this perf issue but had
been deprioritized in favor of feature work. It's now load-bearing.)

## The architectural question

We've been thinking about it this way:

**A new node about a refactor doesn't need to disturb the
life_precious subtree on the other side of the charter.**  Most
events are local. So the obvious-in-retrospect fix is to make
equilibration respect the substrate's own fractal structure:

  - When a new event lands, settle it only within its tendency's
    subtree (or the cluster it joins).
  - Cross-tendency settling only happens when a node *actually*
    straddles tendencies — e.g. an ethics-charged correctness fix
    that has weight on both `life_precious` and `correctness`.
  - For pure within-tendency events (most of them), the equilibrate
    scope drops from O(N) to O(K) where K is the local cluster size.

This is the substrate becoming operationally fractal, not just
visually fractal. It also opens the door to specialization
incentives in autonet: a daemon could volunteer to "host" a
particular tendency's subtree without holding the whole world.
Each daemon equilibrates locally; cross-tendency events trigger
gossip between the affected hosts only.

## Three questions for the substrate agent

We're not sure if this is a simple refactor or a deeper redesign.
Three things we'd want your read on:

1. **Is per-tendency-subtree equilibration correctness-preserving?**
   Are there hidden global invariants (e.g. some shared normalization,
   PCA basis, or charter anchor) that *require* every settle pass to
   touch the whole world? Or is the global settle a convenience of
   the current implementation rather than an algorithmic necessity?

2. **What's the right cross-tendency policy?**  When an event has
   non-trivial weight on multiple tendencies (e.g. coords with
   non-zero values on both `correctness` and `simplicity`), what
   should happen? Settle in both subtrees independently? Settle in
   the "dominant" one and propagate a shadow to the others? Some
   kind of message-passing between subtree hosts?

3. **Should embedding_dim drop?**  Independent of the equilibration
   scope question — was 1024 ever necessary? Tier 3A/3B both
   validated at 1024, but neither tested whether 256 or 128 would
   work too. Dropping dim is the easy, mechanical optimization; we
   want to know whether it's *also* a free win on quality, or if
   there's a reason to hold the line at 1024.

## State on the ground

  - autonet branch is on the substrate-native path (no VL-JEPA,
    no FedAvg, no old contracts; pre-substrate paths deleted)
  - 6-root charter live, used in production code paths
  - Two daemons can federate over libp2p, gossip events, close
    epochs canonically, mint on-chain
  - Visualizers (constellation + topic view) consume the substrate
    over WS API
  - Tier 3A LLM-binary-flag prompt is wired but kept off the default
    hot path; usefulness embedder (MiniLM) carries observation tail
  - Substrate now has 300 events of real work-unit data on disk
    (not enough to launch with, but enough for the visualizers to
    show real shape)

Phase 10.8 (the perf flag) is sitting between us and a credible
launch. We'd rather solve it correctly than band-aid it. Your read
on the three questions above would shape which direction we take.
