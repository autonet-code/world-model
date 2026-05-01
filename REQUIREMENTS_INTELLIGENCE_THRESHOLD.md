# Requirements: Intelligence Threshold for Replacing Frontier LLMs

**Status:** open research thread.
**Owner:** to be picked up by an agent.
**Parent context:** the world-model engine is intended to eventually replace
the centralized LLM as the reasoner in a multi-node agent framework. The
LLM is a bootstrap I/O codec, not the brain. Before we build the bridge,
the engine has to demonstrate it can be *competitive* — measured in the
unit users actually care about: cost per task done, at task quality
comparable to Opus 4.7 / GPT-5.5.

---

## The economic threshold

Users don't pay for tokens, they pay per outcome. A task that costs $5 on
Opus and $5 on the network is a wash on price; the network only wins if
it costs **less** for the same task quality, or substantially better at
the same price.

So the question this research thread has to answer is:

> Given the world-model engine's architecture, is there a tractable size
> at which it produces task-quality comparable to a frontier LLM, at a
> cost-per-task lower than the LLM's API price?

Two metrics, plotted together:

1. **Task quality** vs LLM baseline on the same task (some calibrated
   measure: pass@k, accuracy, alignment to ground truth, structured-
   output validity, etc — pick what the task allows).
2. **Cost per converged equilibration** at that engine size — wall-clock
   on a fixed reference machine OR FLOPs from the inner loop, expressed
   in $ via cloud-GPU hourly rates.

Plot the curve as engine state size grows. The LLM cost curve is roughly
flat per task (it's a fixed model, fixed price). The engine's curve
trends upward (bigger state, more iteration). Find where they cross.

- If they cross at a state size we can plausibly build and run
  (ballpark: under 10^9 tendencies, under several hundred ms per task on
  a reasonable GPU), the bet is alive.
- If they cross at intractable sizes (10^12 tendencies, minutes per
  task), the architecture is wrong as currently shaped.

A "no, this doesn't cross at any tractable size" answer is just as
valuable as a "yes, it crosses at X." Don't push toward yes; report
what's there.

## What "competitive task quality" means concretely

Don't anchor on iris/digits. Those exercise the substrate; they don't
test reasoning. The threshold benchmark needs to be something where:

- An LLM can be run head-to-head as the baseline.
- The task involves multi-step reasoning, not single-shot classification.
- Quality can be measured automatically.

Reasonable candidates (not exhaustive — pick what the engine can
actually be wired to):

- **Mini ARC-style symbolic reasoning** (input/output grids, infer
  rule, apply). The grid relations map naturally onto graph state.
- **Simple programmatic reasoning** — given a few examples, produce
  the next term of a sequence; classify a new instance under an inferred
  rule.
- **Multi-step arithmetic word problems** at small scale (math, GSM8K
  shorter end).
- **Structured logic puzzles** (Knights and Knaves, Einstein-type)
  where the constraint graph is the natural representation.

The engine has to reach **comparable accuracy to a Sonnet- or Haiku-
class LLM** on the chosen task, not Opus-class. The bet isn't "beat
Opus from scratch"; it's "be cheap enough that Opus-quality work shifts
to the network for cost reasons later." Reaching Sonnet/Haiku on a
constrained reasoning task is the interesting first crossing point.

## What's NOT in scope for this thread

- Distributed consensus across nodes — separate problem, downstream of
  this one. Don't worry about peer integration yet.
- The LLM bridge — also downstream. The engine has to prove it carries
  useful reasoning *internally* before we wire LLM I/O.
- Personality modeling — the historical README framing. The engine
  here is being evaluated as a general substrate, not as an Andrei-twin.
- Beating GPT-5.5 / Opus 4.7. Stop short of frontier; aim for the
  *second tier* (Sonnet / Haiku) as the first crossing point.

## What the agent picking this up should produce

A document at the end with:

1. **Task choice and rationale** — what benchmark you settled on and
   why it's a fair test of reasoning that the engine is capable of
   representing.
2. **Architecture iterations attempted** — what you tweaked
   (tendency count, graph density, novelty calibration, contraction-map
   parameters, multi-LOD use, lineage-as-feedback, etc) and the
   measured impact on quality.
3. **Quality-vs-cost curve** — the actual numbers at increasing
   engine sizes, with the LLM baseline marked on it.
4. **Honest verdict** — does the curve suggest a tractable crossing,
   does it plateau short of useful quality, does it diverge in cost
   faster than quality grows? One paragraph, plain language.
5. **What a follow-up project should change** — if the verdict is
   "not yet," what concrete architectural change is most likely to
   move the curve. (E.g., "denser graphs help quality but blow cost,
   need sparse-attention variant"; or "novelty calibration is
   bottlenecking convergence, need a different mechanism.")

## Style notes

- This is research, not a demo. Don't add bolt-on logic to game a
  benchmark. The engine's existing operations
  (`reseed_and_equilibrate`, substitutions, lineage, LOD readout) are
  the surface; iterations should be on parameters and graph structure,
  not new code paths that only fire for one task.
- Negative results are fine. A clean "this architecture caps out at
  Y% on task X regardless of size" is a real contribution.
- Don't push past the second-tier threshold. If you hit Sonnet-comp
  at competitive cost, stop and report. Frontier-comp is a separate
  research effort.

## Reasonable confidence to declare "yes, build the bridge"

Three conditions, all met:

1. On a multi-step reasoning task, the engine reaches **≥80% of a
   Sonnet-class LLM's accuracy** on a held-out test set of at least 100
   instances.
2. At the engine size that achieves (1), per-task wall-clock on a
   reference GPU (e.g., one A100) is **≤ Sonnet's per-task latency** for
   that task on the same hardware (or ≤ 2x slower if cost-per-task
   is meaningfully lower).
3. The architecture iterations leading to (1) and (2) didn't rely on
   task-specific tricks — i.e., the same engine config also performs
   reasonably (within 30% of (1)) on at least one different reasoning
   task without retuning.

If all three hold, the substrate has a real chance and the bridge is
worth building. If any fails, report which one and why.

If none of the three are reachable inside the iteration budget you
allow yourself, return that finding too. The point is information.
