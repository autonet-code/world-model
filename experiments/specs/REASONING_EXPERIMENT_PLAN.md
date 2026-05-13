# Reasoning experiment plan

The retrieval-scaling experiment is the only thing currently running
(`run_experiment.py` against `work_units_filtered.jsonl`). It tests
*retrieval at scale* — does the substrate find semantically related
past sessions better as it sees more. That trend is climbing
monotonically (52 → 58 → 66 → 72% so far) and a credible result
will land when the run finishes.

What that experiment does NOT test: whether the substrate, with a
small local LLM as renderer, can produce answers competitive with a
frontier LLM. That's the *reasoning* claim, and it's the one that
matters for the autonet inference economics story.

This document is the plan for that next experiment. Set up, run,
report. No frontier action-replay (too many confounders); a
question-answering contest with three contestants.

## The contest

For each test question about a tightly bounded domain (autonet
substrate code), three contestants produce an answer:

  1. **Frontier-LLM-alone**: Sonnet (or equivalent) with no context
     beyond the question itself. Tests how much the LLM knows from
     general training.

  2. **Frontier-LLM-with-code**: Same Sonnet, given the relevant
     codebase in context (or as much as fits). Tests the upper bound
     — the best the contestant *can* do given full code access.

  3. **Substrate + small LLM**: The substrate locates the relevant
     graph region; a small local LLM (3B–7B class, e.g. Phi-3-mini,
     Qwen-3-7B, or Llama-3.1-8B) renders that region into an answer.
     This is the substrate-real-power contestant.

The frontier LLM also serves as **oracle**: blind-grades all three
answers per question on a 1–5 rubric (correctness, completeness,
accuracy of code references). Blind = doesn't know which contestant
produced each answer.

## Casting

| Role | Model | Why |
|---|---|---|
| Translator (contestant 3) | small local LLM, 3B–7B | Cheap, what the production network would actually deploy |
| Code-context contestant | Sonnet (or GPT-4-class) | Best frontier LLM available with reasonable context window |
| No-context contestant | Same as code-context | Apples-to-apples comparison, same model, removed advantage |
| Oracle / grader | Same as code-context | One frontier model judges; bias toward its own output is the main risk to mitigate |
| Judge agents (training time) | Same as code-context | LLM-generated structured sub-claims posted into the graph during substrate training |

The oracle being the same model as the frontier contestant is a
known bias risk. Mitigation: ask the oracle to grade against an
**external rubric** (correctness against actual code, completeness,
accuracy of references) rather than "is this answer similar to what
I'd say."

## Domain: autonet substrate

Use only the autonet substrate codebase as the bounded domain.
Specifically: `c:\code\autonet\nodes\common\world_model_substrate\`
plus `c:\code\world-model\world_model\generalized\`.

This is a tight semantic surface. ~30 files, ~5 main modules,
recurring concepts (charter, training, mint, gate, locate, render).
A frontier LLM can hold the whole thing in context. The substrate
can be trained on the full set of Claude session traces against this
codebase (the autonet-only sessions from `~/.claude/projects/C--code-autonet`,
plus `C--code-world-model`).

## Question generation

Have the frontier LLM read the codebase and produce ~30 questions
that exercise the substrate's actual content. Categories to cover:

  - Architectural ("Why does X use Y?")
  - Mechanism ("How does the mint gate work?")
  - Tradeoff ("What's the cost of using locate over keyword retrieval?")
  - Code reference ("Where is the substrate's pruning logic?")
  - Conceptual ("What's the difference between novelty and mint?")

Keep ~6 questions per category. Save as `questions.jsonl`, one per
line: `{question, category, expected_modules}`.

## Substrate training for this experiment

The currently-running run trains on broad session traces with NO
judge-agent posts. For the reasoning experiment, training also
includes:

  - For each work unit, an LLM-generated structured sub-claim
    posted into the graph at the relevant region. Each sub-claim
    captures a procedural insight: "this kind of work was done by
    this approach because of this principle."
  - 1-2 sub-claims per work unit. Calls cost $0.01-0.05 per unit
    depending on session length.

This is what builds the *depth structure* the substrate's claim
depends on. Without it, the substrate is just a semantic index;
with it, the substrate has the procedural knowledge that the small
local LLM can render.

## Eval flow

For each question:

  1. Get answer A1 from frontier-LLM-alone.
  2. Get answer A2 from frontier-LLM-with-codebase.
  3. Substrate locate(question) → region. Render the region
     structure as a prompt to the small local LLM. Get answer A3.
  4. Frontier oracle grades all three blind on 1-5 rubric.

Aggregate across 30 questions:
  - Mean score per contestant.
  - Confidence intervals via bootstrap.
  - Paired comparison (A3 vs A2 specifically — the comparison that
    matters for the "substrate matches frontier" claim).

## Statistical significance

30 questions × 3 contestants = 90 graded answers. With 30 paired
observations, even moderate effect sizes (~0.3 standard deviations)
reach significance with paired t-tests at p<0.05. If the effect is
larger, fewer questions suffice; if subtler, we'd need more.

Plan: start with 30. If the result is ambiguous, expand to 60.

## Estimated cost and time

  - Question generation: ~5 min, $0.50.
  - Substrate retraining with judge-agent sub-claims: 30-60 min
    of substrate work + ~$2-5 of frontier API calls.
  - Three-contestant runs: 30 questions × 3 contestants. Frontier
    contestants: ~$3-5. Local LLM: free, ~30 min.
  - Oracle grading: 90 grades × ~2K tokens each = ~$2-3.
  - Analysis and plotting: ~30 min.
  - Total: half day end-to-end, $8-15 of frontier API spend.

## Honest caveats

  - Tests narrow-domain knowledge synthesis, not general reasoning.
    Result holds for "substrate matches frontier within bounded
    domains," not for unbounded generality.
  - Oracle bias toward its own outputs (mitigation: external rubric).
  - Small LLM choice matters. Phi-3-mini is a reasonable starting
    bet; if it fails, try Qwen-3-7B before declaring the substrate's
    rendering can't work with that parameter budget.
  - The substrate's depth structure depends on judge-agent quality.
    The frontier-LLM-as-judge step is doing real work; the result
    is "substrate + frontier-trained graph + small LLM" matching
    "frontier alone with code."

## Files this experiment will produce

  - `questions.jsonl`: questions used.
  - `substrate_with_judges.jsonl`: substrate training events
    including LLM-generated sub-claims.
  - `answers_a1_no_context.jsonl`: contestant 1 outputs.
  - `answers_a2_with_code.jsonl`: contestant 2 outputs.
  - `answers_a3_substrate.jsonl`: contestant 3 outputs.
  - `oracle_grades.jsonl`: blind 1-5 grades per (question, contestant).
  - `reasoning_results.json`: aggregated stats, confidence intervals.

## What success looks like

  - Substrate (A3) score within 0.5 of frontier-with-code (A2),
    significantly above frontier-alone (A1). Substrate is using the
    accumulated graph to compensate for the small LLM's reasoning
    limits.

## What failure looks like

  - Substrate (A3) scores below frontier-alone (A1). The substrate's
    structure isn't compensating for the small LLM's ignorance.
    Either the graph isn't capturing useful procedural knowledge,
    or the small LLM can't render it usefully.
  - A3 within 0.5 of A1 but well below A2: substrate adds little
    value over the small LLM's own knowledge.
