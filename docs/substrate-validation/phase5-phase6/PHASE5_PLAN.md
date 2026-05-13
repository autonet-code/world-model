# Phase 5: substrate value-prop validation

## What this validates

Two independent questions, four contestants, one corpus.

**Q1 — architecture vs RAG.** Does substrate composition produce
better inference than retrieval alone, holding the LLM fixed?
Comparison: `haiku+substrate` vs `haiku+RAG`.

**Q2 — stack viability.** Does substrate compensate for LLM size?
Comparison: `qwen+substrate` vs `haiku-alone`.

Pass = unambiguous direction on both. Either direction is a real
result; the goal is signal, not validation.

## Why this design

  - **Corpus = renamed toolz subset.** Pure-Python functional
    combinators with rich docstrings + doctest examples.
    Identifiers and signatures renamed to defeat training-memory
    pattern-match (qwen and haiku have likely seen toolz). Concepts
    stay; names are alien.

  - **Sparse inference prompts.** Training-time entries have full
    docstrings. Inference-time queries have one-line summaries plus
    one or two I/O examples. The substrate's compounding value is
    visible only when the inference prompt is insufficient on its
    own.

  - **Sonnet as training judge.** Each training function gets multi-
    axis sub-claims posted into the substrate. Cost on bridge is
    zero; we use the best judge available.

  - **2-tendency world: correctness + simplicity.** Cleanest
    isolation. The four alignment tendencies don't apply to a
    library-implementation task.

  - **Doctest-based correctness scoring.** Each held-out function
    has a few `>>> input → output` lines. Score = fraction of
    doctests that pass when the contestant's implementation is
    substituted.

## Contestants (concrete plumbing)

  - **a1 haiku-alone.** Receives sparse prompt only. No retrieval,
    no substrate.

  - **a2 haiku+RAG.** Receives sparse prompt + top-k similar
    training functions (full docstrings + impls), retrieved by
    cosine similarity over MiniLM embeddings of the sparse prompt.

  - **a3 haiku+substrate.** Receives sparse prompt + output of
    `infer_with_world_model(mode="general")` on the prompt. The
    substrate-rendered region carries the same training corpus but
    surfaces it through the production probe path (locate + render).

  - **a4 qwen+substrate.** Identical to a3 except the renderer LLM
    is qwen3.5:4b via local Ollama.

## Source corpus

`toolz.functoolz` + `toolz.itertoolz`. Selection criteria per
function:

  - Has a docstring with at least one `>>>` example.
  - Pure Python.
  - Implementation ≤ 40 lines.
  - Behavior fully captured by the doctest examples (so passing the
    examples is a clean correctness signal).

Target ~50 functions total, split 40 train / 10 test.

## Rename strategy

For each function:

  - Function name → a synonym at the concept level
    (`compose → fold_funcs`, `curry → partialize`, `pipe → thread`,
    etc.)
  - Parameter names → generic equivalents
    (`func → action`, `*funcs → *steps`, `x → value`).
  - Docstring rewritten to use the renamed identifiers.
  - Doctest examples rewritten to call the renamed function.
  - Implementation rewritten with the new names.

The rewrite is mechanical; no semantic change. We persist the
mapping in `rename_map.json` so all artifacts use the same
identifiers consistently.

## Substrate construction

For each renamed train function:

  1. Build observation. Coords = MiniLM embedding of
     `f"{name}\n\n{docstring}\n\n{impl}"`, projected to dim=64
     via the same usefulness embedder autonet uses.
  2. Submit to the substrate.
  3. Sonnet judge call: read the function, emit per-axis sub-claims.
     One claim minimum per axis (correctness, simplicity); zero
     when the axis doesn't apply.
  4. Each emitted sub-claim is sprouted as a child of the
     observation node, under the matching tendency, with its own
     embedded coords.

This is the same shape as `train_with_judges.py` but on the renamed
toolz corpus instead of work_units_all.

## Inference test (per held-out function)

  1. Build sparse prompt:
     ```
     def {renamed_name}({params}):
         """{one-line summary}
         >>> {example1}
         {expected1}
         >>> {example2}
         {expected2}
         """
     ```
     The `{one-line summary}` is a single short sentence; the
     implementation body is left for the contestant to produce.

  2. Each contestant receives this prompt (a2/a3/a4 also receive
     retrieved/probed context). Each returns a Python function body.

  3. Test harness substitutes the body into the function, runs the
     full doctest example set (not just the 1-2 shown to the
     contestant). Score = (passed examples) / (total examples).

## Reporting

```
contestant       mean_score   pass_at_1
haiku alone      0.X          N/10
haiku + RAG      0.X          N/10
haiku + substr.  0.X          N/10
qwen  + substr.  0.X          N/10
```

Q1 verdict = sign of `haiku+substrate - haiku+RAG`.
Q2 verdict = sign of `qwen+substrate - haiku-alone`.

## Cost

  - Sonnet training calls: 40 functions × 1-2 judge calls = ~50
    bridge calls, ~5-10 min wall time.
  - Inference calls: 10 functions × 4 contestants = 40 calls.
    Haiku via bridge for a1/a2/a3 (~6 min), qwen local for a4
    (~5 min).
  - Test harness: instant.

Total: ~20-30 min wall time once implementation is done.

## Out of scope

  - Multi-N substrate-size scaling. We test at one N (= 40 train).
  - Federation. Single-daemon substrate. (Federation is orthogonal
    to the value-prop question — addressed by the existing Tier 2
    consensus result, separately.)
  - The four alignment tendencies. This corpus doesn't engage them.
