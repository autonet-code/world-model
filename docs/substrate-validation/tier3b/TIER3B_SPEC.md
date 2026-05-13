# Tier 3B spec: do extra usefulness roots produce sharper substrate verdicts?

## Question

Does adding `correctness` and `simplicity` as roots — on top of the
4 charter alignment roots — produce measurably sharper substrate
signal on autonet's corpus than the 4-root baseline?

This is the reframed Tier 3A. Tier 3A asked "does the LLM
embedder improve over the keyword heuristic." Tier 3B asks "does
the extended root set improve over the charter-only set, holding
the embedder fixed (haiku-4-5)."

## Setup

  - Corpus: same 30 turns as Tier 3A (`tier3a_corpus.py`)
  - Embedder: haiku-4-5 via claude-max-proxy (Tier 3A's setup,
    binary-flag prompt extended to 6 axes)
  - Arm A (baseline): 4 charter roots, charter-only prompt
  - Arm B (experiment): 4 charter + `correctness` + `simplicity`
    roots, 6-axis prompt
  - Equilibrate after each turn, snapshot per-axis scores + root
    scores

## Hypotheses (load-bearing predictions)

  - **H1 (signal density)**: Arm B commits non-zero on more
    (turn, axis) pairs than Arm A on the same corpus, *especially*
    for `capability_improving` and `reasoning_heavy` turns where
    the LLM should detect correctness/simplicity signal.
    Operationalized: `n_committed_pairs(B) > n_committed_pairs(A)`
    when restricted to the 4 axes both arms share.
    
    *Test:* compute, for the 4 axes shared between arms, the
    fraction of pairs that commit non-zero in each arm. Arm B
    should be ≥ Arm A on shared axes (otherwise we're losing
    signal somewhere). Then compute the additional signal
    captured on the 2 new axes.

  - **H2 (root-score correlation)**: in Arm B, `correctness`
    and `simplicity` posts should correlate moderately (not
    perfectly) with charter-axis posts. Perfect correlation
    means the new roots are renaming `intelligence` /
    `evolution`. Zero correlation means the LLM can't grade
    them. Moderate correlation (0.3–0.7) means they're capturing
    independent-but-related signal.

    *Test:* compute Pearson correlation between
    `correctness_score` and each charter axis across the 30
    turns. Same for `simplicity`.

  - **H3 (verdict separation)**: Arm B should produce wider
    spread between top-mint and median-mint nodes than Arm A.
    
    *Test:* compute, for each arm, the post-equilibrate
    distribution of root scores across nodes. Standard deviation
    of node-level scores should be ≥ Arm A.

  - **H4 (categorical separation)**: Arm B should distinguish
    `capability_improving` turns from `reasoning_heavy` turns
    more clearly than Arm A, because correctness + simplicity
    differentiate "good code work" from "good explanation work."
    
    *Test:* mean root-score vector for each category. The
    distance between the `capability_improving` vector and the
    `reasoning_heavy` vector should be larger in Arm B than
    Arm A.

## Pass criteria

  - 3/4 hypotheses validated (allowing one to fail without
    invalidating the architecture-level claim).
  - H2 must validate (otherwise the new roots are just renaming
    or noise — both bad outcomes).

## Cost estimate

  - 30 turns × 1 LLM call per arm × 2 arms = 60 calls. With
    haiku via the proxy, ~3-5 minutes wall time per arm.
  - The Arm A cache exists from Tier 3A; we reuse it.

## Out of scope

  - Heuristic baseline comparison (already done in Tier 3A).
  - qwen comparison (already shown identical verdicts to haiku
    under binary prompts in Tier 1C).
  - Modifying autonet's actual `adapter.py` — this experiment
    runs on a self-contained 6-root world.
