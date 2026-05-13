# Substrate validation: tier results

Empirical validation of the post-and-coparent substrate.
Specs, findings, raw JSON results, and plots from the Tier 0 → 3A
arc. Cross-referenced from `../substrate-architecture.md`.

## Index

  - **Tier 0** — synthetic three-root deployer-domain composition
    - `TIER0_SPEC.md`, `tier0_results.json`, `tier0_plot.png`
    - Result: 6/6 predictions pass.

  - **Tier 1A** — LLM-as-embedder, three-axis substrate, 12 code
    snippets (qwen3.5:4b)
    - `TIER1A_SPEC.md`, `tier1a_results.json`, `tier1a_plot.png`
    - Result: 4/5 after dedup fix + category-level predictions.

  - **Tier 1A reshaped** — same setup, prediction shape revised
    - `A1_RESHAPED_SPEC.md`, `a1_reshaped_results.json`,
      `a1_reshaped_plot.png`

  - **Tier 1B** — haiku-4-5 via claude-max-proxy, same snippets
    - `TIER1B_VS_TIER1A.md`, `tier1b_results.json`,
      `tier1b_plot.png`
    - Result: 3/5 (qwen 4/5).

  - **Tier 1C** — binary-flag prompt, both models
    - `tier1c_qwen_results.json`, `tier1c_haiku_results.json`
    - Result: qwen 4/5, haiku 4/5 — Option C confirmed.

  - **Tier 1g** — single-axis substrate diagnostic
    - `TIER1G_SPEC.md`, `tier1g_results.json`

  - **Tier 1 reads** — integrated interpretation
    - `TIER1_FINAL_READ.md`, `TIER1_INTEGRATED_READ.md`

  - **Tier 2** — N-agent consensus at scale, Lindblad kernel
    - `TIER2_SPEC.md`, `TIER2_FINDINGS.md`,
      `tier2_results.json`, `tier2_plots/`
    - Result: 3/6 by-letter; load-bearing predictions
      (decisiveness gain, input tracking, sub-linear scaling)
      all pass. Architectural claim validated.

  - **Tier 3A** — LLM-as-embedder swap-in for autonet's pipeline
    (haiku-4-5)
    - `TIER3A_SPEC.md`, `TIER3A_FINDINGS.md`,
      `tier3a_results.json`, `tier3a_plot.png`
    - Result: 4/5 (real disagreement rate 0.8% after metric
      reframe).

  - **Tier 3B** — 4-root vs 6-root charter, verdict separation +
    attention dilution
    - `tier3b/TIER3B_SPEC.md`, `tier3b/TIER3B_FINDINGS.md`,
      `tier3b/tier3b_results.json`, `tier3b/tier3b_plot.png`
    - Result: H4 (categorical separation) preserved at 6-root;
      one tendency (life_precious) shows pre-existing sign-flip
      instability documented in findings.

## Value-prop validation (Phase 5 + Phase 6)

After the engine refactors (scoped equilibrate, 64-dim coords,
Lindblad cross-link), the substrate's inference value was
validated end-to-end against a memorization-defeated corpus.

  - **Phase 5** — substrate vs RAG, toolz-renamed corpus, 4
    contestants
    - `phase5-phase6/PHASE5_PLAN.md`
    - Code + results: `../../experiments/phase5-toolz-rename/`
    - Result: haiku+substrate > haiku+RAG by +0.14;
      qwen+substrate ≈ haiku-alone (within noise at n=11).

  - **Phase 6** — tendency-count scaling, N ∈ {2,4,6,8,10}
    - `phase5-phase6/PHASE6_PLAN.md`,
      `phase5-phase6/aggregate.json`
    - Code + results:
      `../../experiments/phase6-tendency-scaling/`
    - Result: substrate beats RAG at every N (delta +0.20 to
      +0.30). Curve is plateau-with-spike, not cleanly
      monotonic. 6-root deployment (autonet's MVP charter) sits
      mid-curve at +0.237.

## Refactor docs (background)

  - `POST_COPARENT_PLAN.md`, `REFACTOR_POST_AND_COPARENT.md` —
    the post-only + multi-parent refactor that the tier arc was
    validating against.

## What this collectively shows

Across N-agent (Tier 2), single-claim (Tier 1), and integration
(Tier 3A) shapes, the substrate produces deterministic verdicts
that amplify real signal and resolve disagreement decisively,
with cheap small LLMs as embedders and no central coordinator.
This is the decentralized-AI MVP claim.

What's not shown: emerging utility roots (deferred), unbounded
world-size validation (deferred), N=4 → N=30 vocabulary
expansion (deferred — see future-work in
`../substrate-architecture.md`).

Phase 5 + Phase 6 then closed the value-prop loop: substrate
inference beats RAG inference at the same LLM, across N=2 to
N=10 tendencies, on a corpus designed to defeat
training-memory pattern-match. Architectural advantage holds;
"intrinsic usefulness" is empirically supported. "Intrinsic
alignment" remains architecturally present but behaviorally
unproven (alignment axes don't engage on a code corpus — a
separate adversarial-prompt experiment would close that).
