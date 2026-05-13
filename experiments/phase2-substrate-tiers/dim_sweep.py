#!/usr/bin/env python3
"""Phase 2.1+2.2: PCA dim sweep for the embedding tail.

Question: MiniLM emits 384-dim embeddings (not 1024 — that figure in
POST_AUTONET_FINDINGS.md predates the model swap). At which target
dim does category-separation collapse?

Approach: take the per-project work_units_*.jsonl corpora (~240
labeled work units across 6 projects), embed each via MiniLM, fit
PCA on the matrix, project to {384, 256, 128, 64, 32, 16, 8}. At
each dim, measure category-separation via the same metric Tier 3B
used (H4): mean-vector distance between pairs of category clusters.

The "categories" here are projects (autonet, world-model, dao,
matrix, hackathon, research). Inter-project distance is a proxy for
"does the substrate's coord space distinguish substantively different
work."

Output: dim_sweep_results.json + console table. Recommendation =
smallest dim where mean pairwise category distance stays >= 85% of
the native-dim baseline.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent  # substrate_experiment/
RESULTS_PATH = HERE / "dim_sweep_results.json"


def load_work_units() -> Tuple[List[str], List[str]]:
    """Returns (texts, labels) where label = project name."""
    texts: List[str] = []
    labels: List[str] = []
    for path in sorted(EXP_ROOT.glob("work_units_*.jsonl")):
        if path.name in ("work_units_all.jsonl", "work_units_filtered.jsonl"):
            continue
        project = path.stem.removeprefix("work_units_")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                problem = (row.get("problem") or "")[:400]
                resolution = (row.get("resolution") or "")[:800]
                if not problem and not resolution:
                    continue
                texts.append(f"{problem} | {resolution}")
                labels.append(project)
    return texts, labels


def category_mean(matrix: np.ndarray, labels: List[str], category: str) -> np.ndarray:
    mask = np.array([l == category for l in labels])
    if not mask.any():
        return np.zeros(matrix.shape[1])
    return matrix[mask].mean(axis=0)


def pairwise_category_distances(matrix: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """All pairwise mean-vector distances between distinct categories."""
    cats = sorted(set(labels))
    out: Dict[str, float] = {}
    for i, ci in enumerate(cats):
        mi = category_mean(matrix, labels, ci)
        for cj in cats[i + 1:]:
            mj = category_mean(matrix, labels, cj)
            d = float(np.linalg.norm(mi - mj))
            out[f"{ci}__vs__{cj}"] = d
    return out


def mean_intra_category_spread(matrix: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Per-category mean distance from members to their centroid."""
    cats = sorted(set(labels))
    out: Dict[str, float] = {}
    for c in cats:
        mask = np.array([l == c for l in labels])
        if mask.sum() < 2:
            out[c] = 0.0
            continue
        sub = matrix[mask]
        centroid = sub.mean(axis=0)
        out[c] = float(np.mean(np.linalg.norm(sub - centroid, axis=1)))
    return out


def main() -> int:
    texts, labels = load_work_units()
    cats = sorted(set(labels))
    counts = {c: labels.count(c) for c in cats}
    print(f"  corpus: {len(texts)} work units across {len(cats)} projects")
    for c in cats:
        print(f"    {c:>14}: {counts[c]}")

    print("  loading MiniLM (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("  encoding...")
    embeddings = model.encode(texts, convert_to_numpy=True,
                              show_progress_bar=False, batch_size=32)
    base_dim = embeddings.shape[1]
    print(f"  embeddings: {embeddings.shape} (native dim = {base_dim})")

    pca_ceiling = min(embeddings.shape) - 1  # PCA cap
    target_dims = [base_dim, 256, 128, 64, 32, 16, 8]
    target_dims = sorted({d for d in target_dims if 1 <= d <= base_dim}, reverse=True)

    # Baseline metrics at native dim (no PCA).
    baseline_pairwise = pairwise_category_distances(embeddings, labels)
    mean_pairwise_baseline = sum(baseline_pairwise.values()) / max(len(baseline_pairwise), 1)
    print(f"  baseline mean pairwise category distance "
          f"(dim={base_dim}): {mean_pairwise_baseline:.4f}")

    results: Dict[str, Any] = {
        "base_dim": base_dim,
        "n_items": len(texts),
        "categories": cats,
        "per_category_count": counts,
        "by_dim": {},
    }

    print()
    print(f"  {'dim':>5}  {'mean cat-dist':>16}  {'retained vs base':>18}  "
          f"{'mean intra-spread':>18}")
    print(f"  {'-' * 5}  {'-' * 16}  {'-' * 18}  {'-' * 18}")

    for dim in target_dims:
        if dim == base_dim:
            projected = embeddings
        elif dim > pca_ceiling:
            # Skip dims we can't reach via PCA (n_components must be
            # <= min(n_samples, n_features) - 1)
            continue
        else:
            pca = PCA(n_components=dim, random_state=0)
            projected = pca.fit_transform(embeddings)

        pairwise = pairwise_category_distances(projected, labels)
        intra = mean_intra_category_spread(projected, labels)
        mean_pairwise = sum(pairwise.values()) / max(len(pairwise), 1)
        retained = mean_pairwise / mean_pairwise_baseline if mean_pairwise_baseline > 0 else 0.0
        mean_intra = sum(intra.values()) / max(len(intra), 1)

        print(f"  {dim:>5}  {mean_pairwise:>16.4f}  {retained:>17.1%}  "
              f"{mean_intra:>18.4f}")

        results["by_dim"][str(dim)] = {
            "mean_pairwise_distance": mean_pairwise,
            "retained_vs_baseline": retained,
            "all_pairwise": pairwise,
            "intra_spread": intra,
            "mean_intra_spread": mean_intra,
        }

    # Recommendation: smallest dim where retained >= 0.85.
    candidates = sorted(
        (int(d), results["by_dim"][d]["retained_vs_baseline"])
        for d in results["by_dim"]
    )
    chosen = None
    for d, retained in candidates:
        if retained >= 0.85:
            chosen = d
            break
    if chosen is None:
        chosen = max(d for d, _ in candidates)
    results["recommended_dim"] = chosen

    print()
    print(f"  Recommendation: target_dim = {chosen} (smallest dim where "
          f"mean pairwise distance retention >= 85% of native)")

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  results -> {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
