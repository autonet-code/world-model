#!/usr/bin/env python3
"""Phase 6 orchestrator: run substrate build + contest for each N in {2,4,6,8,10}."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
N_VALUES = [2, 4, 6, 8, 10]


def main() -> int:
    for n in N_VALUES:
        print(f"\n========== N = {n} ==========")
        snap_path = HERE / f"substrate_N{n}.json"
        contest_path = HERE / f"contest_N{n}.jsonl"

        # Build substrate snapshot.
        r = subprocess.run(
            [sys.executable, str(HERE / "build_substrate.py"),
             "--n", str(n), "--out", str(snap_path)],
            cwd=str(HERE),
        )
        if r.returncode != 0:
            print(f"  build_substrate failed for N={n}")
            return r.returncode

        # Contest.
        r = subprocess.run(
            [sys.executable, str(HERE / "run_contest.py"),
             "--n", str(n), "--out", str(contest_path)],
            cwd=str(HERE),
        )
        if r.returncode != 0:
            print(f"  run_contest failed for N={n}")
            return r.returncode

    # Aggregate.
    print("\n========== AGGREGATE ==========")
    aggregate = []
    for n in N_VALUES:
        rows = []
        contest_path = HERE / f"contest_N{n}.jsonl"
        if not contest_path.exists():
            continue
        for line in contest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        rag_mean = sum(r["rag"]["score"] for r in rows) / max(len(rows), 1)
        sub_mean = sum(r["substrate"]["score"] for r in rows) / max(len(rows), 1)
        aggregate.append({
            "n": n, "n_test": len(rows),
            "rag_mean": rag_mean,
            "substrate_mean": sub_mean,
            "delta": sub_mean - rag_mean,
        })
        print(f"  N={n:>2}  rag={rag_mean:.3f}  sub={sub_mean:.3f}  delta={sub_mean-rag_mean:+.3f}  (n_test={len(rows)})")

    (HERE / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\n  aggregate -> {HERE / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
