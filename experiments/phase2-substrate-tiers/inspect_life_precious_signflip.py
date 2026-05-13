"""Phase 9.1 diagnostic: where did haiku's life_precious sign flip
come from?

Reads tier3a_llm_cache.jsonl (4-axis, haiku) and tier3b_llm_cache.jsonl
(6-axis, haiku), aligns by turn_obs_id, and reports per-turn deltas
on life_precious. Aggregates the total to confirm the headline
(-1267 -> +5527) and surfaces which turns drove the flip.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
CACHE_4 = HERE / "tier3a_llm_cache.jsonl"
CACHE_6 = HERE / "tier3b_llm_cache.jsonl"


def load_cache(path: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["cache_key"]].append(row)
    return out


def coerce(v: Any) -> int:
    if isinstance(v, bool):
        return 1 if v else -1
    try:
        x = float(v)
        if x > 0.5:
            return 1
        if x < -0.5:
            return -1
        return 0
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("+1", "1", "yes", "true"):
            return 1
        if s in ("-1", "no", "false"):
            return -1
    return 0


def axis_median(samples: List[dict], axis: str) -> int | None:
    valid = []
    for s in samples:
        parsed = s.get("parsed")
        if not isinstance(parsed, dict):
            continue
        if axis not in parsed:
            continue
        valid.append(coerce(parsed[axis]))
    if not valid:
        return None
    return round(median(valid))


def main() -> None:
    cache_4 = load_cache(CACHE_4)
    cache_6 = load_cache(CACHE_6)

    haiku_4: Dict[str, int] = {}
    for key, samples in cache_4.items():
        if not key.startswith("haiku:"):
            continue
        obs_id = key.split(":", 1)[1]
        v = axis_median(samples, "life_precious")
        if v is not None:
            haiku_4[obs_id] = v

    haiku_6: Dict[str, int] = {}
    for key, samples in cache_6.items():
        if not key.startswith("haiku-6:"):
            continue
        obs_id = key.split(":", 1)[1]
        v = axis_median(samples, "life_precious")
        if v is not None:
            haiku_6[obs_id] = v

    common = sorted(set(haiku_4) & set(haiku_6))
    only_4 = sorted(set(haiku_4) - set(haiku_6))
    only_6 = sorted(set(haiku_6) - set(haiku_4))

    print(f"Cache key prefixes seen (4-axis): "
          f"{sorted({k.split(':',1)[0] for k in cache_4})}")
    print(f"Cache key prefixes seen (6-axis): "
          f"{sorted({k.split(':',1)[0] for k in cache_6})}")
    print(f"haiku-4 obs with life_precious: {len(haiku_4)}")
    print(f"haiku-6 obs with life_precious: {len(haiku_6)}")
    print(f"common obs: {len(common)}")
    print(f"only in 4-axis: {len(only_4)}, only in 6-axis: {len(only_6)}")
    print()

    sum_4 = sum(haiku_4.get(o, 0) for o in common)
    sum_6 = sum(haiku_6.get(o, 0) for o in common)
    print(f"sum life_precious (haiku, common turns): "
          f"4-axis={sum_4:+d}  6-axis={sum_6:+d}")
    print()

    # Per-turn delta
    flips: List[tuple[str, int, int]] = []
    for obs_id in common:
        v4 = haiku_4[obs_id]
        v6 = haiku_6[obs_id]
        if v4 != v6:
            flips.append((obs_id, v4, v6))

    print(f"turns where life_precious changed: {len(flips)}/{len(common)}")
    print(f"  +1 -> -1: {sum(1 for _,a,b in flips if a==1 and b==-1)}")
    print(f"  -1 -> +1: {sum(1 for _,a,b in flips if a==-1 and b==1)}")
    print(f"   0 -> +1: {sum(1 for _,a,b in flips if a==0 and b==1)}")
    print(f"  +1 ->  0: {sum(1 for _,a,b in flips if a==1 and b==0)}")
    print(f"   0 -> -1: {sum(1 for _,a,b in flips if a==0 and b==-1)}")
    print(f"  -1 ->  0: {sum(1 for _,a,b in flips if a==-1 and b==0)}")
    print()

    print("All flips, sorted by direction:")
    for obs_id, v4, v6 in sorted(flips, key=lambda r: (r[1], r[2])):
        print(f"  {obs_id}  {v4:+d} -> {v6:+d}")


if __name__ == "__main__":
    main()
