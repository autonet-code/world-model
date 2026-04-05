"""
Test Wikidata-based novelty against epoch data.

Uses Wikidata's knowledge graph to compute novelty scores for
historical events and compares against assigned epoch novelty values.
"""

import json
import math
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from wikidata import compute_wikidata_novelty, best_match


@dataclass
class EpochEvent:
    """An event from the epoch data."""
    name: str
    epoch_id: str
    epoch_name: str
    assigned_novelty: float
    log_novelty: float


def load_epoch_events(path: str) -> list[EpochEvent]:
    """Load events from epoch JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    events = []
    for epoch in data.get('epochs', []):
        epoch_id = epoch.get('id', '')
        epoch_name = epoch.get('name', '')
        novelty_score = epoch.get('noveltyScore', 0)
        log_novelty = math.log10(novelty_score + 1)

        for event in epoch.get('events', []):
            events.append(EpochEvent(
                name=event.get('name', ''),
                epoch_id=epoch_id,
                epoch_name=epoch_name,
                assigned_novelty=novelty_score,
                log_novelty=log_novelty,
            ))

    return events


def test_wikidata_novelty(events: list[EpochEvent], max_events: int = None, use_reference: bool = False):
    """
    Test Wikidata novelty computation against epoch events.

    Args:
        events: List of epoch events to test
        max_events: Optional limit on events to test (for faster iteration)
        use_reference: If True, compute novelty relative to epoch's main concept
    """
    results = []
    errors = []

    # Build epoch reference concepts (use epoch name as reference)
    epoch_references = {}
    current_epoch = None
    for event in events:
        if event.epoch_id not in epoch_references:
            # Use the epoch name as the reference concept
            epoch_references[event.epoch_id] = event.epoch_name

    test_events = events[:max_events] if max_events else events

    print(f"Testing {len(test_events)} events...")
    if use_reference:
        print("(Using epoch-relative novelty)")
    print("=" * 70)

    for i, event in enumerate(test_events):
        print(f"\n[{i+1}/{len(test_events)}] {event.name}")
        print(f"  Epoch: {event.epoch_name} (assigned novelty: {event.assigned_novelty:,.0f})")

        # Check if we can find it in Wikidata
        match = best_match(event.name)
        if not match:
            print(f"  ERROR: Not found in Wikidata")
            errors.append((event.name, "Not found"))
            continue

        qid, label, desc = match
        print(f"  Wikidata: {label} ({qid})")

        # Compute novelty (with or without reference)
        if use_reference:
            reference = epoch_references.get(event.epoch_id)
            result = compute_wikidata_novelty(event.name, reference_text=reference)
        else:
            result = compute_wikidata_novelty(event.name)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            errors.append((event.name, result['error']))
            continue

        computed = result['composite']
        print(f"  Computed novelty: {computed:.3f}")
        print(f"    Integration resistance: {result['components']['integration_resistance']:.3f}")
        print(f"    Contradiction depth: {result['components']['contradiction_depth']:.3f}")
        print(f"    Coverage gap: {result['components']['coverage_gap']:.3f}")
        print(f"    Allocation disruption: {result['components']['allocation_disruption']:.3f}")

        results.append({
            'event': event.name,
            'epoch_id': event.epoch_id,
            'assigned': event.assigned_novelty,
            'log_assigned': event.log_novelty,
            'computed': computed,
            'components': result['components'],
            'qid': result['qid'],
        })

    return results, errors


def analyze_results(results: list[dict]) -> dict:
    """Analyze correlation between computed and assigned novelty."""
    if len(results) < 2:
        return {'error': 'Not enough results to analyze'}

    # Group by epoch for epoch-level correlation
    by_epoch = {}
    for r in results:
        epoch = r['epoch_id']
        if epoch not in by_epoch:
            by_epoch[epoch] = []
        by_epoch[epoch].append(r)

    # Compute epoch averages
    epoch_computed = []
    epoch_log_assigned = []

    for epoch_id, epoch_results in by_epoch.items():
        avg_computed = statistics.mean(r['computed'] for r in epoch_results)
        log_assigned = epoch_results[0]['log_assigned']
        epoch_computed.append(avg_computed)
        epoch_log_assigned.append(log_assigned)

    # Pearson correlation
    n = len(epoch_computed)
    if n < 2:
        correlation = 0.0
    else:
        mean_x = statistics.mean(epoch_computed)
        mean_y = statistics.mean(epoch_log_assigned)

        numerator = sum((x - mean_x) * (y - mean_y)
                       for x, y in zip(epoch_computed, epoch_log_assigned))

        var_x = sum((x - mean_x) ** 2 for x in epoch_computed)
        var_y = sum((y - mean_y) ** 2 for y in epoch_log_assigned)

        denominator = math.sqrt(var_x * var_y)
        correlation = numerator / denominator if denominator > 0 else 0.0

    # Event-level stats
    all_computed = [r['computed'] for r in results]

    return {
        'events_tested': len(results),
        'epochs_tested': len(by_epoch),
        'correlation': correlation,
        'computed_mean': statistics.mean(all_computed),
        'computed_stdev': statistics.stdev(all_computed) if len(all_computed) > 1 else 0,
        'computed_range': (min(all_computed), max(all_computed)),
        'epoch_averages': {
            epoch_id: statistics.mean(r['computed'] for r in epoch_results)
            for epoch_id, epoch_results in by_epoch.items()
        },
    }


def print_report(results: list[dict], analysis: dict, errors: list):
    """Print a formatted report."""
    print("\n" + "=" * 70)
    print("WIKIDATA NOVELTY TEST REPORT")
    print("=" * 70)

    print(f"\nEvents tested: {analysis['events_tested']}")
    print(f"Epochs tested: {analysis['epochs_tested']}")
    print(f"Errors: {len(errors)}")

    print(f"\nComputed novelty statistics:")
    print(f"  Range: {analysis['computed_range'][0]:.3f} - {analysis['computed_range'][1]:.3f}")
    print(f"  Mean: {analysis['computed_mean']:.3f}")
    print(f"  Stdev: {analysis['computed_stdev']:.3f}")

    print(f"\nCorrelation with log-novelty: {analysis['correlation']:.4f}")

    if analysis['correlation'] > 0.7:
        print("  --> STRONG positive correlation")
    elif analysis['correlation'] > 0.4:
        print("  --> MODERATE positive correlation")
    elif analysis['correlation'] > 0:
        print("  --> WEAK positive correlation")
    else:
        print("  --> NO correlation (or negative)")

    print("\nEpoch averages:")
    for epoch_id, avg in sorted(analysis['epoch_averages'].items()):
        print(f"  {epoch_id}: {avg:.3f}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, error in errors[:10]:
            print(f"  - {name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print("\n" + "=" * 70)


def main():
    # Load epoch data
    data_path = Path(r"C:\code\sidekick-web\assets\novelty_1.json")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading epoch data from {data_path}")
    events = load_epoch_events(data_path)
    print(f"Loaded {len(events)} events")

    # Run tests (limit to first 20 for speed during development)
    results, errors = test_wikidata_novelty(events, max_events=25)

    # Analyze
    if results:
        analysis = analyze_results(results)
        print_report(results, analysis, errors)

    return results, errors


if __name__ == "__main__":
    main()
