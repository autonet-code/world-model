"""
Test Harness - Validate novelty model against historical epoch data.

Loads epoch data from sidekick-web's novelty_1.json and tests our model's
ability to compute novelty scores that correlate with the assigned values.

Test Strategy:
1. Build a world model from each epoch's events and description
2. For each event in epoch N, compute novelty against epoch N-1's world model
3. Compare our computed scores to the assigned epoch novelty scores
4. Measure correlation and report findings
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import statistics

# Our novelty system
from core import Concept
from world_model.novelty import (
    WorldModelReference,
    AnchoredNoveltyMeasure,
    compute_novelty,
)
from world_model import (
    Observation,
    ObservationStore,
    AgentSet,
    Tendency,
    Tree,
    TreeStore,
    Node,
    Position,
)

# Try to import embeddings, fall back to simple similarity if unavailable
try:
    from embeddings import relation_fit_score, preload_cache
    USE_NEURAL = True
    print("Using neural similarity (embeddings + NLI)")
except ImportError:
    USE_NEURAL = False
    print("Neural models not available, using word overlap similarity")


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class Event:
    """A historical event from the epoch data."""
    name: str
    time: str
    details: str  # Wikipedia URL

    @property
    def year(self) -> float:
        """Extract year from time string."""
        try:
            # Handle various formats: "-13800000000-01-01", "2022-11-30", etc.
            if self.time.startswith("-"):
                # Negative year (BCE)
                parts = self.time[1:].split("-")
                return -float(parts[0])
            else:
                parts = self.time.split("-")
                return float(parts[0])
        except:
            return 0.0


@dataclass
class Epoch:
    """An epoch from the novelty timeline."""
    id: str
    name: str
    time_start: float
    time_end: float
    novelty_score: float
    description: str
    events: list[Event] = field(default_factory=list)

    @property
    def log_novelty(self) -> float:
        """Log-scaled novelty for comparison."""
        return math.log10(self.novelty_score + 1)


def load_epoch_data(path: str) -> list[Epoch]:
    """Load epochs from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    epochs = []
    for e in data.get("epochs", []):
        events = [
            Event(
                name=ev.get("name", ""),
                time=ev.get("time", ""),
                details=ev.get("details", ""),
            )
            for ev in e.get("events", [])
        ]

        epoch = Epoch(
            id=e.get("id", ""),
            name=e.get("name", ""),
            time_start=e.get("timeStart", 0.0),
            time_end=e.get("timeEnd", 0.0),
            novelty_score=e.get("noveltyScore", 0.0),
            description=e.get("description", ""),
            events=events,
        )
        epochs.append(epoch)

    return epochs


# =============================================================================
# World Model Building
# =============================================================================

def build_world_model_from_epoch(epoch: Epoch, prior_epochs: list[Epoch] = None) -> WorldModelReference:
    """
    Build a world model representing knowledge up to and including this epoch.

    The world model includes:
    - Observations from all events in this and prior epochs
    - Trees representing the major themes/values of the epoch
    - Default agent allocations (could be customized per epoch)
    """
    observations = ObservationStore()
    trees = TreeStore()

    # Collect all events from prior epochs + this one
    all_epochs = (prior_epochs or []) + [epoch]

    for ep in all_epochs:
        # Add epoch description as an observation
        if ep.description:
            obs = Observation(content=ep.description[:280])
            observations.add(obs)

        # Add each event as an observation
        for event in ep.events:
            obs = Observation(content=f"{event.name}")
            observations.add(obs)

    # Create a tree representing the epoch's core theme
    tree = Tree(root_value=epoch.name)
    root = tree.root_node

    # Add description as a supporting node
    if epoch.description:
        desc_node = Node(content=epoch.description[:200])
        desc_node.add_stake("curiosity", 0.3)
        desc_node.add_stake("meaning", 0.3)
        root.add_child(desc_node, Position.PRO)

    # Add events as supporting nodes
    for event in epoch.events:
        event_node = Node(content=event.name)
        event_node.add_stake("curiosity", 0.2)
        root.add_child(event_node, Position.PRO)

    trees.add(tree)

    # Default agent allocations
    agents = AgentSet()

    return WorldModelReference(trees, agents, observations)


# =============================================================================
# Novelty Computation
# =============================================================================

@dataclass
class NoveltyTestResult:
    """Result of testing novelty for one event."""
    event_name: str
    source_epoch: str
    reference_epoch: str
    computed_novelty: float
    epoch_assigned_novelty: float
    epoch_log_novelty: float


def compute_event_novelty(
    event: Event,
    reference: WorldModelReference,
    similarity_fn=None,
) -> float:
    """Compute novelty of an event against a reference world model."""
    concept = Concept(content=event.name)

    measure = AnchoredNoveltyMeasure(reference, similarity_fn)
    score = measure.measure(concept)

    return score.composite_score


# =============================================================================
# Test Runner
# =============================================================================

def run_novelty_tests(
    epochs: list[Epoch],
    use_neural: bool = False,
) -> list[NoveltyTestResult]:
    """
    Run novelty tests across all epochs.

    For each epoch N > 0:
    - Build a world model from epochs 0 to N-1
    - Compute novelty of each event in epoch N against that reference
    - Record results for correlation analysis
    """
    results = []

    # Determine similarity function
    if use_neural:
        similarity_fn = relation_fit_score

        # Preload all event names for efficiency
        all_texts = []
        for epoch in epochs:
            all_texts.append(epoch.name)
            all_texts.append(epoch.description[:200] if epoch.description else "")
            for event in epoch.events:
                all_texts.append(event.name)
        preload_cache([t for t in all_texts if t])
        print(f"Preloaded {len(all_texts)} text embeddings")
    else:
        similarity_fn = None

    # Test each epoch against its predecessor
    for i, epoch in enumerate(epochs):
        if i == 0:
            # No prior epoch to test against
            continue

        # Build reference from all prior epochs
        prior_epochs = epochs[:i]
        reference = build_world_model_from_epoch(epochs[i-1], prior_epochs[:-1] if len(prior_epochs) > 1 else None)

        print(f"\nTesting epoch {epoch.id}: {epoch.name}")
        print(f"  Reference: {epochs[i-1].name}")
        print(f"  Assigned novelty: {epoch.novelty_score:,.0f} (log: {epoch.log_novelty:.2f})")

        # Test each event in this epoch
        for event in epoch.events:
            computed = compute_event_novelty(event, reference, similarity_fn)

            result = NoveltyTestResult(
                event_name=event.name,
                source_epoch=epoch.id,
                reference_epoch=epochs[i-1].id,
                computed_novelty=computed,
                epoch_assigned_novelty=epoch.novelty_score,
                epoch_log_novelty=epoch.log_novelty,
            )
            results.append(result)

            print(f"    {event.name}: computed={computed:.4f}")

    return results


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(results: list[NoveltyTestResult]) -> dict:
    """Analyze correlation between computed and assigned novelty scores."""
    if not results:
        return {"error": "No results to analyze"}

    # Group by epoch
    by_epoch = {}
    for r in results:
        if r.source_epoch not in by_epoch:
            by_epoch[r.source_epoch] = []
        by_epoch[r.source_epoch].append(r)

    # Compute epoch-level averages
    epoch_computed = []
    epoch_assigned_log = []

    for epoch_id, epoch_results in by_epoch.items():
        avg_computed = statistics.mean(r.computed_novelty for r in epoch_results)
        log_assigned = epoch_results[0].epoch_log_novelty

        epoch_computed.append(avg_computed)
        epoch_assigned_log.append(log_assigned)

    # Compute correlation (Pearson)
    n = len(epoch_computed)
    if n < 2:
        correlation = 0.0
    else:
        mean_x = statistics.mean(epoch_computed)
        mean_y = statistics.mean(epoch_assigned_log)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(epoch_computed, epoch_assigned_log))

        var_x = sum((x - mean_x) ** 2 for x in epoch_computed)
        var_y = sum((y - mean_y) ** 2 for y in epoch_assigned_log)

        denominator = math.sqrt(var_x * var_y)
        correlation = numerator / denominator if denominator > 0 else 0.0

    # Summary statistics
    all_computed = [r.computed_novelty for r in results]

    return {
        "total_events_tested": len(results),
        "epochs_tested": len(by_epoch),
        "correlation_with_log_novelty": correlation,
        "computed_novelty_range": (min(all_computed), max(all_computed)),
        "computed_novelty_mean": statistics.mean(all_computed),
        "computed_novelty_stdev": statistics.stdev(all_computed) if len(all_computed) > 1 else 0,
        "epoch_averages": {
            epoch_id: statistics.mean(r.computed_novelty for r in epoch_results)
            for epoch_id, epoch_results in by_epoch.items()
        },
    }


def print_report(results: list[NoveltyTestResult], analysis: dict):
    """Print a formatted report of the test results."""
    print("\n" + "=" * 70)
    print("NOVELTY MODEL TEST REPORT")
    print("=" * 70)

    print(f"\nTotal events tested: {analysis['total_events_tested']}")
    print(f"Epochs tested: {analysis['epochs_tested']}")

    print(f"\nComputed novelty statistics:")
    print(f"  Range: {analysis['computed_novelty_range'][0]:.4f} - {analysis['computed_novelty_range'][1]:.4f}")
    print(f"  Mean: {analysis['computed_novelty_mean']:.4f}")
    print(f"  Stdev: {analysis['computed_novelty_stdev']:.4f}")

    print(f"\nCorrelation with assigned log-novelty: {analysis['correlation_with_log_novelty']:.4f}")

    if analysis['correlation_with_log_novelty'] > 0.7:
        print("  --> STRONG positive correlation")
    elif analysis['correlation_with_log_novelty'] > 0.4:
        print("  --> MODERATE positive correlation")
    elif analysis['correlation_with_log_novelty'] > 0:
        print("  --> WEAK positive correlation")
    else:
        print("  --> NO correlation (or negative)")

    print("\nEpoch-level computed novelty averages:")
    for epoch_id, avg in sorted(analysis['epoch_averages'].items()):
        print(f"  {epoch_id}: {avg:.4f}")

    print("\n" + "=" * 70)
    print("END REPORT")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    # Path to epoch data
    data_path = Path(r"C:\code\sidekick-web\assets\novelty_1.json")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading epoch data from {data_path}")
    epochs = load_epoch_data(data_path)
    print(f"Loaded {len(epochs)} epochs with {sum(len(e.events) for e in epochs)} total events")

    # List epochs
    print("\nEpochs:")
    for e in epochs:
        print(f"  {e.id}: {e.name} (novelty: {e.novelty_score:,.0f}, events: {len(e.events)})")

    # Run tests
    print("\n" + "=" * 70)
    print("RUNNING NOVELTY TESTS")
    print("=" * 70)

    results = run_novelty_tests(epochs, use_neural=USE_NEURAL)

    # Analyze
    analysis = analyze_results(results)

    # Report
    print_report(results, analysis)

    return results, analysis


if __name__ == "__main__":
    main()
