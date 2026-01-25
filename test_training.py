#!/usr/bin/env python3
"""
Test the ML-style training loop with visualization.

This demonstrates:
1. Train/validation split
2. Convergence detection across epochs
3. Learning rate decay
4. Statistical significance testing
5. Logging to JSON (can be visualized) or TensorBoard
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from world_model import (
    ObservationStore, Observation,
    AgentSet, Tendency,
    WorldModel, create_world_model,
)
from world_model.dynamics import (
    Trainer, TrainConfig, TrainHistory,
    Validator, ValidationResult,
    ConsoleLogger, JSONLogger,
)


def main():
    print("=" * 70)
    print("WORLD MODEL TRAINING TEST")
    print("=" * 70)

    # Load observations
    print("\nLoading observations...")
    model = create_world_model("Andrei", "observations.json")
    print(f"Loaded {len(model.observations)} observations")

    # Show initial allocations
    print("\nInitial agent allocations:")
    for agent in sorted(model.agents.all(), key=lambda a: -a.allocation):
        print(f"  {agent.tendency.value}: {agent.allocation:.1%}")

    # Configure training
    config = TrainConfig(
        max_epochs=5,           # Run up to 5 epochs
        min_epochs=2,           # At least 2 epochs
        convergence_threshold=0.01,  # Stop if allocations change < 1%
        patience=2,             # Wait 2 epochs before early stopping
        initial_lr=0.15,        # Start with 15% learning rate
        lr_decay=0.85,          # Decay to 85% each epoch
        validation_split=0.2,   # Hold out 20% for validation
        min_allocation=0.05,    # No tendency below 5%
        max_allocation=0.40,    # No tendency above 40%
    )

    print(f"\nTraining configuration:")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Convergence threshold: {config.convergence_threshold:.1%}")
    print(f"  Validation split: {config.validation_split:.0%}")
    print(f"  Initial learning rate: {config.initial_lr:.0%}")

    # Set up loggers
    json_logger = JSONLogger("training_log.json")
    console_logger = ConsoleLogger()

    # Create trainer
    trainer = Trainer(config)

    # Run training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    trees, history = trainer.train(
        observations=model.observations,
        agents=model.agents,
        logger=json_logger,  # Log to JSON for visualization
        verbose=True,
    )

    # Training summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nEpochs run: {history.epochs_run}")
    print(f"Converged: {history.converged}")
    print(f"Best epoch: {history.best_epoch}")
    print(f"Final learning rate: {history.metrics[-1].learning_rate:.3f}")

    # Allocation history
    print("\nAllocation progression:")
    print("  Epoch | " + " | ".join(f"{t.value[:4]:>6}" for t in Tendency))
    print("  " + "-" * 60)
    for m in history.metrics:
        allocs = " | ".join(f"{m.allocations.get(t, 0):>6.1%}" for t in Tendency)
        print(f"  {m.epoch:5} | {allocs}")

    # Winner history
    print("\nWinners by epoch:")
    for m in history.metrics:
        print(f"  Epoch {m.epoch}: {m.winner.value if m.winner else 'None'} (score: {m.winning_score:.3f})")

    # Convergence plot data
    print("\nConvergence (total allocation change per epoch):")
    for m in history.metrics:
        change = sum(abs(c) for c in m.allocation_changes.values())
        bar = "#" * int(change * 100)
        print(f"  Epoch {m.epoch}: {change:.3f} |{bar}")

    # Final allocations
    print("\nFinal allocations:")
    for agent in sorted(model.agents.all(), key=lambda a: -a.allocation):
        initial = 1/7  # Started uniform
        change = agent.allocation - initial
        direction = "+" if change > 0 else ""
        print(f"  {agent.tendency.value}: {agent.allocation:.1%} ({direction}{change:.1%})")

    # Validation results if available
    if history.validation_results:
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)

        for val_result in history.validation_results:
            print(f"\nValidation at epoch {val_result.epoch}:")
            print(f"  Accuracy: {val_result.accuracy:.1%}")
            print(f"  P-value: {val_result.p_value:.4f}")
            print(f"  Significant: {val_result.is_significant}")
            print(f"  Observations tested: {val_result.total_tested}")

    # Save updated model
    model.save("andrei_trained.json")
    print(f"\nSaved trained model to andrei_trained.json")

    # Point to JSON log
    print(f"\nTraining log saved to training_log.json")
    print("You can visualize this with any JSON viewer or load into a plotting tool.")

    # Quick visualization of final state
    if final_result and final_result.winner:
        print("\n" + "=" * 70)
        print(f"FINAL DEBATE WINNER: {final_result.winner.value.upper()}")
        print("=" * 70)
        winning_claim = next(c for c in final_result.claims if c.proposer == final_result.winner)
        print(f"\nWinning claim: \"{winning_claim.tree.root_value}\"")
        print(f"Score: {winning_claim.score:.3f}")
        print(f"Supporting observations: {len(winning_claim.tree.all_nodes()) - 1}")


def visualize_log(log_file: str = "training_log.json"):
    """Simple ASCII visualization of training log."""
    print("\n" + "=" * 70)
    print("TRAINING LOG VISUALIZATION")
    print("=" * 70)

    with open(log_file, 'r') as f:
        log = json.load(f)

    # Allocation over time
    print("\nAllocation trajectories:")
    tendencies = list(log['epochs'][0]['allocations'].keys()) if log['epochs'] else []

    for tendency in tendencies:
        values = [e['allocations'].get(tendency, 0) for e in log['epochs']]
        if values:
            print(f"\n  {tendency}:")
            for i, v in enumerate(values):
                bar = "#" * int(v * 50)
                print(f"    E{i+1}: {v:.1%} |{bar}")

    # Scores over time
    print("\nScore trajectories:")
    for tendency in tendencies:
        scores = [e['scores'].get(tendency, 0) for e in log['epochs']]
        if scores:
            max_score = max(scores) if scores else 1
            print(f"\n  {tendency}:")
            for i, s in enumerate(scores):
                bar = "#" * int(s / max(max_score, 0.001) * 30)
                print(f"    E{i+1}: {s:.3f} |{bar}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        visualize_log()
    else:
        main()
        print("\n" + "-" * 70)
        print("Run with --visualize to see ASCII visualization of training log")
