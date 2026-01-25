#!/usr/bin/env python3
"""
Visualize training logs from JSON files.

Usage:
    python visualize_training.py training_log.json
    python visualize_training.py training_log.json --plot  # Requires matplotlib
"""

import json
import sys
from pathlib import Path


def load_log(path: str) -> dict:
    """Load training log from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def ascii_bar(value: float, max_value: float = 1.0, width: int = 30) -> str:
    """Create ASCII bar representation."""
    filled = int((value / max_value) * width)
    return "#" * filled + "-" * (width - filled)


def print_allocation_trajectories(log: dict):
    """Print allocation changes over epochs."""
    print("\n" + "=" * 70)
    print("ALLOCATION TRAJECTORIES")
    print("=" * 70)

    epochs = log.get('epochs', [])
    if not epochs:
        print("No epoch data found.")
        return

    # Get all tendencies
    tendencies = list(epochs[0].get('allocations', {}).keys())

    for tendency in sorted(tendencies):
        print(f"\n{tendency.upper()}:")
        for epoch in epochs:
            alloc = epoch.get('allocations', {}).get(tendency, 0)
            bar = ascii_bar(alloc, max_value=0.5)
            print(f"  E{epoch['epoch']:2d}: {alloc:5.1%} |{bar}|")


def print_score_trajectories(log: dict):
    """Print score changes over epochs."""
    print("\n" + "=" * 70)
    print("SCORE TRAJECTORIES")
    print("=" * 70)

    epochs = log.get('epochs', [])
    if not epochs:
        return

    # Find max score for scaling
    max_score = max(
        max(e.get('scores', {}).values()) if e.get('scores') else 1
        for e in epochs
    )

    tendencies = list(epochs[0].get('scores', {}).keys())

    for tendency in sorted(tendencies):
        print(f"\n{tendency.upper()}:")
        for epoch in epochs:
            score = epoch.get('scores', {}).get(tendency, 0)
            bar = ascii_bar(score, max_value=max_score)
            print(f"  E{epoch['epoch']:2d}: {score:6.3f} |{bar}|")


def print_winners(log: dict):
    """Print winner for each epoch."""
    print("\n" + "=" * 70)
    print("WINNERS BY EPOCH")
    print("=" * 70)

    epochs = log.get('epochs', [])
    for epoch in epochs:
        winner = epoch.get('winner', 'None')
        score = epoch.get('winning_score', 0)
        lr = epoch.get('learning_rate', 0)
        print(f"  Epoch {epoch['epoch']}: {winner.upper():12s} (score: {score:.3f}, lr: {lr:.3f})")


def print_convergence(log: dict):
    """Print convergence metrics."""
    print("\n" + "=" * 70)
    print("CONVERGENCE")
    print("=" * 70)

    epochs = log.get('epochs', [])
    for epoch in epochs:
        changes = epoch.get('allocation_changes', {})
        total_change = sum(abs(c) for c in changes.values())
        bar = ascii_bar(total_change, max_value=0.2, width=40)
        print(f"  Epoch {epoch['epoch']}: {total_change:.3f} |{bar}|")


def print_validation(log: dict):
    """Print validation results."""
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    validations = log.get('validations', [])
    if not validations:
        print("  No validation data found.")
        return

    for val in validations:
        epoch = val.get('epoch', '?')
        accuracy = val.get('accuracy', 0)
        baseline = val.get('baseline', 0)
        p_value = val.get('p_value', 1)
        significant = "**" if p_value < 0.05 else ""

        print(f"  Epoch {epoch}:")
        print(f"    Accuracy: {accuracy:.1%} (baseline: {baseline:.1%})")
        print(f"    P-value:  {p_value:.4f} {significant}")


def print_summary(log: dict):
    """Print training summary."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    config = log.get('config', {})
    epochs = log.get('epochs', [])

    print(f"\n  Configuration:")
    print(f"    Max epochs: {config.get('max_epochs', '?')}")
    print(f"    Initial LR: {config.get('initial_lr', '?')}")
    print(f"    LR decay: {config.get('lr_decay', '?')}")
    print(f"    Convergence threshold: {config.get('convergence_threshold', '?')}")

    if epochs:
        final = epochs[-1]
        print(f"\n  Final State (Epoch {final['epoch']}):")
        print(f"    Winner: {final.get('winner', 'None').upper()}")
        print(f"    Learning rate: {final.get('learning_rate', 0):.4f}")

        print(f"\n  Final Allocations:")
        allocs = final.get('allocations', {})
        for t, a in sorted(allocs.items(), key=lambda x: -x[1]):
            print(f"    {t}: {a:.1%}")


def plot_training(log: dict, output_path: str = None):
    """Generate matplotlib plots for training visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    epochs_data = log.get('epochs', [])
    if not epochs_data:
        print("No epoch data to plot.")
        return

    epoch_nums = [e['epoch'] for e in epochs_data]
    tendencies = list(epochs_data[0].get('allocations', {}).keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Allocation trajectories
    ax1 = axes[0, 0]
    for tendency in tendencies:
        values = [e.get('allocations', {}).get(tendency, 0) for e in epochs_data]
        ax1.plot(epoch_nums, values, marker='o', label=tendency)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Allocation')
    ax1.set_title('Allocation Trajectories')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Score trajectories
    ax2 = axes[0, 1]
    for tendency in tendencies:
        values = [e.get('scores', {}).get(tendency, 0) for e in epochs_data]
        ax2.plot(epoch_nums, values, marker='s', label=tendency)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Score Trajectories')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence (total allocation change)
    ax3 = axes[1, 0]
    convergence = [
        sum(abs(c) for c in e.get('allocation_changes', {}).values())
        for e in epochs_data
    ]
    ax3.bar(epoch_nums, convergence, color='steelblue', alpha=0.7)
    ax3.axhline(y=log.get('config', {}).get('convergence_threshold', 0.01),
                color='red', linestyle='--', label='Threshold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Allocation Change')
    ax3.set_title('Convergence (Allocation Delta)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning rate decay
    ax4 = axes[1, 1]
    lrs = [e.get('learning_rate', 0) for e in epochs_data]
    ax4.plot(epoch_nums, lrs, marker='d', color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Decay')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_training.py <log_file.json> [--plot]")
        print("       python visualize_training.py <log_file.json> --plot --output chart.png")
        sys.exit(1)

    log_path = sys.argv[1]
    do_plot = '--plot' in sys.argv

    if not Path(log_path).exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    log = load_log(log_path)

    # ASCII visualization
    print_summary(log)
    print_winners(log)
    print_allocation_trajectories(log)
    print_score_trajectories(log)
    print_convergence(log)
    print_validation(log)

    # Matplotlib visualization
    if do_plot:
        output = None
        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            if idx + 1 < len(sys.argv):
                output = sys.argv[idx + 1]
        plot_training(log, output)


if __name__ == "__main__":
    main()
