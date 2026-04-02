#!/usr/bin/env python3
"""
Parse training logs and generate per-rank loss comparison plots.
Compares losses from two log files for each rank/process.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Default regex patterns for log parsing
DEFAULT_PATTERNS = [
    # FlagScale train_tracker format: [default0]:train_tracker at step 0: ... loss:0.144
    r"\[default(\d+)\]:train_tracker at step (\d+).*?loss:([\d.]+)",
    # starVLA format: [rank0]:loss: 1.0772619247436523
    r"\[rank(\d+)\]:loss:\s*([\d.]+)",
]


def parse_log_file(
    log_path: str,
    patterns: Optional[List[str]] = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Parse log file and extract loss data per process/rank.

    Args:
        log_path: Path to log file
        patterns: List of regex patterns to match. Each pattern should have either:
            - 3 groups: (rank, step, loss) for formats with explicit steps
            - 2 groups: (rank, loss) for formats without explicit steps (step auto-incremented)

    Returns:
        Dict mapping process_id to list of (step, loss) tuples
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    process_losses: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    print(f"Parsing log file: {log_path}")
    step_counter: Dict[int, int] = defaultdict(int)  # Track step for formats without explicit steps

    try:
        with open(log_path, "r") as f:
            for line in f:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        groups = match.groups()
                        if len(groups) == 3:
                            # Format with explicit step: (rank, step, loss)
                            process_id = int(groups[0])
                            step = int(groups[1])
                            loss = float(groups[2])
                        elif len(groups) == 2:
                            # Format without explicit step: (rank, loss)
                            process_id = int(groups[0])
                            loss = float(groups[1])
                            step = step_counter[process_id]
                            step_counter[process_id] += 1
                        else:
                            continue

                        process_losses[process_id].append((step, loss))
                        break  # Stop after first matching pattern

    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_path}")
        return process_losses

    # Sort by step for each process
    for process_id in process_losses:
        process_losses[process_id].sort(key=lambda x: x[0])

    return process_losses


def plot_loss_comparison_per_rank(
    log_path1: str,
    log_path2: str,
    output_dir: Optional[str] = None,
    label1: str = "Log 1",
    label2: str = "Log 2",
    patterns: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Generate per-rank loss comparison plots.

    Args:
        log_path1: Path to first log file
        log_path2: Path to second log file
        output_dir: Directory to save plots (default: same as log_path1)
        label1: Label for first log file in plots
        label2: Label for second log file in plots
        patterns: Custom regex patterns for parsing

    Returns:
        Output directory path, or None if no data found
    """
    # Parse both log files
    process_losses1 = parse_log_file(log_path1, patterns)
    process_losses2 = parse_log_file(log_path2, patterns)

    if not process_losses1 and not process_losses2:
        print("No loss data found in either log file!")
        return None

    # Get all process IDs from both logs
    all_process_ids = set(process_losses1.keys()) | set(process_losses2.keys())

    if not all_process_ids:
        print("No process data found!")
        return None

    print(f"\nFound data for processes: {sorted(all_process_ids)}")
    for pid in sorted(all_process_ids):
        count1 = len(process_losses1.get(pid, []))
        count2 = len(process_losses2.get(pid, []))
        print(f"  Process {pid}: {count1} entries ({label1}), {count2} entries ({label2})")

    # Create output directory
    if output_dir is None:
        output_dir_path = Path(log_path1).parent
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Create a plot for each rank/process
    for process_id in sorted(all_process_ids):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Build step -> loss dicts for alignment by actual step number
        step_to_loss1 = {step: loss for step, loss in process_losses1.get(process_id, [])}
        step_to_loss2 = {step: loss for step, loss in process_losses2.get(process_id, [])}

        # Plot each series using its actual step numbers
        if step_to_loss1:
            steps1 = sorted(step_to_loss1.keys())
            losses1 = [step_to_loss1[s] for s in steps1]
            ax.plot(
                steps1,
                losses1,
                "b-",
                linewidth=2,
                label=label1,
                alpha=0.8,
                marker="o",
                markersize=2,
            )

        if step_to_loss2:
            steps2 = sorted(step_to_loss2.keys())
            losses2 = [step_to_loss2[s] for s in steps2]
            ax.plot(
                steps2,
                losses2,
                "r--",
                linewidth=2,
                label=label2,
                alpha=0.8,
                marker="s",
                markersize=2,
            )

        common_steps = sorted(set(step_to_loss1.keys()) & set(step_to_loss2.keys()))
        print(
            f"Process {process_id} - common steps: {len(common_steps)} "
            f"({label1} {len(step_to_loss1)}, "
            f"{label2} {len(step_to_loss2)})"
        )

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Loss Comparison - Rank {process_id}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="best")

        # Add statistics
        stats_lines = []
        if step_to_loss1:
            all_losses1 = [step_to_loss1[s] for s in sorted(step_to_loss1.keys())]
            final1 = all_losses1[-1]
            min1 = min(all_losses1)
            stats_lines.append(f"{label1}:")
            stats_lines.append(f"  Final: {final1:.4f}")
            stats_lines.append(f"  Min: {min1:.4f}")

        if step_to_loss2:
            all_losses2 = [step_to_loss2[s] for s in sorted(step_to_loss2.keys())]
            final2 = all_losses2[-1]
            min2 = min(all_losses2)
            stats_lines.append(f"{label2}:")
            stats_lines.append(f"  Final: {final2:.4f}")
            stats_lines.append(f"  Min: {min2:.4f}")

        if stats_lines:
            stats_text = "\n".join(stats_lines)
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

        plt.tight_layout()

        # Save figure
        output_path = output_dir_path / f"loss_comparison_rank_{process_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot for rank {process_id} to: {output_path}")
        plt.close()

    print(f"\nAll plots saved to: {output_dir_path}")
    return output_dir_path


def plot_relative_diff_per_rank(
    log_path1: str,
    log_path2: str,
    output_dir: Optional[str] = None,
    label1: str = "Log 1",
    label2: str = "Log 2",
    patterns: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Generate per-rank relative difference plots between two loss curves.

    Args:
        log_path1: Path to first log file
        log_path2: Path to second log file (used as reference/baseline)
        output_dir: Directory to save plots (default: same as log_path1)
        label1: Label for first log file
        label2: Label for second log file (baseline)
        patterns: Custom regex patterns for parsing

    Returns:
        Output directory path, or None if no data found
    """
    # Parse both log files
    process_losses1 = parse_log_file(log_path1, patterns)
    process_losses2 = parse_log_file(log_path2, patterns)

    if not process_losses1 or not process_losses2:
        print("Need data from both log files to compute relative difference!")
        return None

    # Get common process IDs
    common_process_ids = set(process_losses1.keys()) & set(process_losses2.keys())

    if not common_process_ids:
        print("No common process IDs found between the two logs!")
        return None

    print(f"\nComputing relative difference for processes: {sorted(common_process_ids)}")

    # Create output directory
    if output_dir is None:
        output_dir_path = Path(log_path1).parent
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Create a plot for each rank/process
    for process_id in sorted(common_process_ids):
        # Build step -> loss dicts for alignment by actual step number
        step_to_loss1 = {step: loss for step, loss in process_losses1.get(process_id, [])}
        step_to_loss2 = {step: loss for step, loss in process_losses2.get(process_id, [])}

        # Align on common steps
        common_steps = sorted(set(step_to_loss1.keys()) & set(step_to_loss2.keys()))
        if not common_steps:
            print(f"Process {process_id}: no overlapping steps to compare.")
            continue

        steps = np.array(common_steps)
        losses1_arr = np.array([step_to_loss1[s] for s in common_steps])
        losses2_arr = np.array([step_to_loss2[s] for s in common_steps])

        # Compute relative difference: (loss1 - loss2) / loss2 * 100
        # Use loss2 (baseline) as reference
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.where(
                losses2_arr != 0, (losses1_arr - losses2_arr) / losses2_arr * 100, 0
            )

        # Also compute absolute difference
        abs_diff = losses1_arr - losses2_arr

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Relative difference (%)
        ax1 = axes[0]
        ax1.plot(steps, rel_diff, "g-", linewidth=1.5, alpha=0.8, marker="o", markersize=1)
        ax1.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax1.fill_between(steps, 0, rel_diff, alpha=0.3, color="green")
        ax1.set_xlabel("Training Step", fontsize=12)
        ax1.set_ylabel("Relative Difference (%)", fontsize=12)
        ax1.set_title(
            f"Relative Loss Difference - Rank {process_id}\n"
            f"({label1} - {label2}) / {label2} x 100%",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)

        # Add statistics for relative diff
        mean_rel = np.mean(rel_diff)
        std_rel = np.std(rel_diff)
        max_rel = np.max(np.abs(rel_diff))
        stats_text1 = f"Mean: {mean_rel:+.2f}%\nStd: {std_rel:.2f}%\nMax |diff|: {max_rel:.2f}%"
        ax1.text(
            0.98,
            0.98,
            stats_text1,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

        # Plot 2: Absolute difference
        ax2 = axes[1]
        ax2.plot(steps, abs_diff, "b-", linewidth=1.5, alpha=0.8, marker="o", markersize=1)
        ax2.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax2.fill_between(steps, 0, abs_diff, alpha=0.3, color="blue")
        ax2.set_xlabel("Training Step", fontsize=12)
        ax2.set_ylabel("Absolute Difference", fontsize=12)
        ax2.set_title(
            f"Absolute Loss Difference - Rank {process_id}\n({label1} - {label2})",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)

        # Add statistics for absolute diff
        mean_abs = np.mean(abs_diff)
        std_abs = np.std(abs_diff)
        max_abs = np.max(np.abs(abs_diff))
        stats_text2 = f"Mean: {mean_abs:+.4f}\nStd: {std_abs:.4f}\nMax |diff|: {max_abs:.4f}"
        ax2.text(
            0.98,
            0.98,
            stats_text2,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )

        plt.tight_layout()

        # Save figure
        output_path = output_dir_path / f"loss_relative_diff_rank_{process_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved relative diff plot for rank {process_id} to: {output_path}")
        plt.close()

    print(f"\nAll relative diff plots saved to: {output_dir_path}")
    return output_dir_path


def main():
    parser = argparse.ArgumentParser(
        description="Parse training logs and generate per-rank loss comparison plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with two log files
  python plot_loss_comparison.py log1.txt log2.txt

  # Specify output directory and labels
  python plot_loss_comparison.py log1.txt log2.txt -o ./plots -l1 "New" -l2 "Baseline"

  # Use custom regex pattern (with explicit step)
  python plot_loss_comparison.py log1.txt log2.txt --pattern "rank(\\d+).*step(\\d+).*loss=(\\d+\\.\\d+)"

  # Use custom regex pattern (auto-increment step)
  python plot_loss_comparison.py log1.txt log2.txt --pattern "\\[rank(\\d+)\\]:loss:(\\d+\\.\\d+)"

  # Only generate comparison plots (no relative diff)
  python plot_loss_comparison.py log1.txt log2.txt --no-diff

  # Only generate relative diff plots
  python plot_loss_comparison.py log1.txt log2.txt --diff-only
        """,
    )

    parser.add_argument("log1", help="Path to first log file")
    parser.add_argument("log2", help="Path to second log file (baseline for relative diff)")
    parser.add_argument("-o", "--output-dir", help="Output directory for plots")
    parser.add_argument("-l1", "--label1", default="Log 1", help="Label for first log file")
    parser.add_argument(
        "-l2", "--label2", default="Log 2", help="Label for second log file (baseline)"
    )
    parser.add_argument(
        "-p",
        "--pattern",
        action="append",
        dest="patterns",
        help="Custom regex pattern for parsing. Can be specified multiple times. "
        "Pattern should have 3 groups (rank, step, loss) or 2 groups (rank, loss).",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Skip generating relative difference plots",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="Only generate relative difference plots",
    )

    args = parser.parse_args()

    if args.no_diff and args.diff_only:
        parser.error("Cannot specify both --no-diff and --diff-only")

    # Use custom patterns if provided, otherwise use defaults
    patterns = args.patterns if args.patterns else None

    if not args.diff_only:
        plot_loss_comparison_per_rank(
            args.log1,
            args.log2,
            args.output_dir,
            args.label1,
            args.label2,
            patterns,
        )

    if not args.no_diff:
        plot_relative_diff_per_rank(
            args.log1,
            args.log2,
            args.output_dir,
            args.label1,
            args.label2,
            patterns,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
