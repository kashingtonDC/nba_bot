"""
Calibration metrics for backtested predictions.

Pure functions over lists of (predicted_probability, actual_outcome) pairs.
No I/O. Plotting is in a separate function that imports matplotlib lazily.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# --- Scoring rules ---------------------------------------------------------

def log_loss(predictions: List[Tuple[float, bool]],
             clip: float = 1e-15) -> float:
    """
    Mean negative log-likelihood of a list of (p, y) pairs.

    Lower is better. Strictly proper — minimized when predictions are
    perfectly calibrated. Sensitive to overconfidence: a prediction of 0.99
    on a wrong outcome contributes ~4.6 to the loss, while a prediction of
    0.6 on a wrong outcome contributes ~0.92.
    """
    if not predictions:
        raise ValueError("Cannot compute log_loss on empty predictions")
    total = 0.0
    for p, y in predictions:
        p = max(min(p, 1.0 - clip), clip)
        total += -math.log(p) if y else -math.log(1.0 - p)
    return total / len(predictions)


def brier_score(predictions: List[Tuple[float, bool]]) -> float:
    """
    Mean squared error between predicted probability and binary outcome.

    Lower is better. Less harshly punishing of overconfidence than log-loss.
    Range: [0, 1] where 0 is perfect.
    """
    if not predictions:
        raise ValueError("Cannot compute brier_score on empty predictions")
    total = 0.0
    for p, y in predictions:
        target = 1.0 if y else 0.0
        total += (p - target) ** 2
    return total / len(predictions)


# --- Calibration buckets ---------------------------------------------------

@dataclass(frozen=True)
class CalibrationBucket:
    """One bucket on a calibration curve."""
    bucket_low: float           # lower edge of probability bucket
    bucket_high: float          # upper edge
    n: int                      # count of predictions in this bucket
    avg_predicted: float        # mean predicted probability for predictions in this bucket
    fraction_actual: float      # fraction of those predictions that resolved YES


def calibration_curve(predictions: List[Tuple[float, bool]],
                      n_bins: int = 10) -> List[CalibrationBucket]:
    """
    Bin predictions by predicted probability and compute the actual fraction
    of YES outcomes in each bin. A perfectly calibrated model has
    avg_predicted == fraction_actual in every bucket.

    Empty buckets are returned with n=0 (consumer can skip them when plotting).
    """
    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: List[CalibrationBucket] = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        # Inclusive upper bound on the last bucket so 1.0 lands somewhere
        if i == n_bins - 1:
            in_bucket = [(p, y) for p, y in predictions if low <= p <= high]
        else:
            in_bucket = [(p, y) for p, y in predictions if low <= p < high]
        n = len(in_bucket)
        if n == 0:
            buckets.append(CalibrationBucket(
                bucket_low=low, bucket_high=high,
                n=0, avg_predicted=0.0, fraction_actual=0.0,
            ))
            continue
        avg_p = sum(p for p, _ in in_bucket) / n
        frac_y = sum(1 for _, y in in_bucket if y) / n
        buckets.append(CalibrationBucket(
            bucket_low=low, bucket_high=high,
            n=n, avg_predicted=avg_p, fraction_actual=frac_y,
        ))
    return buckets


def expected_calibration_error(buckets: List[CalibrationBucket]) -> float:
    """
    Sample-size-weighted mean absolute gap between predicted and actual.

    A well-calibrated model has ECE near zero. Magnitude is interpretable:
    ECE = 0.05 means "on average, predictions are 5 percentage points off."

    Standard reference: Naeini et al. (2015), "Obtaining well calibrated
    probabilities using Bayesian binning." This is the equal-width-bin
    version of ECE.
    """
    total_n = sum(b.n for b in buckets)
    if total_n == 0:
        return 0.0
    weighted_error = sum(
        b.n * abs(b.avg_predicted - b.fraction_actual)
        for b in buckets if b.n > 0
    )
    return weighted_error / total_n


def signed_bias(buckets: List[CalibrationBucket]) -> float:
    """
    Sample-size-weighted average of (actual - predicted).

    Negative means the model is overestimating (predictions higher than
    reality). Positive means underestimating. Magnitude in probability
    points.
    """
    total_n = sum(b.n for b in buckets)
    if total_n == 0:
        return 0.0
    weighted = sum(
        b.n * (b.fraction_actual - b.avg_predicted)
        for b in buckets if b.n > 0
    )
    return weighted / total_n


# --- Pretty-printers -------------------------------------------------------

def format_calibration_table(buckets: List[CalibrationBucket]) -> str:
    """Render a calibration table for console output."""
    lines = []
    header = f"{'Bucket':<14} {'N':>5} {'Predicted':>10} {'Actual':>10} {'Diff':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for b in buckets:
        if b.n == 0:
            continue
        diff = b.fraction_actual - b.avg_predicted
        lines.append(
            f"{b.bucket_low:.2f}-{b.bucket_high:.2f}     "
            f"{b.n:>5} "
            f"{b.avg_predicted:>10.3f} "
            f"{b.fraction_actual:>10.3f} "
            f"{diff:>+8.3f}"
        )
    return "\n".join(lines)


# --- Plotting (matplotlib loaded lazily) -----------------------------------

def plot_calibration_curve(
    buckets: List[CalibrationBucket],
    output_path: str,
    title: str = "Model calibration",
    extra_label: Optional[str] = None,
) -> None:
    """
    Render the calibration curve as a PNG.

    Lazy import of matplotlib so the rest of the module is usable without it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = [b for b in buckets if b.n > 0]
    if not used:
        raise ValueError("No non-empty buckets to plot")

    xs = [b.avg_predicted for b in used]
    ys = [b.fraction_actual for b in used]
    sizes = [max(20, 5 * b.n) for b in used]  # marker size scales with sample count

    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--",
            linewidth=1, label="perfectly calibrated")

    # Calibration curve
    ax.plot(xs, ys, color="#d4a574", linewidth=2, marker="o",
            markersize=0, alpha=0.6)
    ax.scatter(xs, ys, s=sizes, color="#d4a574",
               edgecolor="white", linewidth=1.5, zorder=3)

    # Annotate each bucket with its sample count
    for b in used:
        ax.annotate(
            f"n={b.n}",
            xy=(b.avg_predicted, b.fraction_actual),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8, color="#666",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction actually positive")
    ax.set_title(title, fontsize=12)
    if extra_label:
        ax.text(0.02, 0.98, extra_label, transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                family="monospace", color="#444")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
