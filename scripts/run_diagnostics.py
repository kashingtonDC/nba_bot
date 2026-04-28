"""
Diagnostic experiments to understand the model's calibration.

The first backtest run revealed systematic overconfidence — the model
predicts the higher seed wins more often than they actually do, with the
worst miscalibration in the middle (50-70%) of the prediction range.

This script runs three diagnostic experiments to identify the root cause:

  1. Bias by state index. Are pre-series predictions (which depend on
     prior alone) biased differently from post-game predictions (which
     depend on observed games)? If the bias is concentrated in pre-series,
     the prior is wrong. If both are biased similarly, something deeper.

  2. HCA sweep. Sweep home-court advantage in [0, 1, 1.5, 2.0, 2.5, 3.5].
     Higher seeds host G1, G2, G5, G7 (4 of 7 games). If HCA is overstated,
     higher seeds are systematically favored more than reality.

  3. Prior regression sweep. Multiply the pre-series differential by a
     factor c in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]. Tests whether
     regular-season NRtg overstates playoff team strength gaps.

Usage
-----
    python scripts/run_diagnostics.py

Outputs
-------
  - dashboard/diagnostics/state_bias.png
  - dashboard/diagnostics/hca_sweep.png
  - dashboard/diagnostics/prior_sweep.png
  - Console table with summary stats per experiment
  - diagnostics_results.json with raw data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.historical_data import HistoricalDataFetcher
from backtest.replay import replay_all_seasons, PredictionEvent
from backtest.metrics import (
    log_loss, brier_score, calibration_curve,
    format_calibration_table, expected_calibration_error, signed_bias,
)


DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
DEFAULT_KAPPA = 0.0  # Hold this fixed across diagnostics; we can re-sweep later


def _events_to_preds(events: List[PredictionEvent]) -> List[Tuple[float, bool]]:
    return [(e.p_higher_wins, e.higher_seed_won_series) for e in events]


def _summarize(events: List[PredictionEvent], label: str) -> Dict[str, Any]:
    if not events:
        return {"label": label, "n": 0}
    preds = _events_to_preds(events)
    buckets = calibration_curve(preds, n_bins=10)
    return {
        "label": label,
        "n": len(events),
        "log_loss": log_loss(preds),
        "brier_score": brier_score(preds),
        "ece": expected_calibration_error(buckets),
        "signed_bias": signed_bias(buckets),
        "buckets": [
            {
                "bucket_low": b.bucket_low,
                "bucket_high": b.bucket_high,
                "n": b.n,
                "avg_predicted": b.avg_predicted,
                "fraction_actual": b.fraction_actual,
            }
            for b in buckets
        ],
    }


# --- Experiment 1: bias by state index -------------------------------------

def experiment_state_bias(seasons_data, kappa: float) -> Dict[str, Any]:
    """Split events into pre-series vs post-game and compare calibration."""
    all_events = replay_all_seasons(seasons_data, kappa=kappa)
    pre_series = [e for e in all_events if e.state_index == 0]
    post_g1 = [e for e in all_events if e.state_index == 1]
    post_g2 = [e for e in all_events if e.state_index == 2]
    post_g3plus = [e for e in all_events if e.state_index >= 3]

    return {
        "experiment": "state_bias",
        "groups": [
            _summarize(pre_series, "pre_series (state=0)"),
            _summarize(post_g1, "post_G1 (state=1)"),
            _summarize(post_g2, "post_G2 (state=2)"),
            _summarize(post_g3plus, "post_G3+ (state>=3)"),
        ],
    }


# --- Experiment 2: HCA sweep -----------------------------------------------

def experiment_hca_sweep(seasons_data, kappa: float,
                          hcas: List[float]) -> Dict[str, Any]:
    """Sweep home-court advantage and report metrics for each."""
    results = []
    for hca in hcas:
        events = replay_all_seasons(seasons_data, kappa=kappa, hca=hca)
        summary = _summarize(events, f"HCA={hca}")
        summary["hca"] = hca
        results.append(summary)
    return {
        "experiment": "hca_sweep",
        "kappa": kappa,
        "results": results,
    }


# --- Experiment 3: prior regression sweep ----------------------------------

def experiment_prior_sweep(seasons_data, kappa: float,
                            cs: List[float]) -> Dict[str, Any]:
    """Sweep prior-regression coefficient and report metrics for each."""
    results = []
    for c in cs:
        events = replay_all_seasons(seasons_data, kappa=kappa,
                                     prior_regression=c)
        summary = _summarize(events, f"c={c}")
        summary["prior_regression"] = c
        results.append(summary)
    return {
        "experiment": "prior_sweep",
        "kappa": kappa,
        "results": results,
    }


# --- Plotting helpers ------------------------------------------------------

def _plot_sweep(results: List[Dict[str, Any]], x_key: str, x_label: str,
                title: str, output_path: str) -> None:
    """Plot sweep results: log-loss, ECE, signed bias as a function of x_key."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r[x_key] for r in results]
    log_losses = [r["log_loss"] for r in results]
    eces = [r["ece"] for r in results]
    biases = [r["signed_bias"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=110)

    axes[0].plot(xs, log_losses, marker="o", color="#3a6ea5", linewidth=2)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Log-loss (lower better)")
    axes[0].set_title("Log-loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, eces, marker="o", color="#d4a574", linewidth=2)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("ECE (lower better)")
    axes[1].set_title("Expected Calibration Error")
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(0, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    axes[2].plot(xs, biases, marker="o", color="#a05050", linewidth=2)
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Signed bias (actual − predicted)")
    axes[2].set_title("Signed bias (0 = unbiased)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _plot_state_bias(groups: List[Dict[str, Any]], output_path: str) -> None:
    """Bar chart comparing log-loss, ECE, and signed bias across state groups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_groups = [g for g in groups if g.get("n", 0) > 0]
    labels = [g["label"] for g in valid_groups]
    log_losses = [g["log_loss"] for g in valid_groups]
    eces = [g["ece"] for g in valid_groups]
    biases = [g["signed_bias"] for g in valid_groups]
    ns = [g["n"] for g in valid_groups]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=110)

    bars0 = axes[0].bar(labels, log_losses, color="#3a6ea5", alpha=0.8)
    axes[0].set_ylabel("Log-loss")
    axes[0].set_title("Log-loss by state")
    for bar, n in zip(bars0, ns):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"n={n}", ha="center", va="bottom", fontsize=9)

    bars1 = axes[1].bar(labels, eces, color="#d4a574", alpha=0.8)
    axes[1].set_ylabel("ECE")
    axes[1].set_title("Expected Calibration Error by state")

    axes[2].axhline(0, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    bars2 = axes[2].bar(labels, biases,
                         color=["#a05050" if b < 0 else "#50a050" for b in biases],
                         alpha=0.8)
    axes[2].set_ylabel("Signed bias (actual − predicted)")
    axes[2].set_title("Signed bias by state (− = overconfident)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Calibration broken down by prediction state", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


# --- CLI -------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("run_diagnostics")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "diagnostics_results.json")
    parser.add_argument("--plot-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "dashboard" / "diagnostics")
    args = parser.parse_args()

    args.plot_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {len(args.seasons)} seasons of historical data")
    fetcher = HistoricalDataFetcher(seasons=args.seasons)
    seasons_data = fetcher.load_all()
    if not seasons_data:
        log.error("No cached data found. Run scripts/fetch_historical.py first.")
        return 1
    log.info(f"  Loaded {sum(len(p.get('series', [])) for _, p, _ in seasons_data)} series")

    # --- Experiment 1 -----------------------------------------------------
    log.info("Experiment 1: bias by state index")
    state_results = experiment_state_bias(seasons_data, kappa=args.kappa)
    print()
    print(f"=== Experiment 1: bias by state index (KAPPA={args.kappa}) ===")
    print(f"{'State':<25} {'N':>5} {'Log-loss':>10} {'Brier':>10} {'ECE':>8} {'Bias':>10}")
    print("-" * 80)
    for g in state_results["groups"]:
        if g.get("n", 0) == 0:
            continue
        print(f"{g['label']:<25} {g['n']:>5} {g['log_loss']:>10.4f} "
              f"{g['brier_score']:>10.4f} {g['ece']:>8.4f} {g['signed_bias']:>+10.4f}")
    _plot_state_bias(state_results["groups"], str(args.plot_dir / "state_bias.png"))
    log.info(f"  Saved {args.plot_dir / 'state_bias.png'}")

    # --- Experiment 2 -----------------------------------------------------
    log.info("Experiment 2: HCA sweep")
    hcas = [0.0, 1.0, 1.5, 2.0, 2.5, 3.5]
    hca_results = experiment_hca_sweep(seasons_data, kappa=args.kappa, hcas=hcas)
    print()
    print(f"=== Experiment 2: HCA sweep (KAPPA={args.kappa}) ===")
    print(f"{'HCA':<8} {'N':>5} {'Log-loss':>10} {'Brier':>10} {'ECE':>8} {'Bias':>10}")
    print("-" * 60)
    for r in hca_results["results"]:
        print(f"{r['hca']:<8} {r['n']:>5} {r['log_loss']:>10.4f} "
              f"{r['brier_score']:>10.4f} {r['ece']:>8.4f} {r['signed_bias']:>+10.4f}")
    _plot_sweep(hca_results["results"], "hca", "Home-court advantage (points)",
                f"HCA sweep (KAPPA={args.kappa})",
                str(args.plot_dir / "hca_sweep.png"))
    log.info(f"  Saved {args.plot_dir / 'hca_sweep.png'}")

    # --- Experiment 3 -----------------------------------------------------
    log.info("Experiment 3: prior regression sweep")
    cs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prior_results = experiment_prior_sweep(seasons_data, kappa=args.kappa, cs=cs)
    print()
    print(f"=== Experiment 3: prior regression sweep (KAPPA={args.kappa}) ===")
    print(f"{'c':<8} {'N':>5} {'Log-loss':>10} {'Brier':>10} {'ECE':>8} {'Bias':>10}")
    print("-" * 60)
    for r in prior_results["results"]:
        print(f"{r['prior_regression']:<8} {r['n']:>5} {r['log_loss']:>10.4f} "
              f"{r['brier_score']:>10.4f} {r['ece']:>8.4f} {r['signed_bias']:>+10.4f}")
    _plot_sweep(prior_results["results"], "prior_regression",
                "Prior regression coefficient c",
                f"Prior regression sweep (KAPPA={args.kappa})",
                str(args.plot_dir / "prior_sweep.png"))
    log.info(f"  Saved {args.plot_dir / 'prior_sweep.png'}")

    # --- Save raw results ------------------------------------------------
    args.output.write_text(json.dumps({
        "seasons": args.seasons,
        "kappa": args.kappa,
        "experiments": {
            "state_bias": state_results,
            "hca_sweep": hca_results,
            "prior_sweep": prior_results,
        },
    }, indent=2))
    log.info(f"Wrote raw results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
