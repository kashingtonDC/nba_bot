"""
Run the backtest: sweep KAPPA over historical playoff data, report
calibration metrics, and save a calibration curve image.

Usage
-----
    python scripts/run_backtest.py
    python scripts/run_backtest.py --kappas 0 0.25 0.5 1.0
    python scripts/run_backtest.py --output backtest_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make `import backtest.*` and `import model` work from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.historical_data import HistoricalDataFetcher
from backtest.replay import replay_all_seasons, event_to_dict
from backtest.metrics import (
    log_loss, brier_score, calibration_curve,
    format_calibration_table, plot_calibration_curve,
)


DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
DEFAULT_KAPPAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("run_backtest")

    parser = argparse.ArgumentParser(description="Run KAPPA sweep over historical playoff data.")
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help=f"Seasons to backtest. Default: {' '.join(DEFAULT_SEASONS)}",
    )
    parser.add_argument(
        "--kappas", nargs="+", type=float, default=DEFAULT_KAPPAS,
        help=f"KAPPA values to sweep. Default: {DEFAULT_KAPPAS}",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "backtest_results.json",
        help="JSON output path for raw results.",
    )
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path(__file__).resolve().parent.parent / "dashboard" / "calibration.png",
        help="PNG output path for calibration curve at the best KAPPA.",
    )
    parser.add_argument(
        "--prior-regression", type=float, default=0.6,
        help=("Multiplier on the rating differential (default 0.6, "
              "matching the calibrated value in config.py). Set to 1.0 to "
              "disable prior shrinkage and reproduce the un-calibrated model."),
    )
    args = parser.parse_args()

    # Step 1: load cached historical data
    log.info(f"Loading {len(args.seasons)} seasons of historical data")
    fetcher = HistoricalDataFetcher(seasons=args.seasons)
    seasons_data = fetcher.load_all()
    if not seasons_data:
        log.error("No cached historical data found. Run scripts/fetch_historical.py first.")
        return 1

    total_series = sum(len(p.get("series", [])) for _, p, _ in seasons_data)
    log.info(f"  Loaded {len(seasons_data)} seasons, {total_series} series total")

    # Step 2: sweep KAPPA
    sweep_results: List[Dict[str, Any]] = []
    log.info(f"Sweeping KAPPA over {args.kappas} (prior_regression={args.prior_regression})")

    for kappa in args.kappas:
        events = replay_all_seasons(
            seasons_data, kappa=kappa,
            prior_regression=args.prior_regression,
        )
        predictions = [(e.p_higher_wins, e.higher_seed_won_series) for e in events]

        # Subset of events where Four Factors actually mattered (kappa > 0
        # only differs from kappa = 0 when there's at least 1 game with
        # box-score data observed).
        ff_events = [e for e in events if e.n_games_with_factors > 0]
        ff_preds = [(e.p_higher_wins, e.higher_seed_won_series) for e in ff_events]

        ll_all = log_loss(predictions)
        bs_all = brier_score(predictions)

        result = {
            "kappa": kappa,
            "n_predictions": len(predictions),
            "log_loss": ll_all,
            "brier_score": bs_all,
            "n_predictions_with_factors": len(ff_preds),
            "log_loss_with_factors_only": log_loss(ff_preds) if ff_preds else None,
            "brier_score_with_factors_only": brier_score(ff_preds) if ff_preds else None,
        }
        sweep_results.append(result)
        log.info(
            f"  KAPPA={kappa:>4.2f}  "
            f"n={len(predictions):>4}  "
            f"log_loss={ll_all:.4f}  "
            f"brier={bs_all:.4f}"
        )

    # Step 3: pick the best KAPPA by log-loss
    best = min(sweep_results, key=lambda r: r["log_loss"])
    log.info(f"Best KAPPA by log-loss: {best['kappa']} (loss={best['log_loss']:.4f})")

    # Step 4: produce calibration curve at the best KAPPA
    best_events = replay_all_seasons(
        seasons_data, kappa=best["kappa"],
        prior_regression=args.prior_regression,
    )
    best_preds = [(e.p_higher_wins, e.higher_seed_won_series) for e in best_events]
    buckets = calibration_curve(best_preds, n_bins=10)

    print()
    print(f"Sweep results (prior_regression={args.prior_regression})")
    print(f"{'KAPPA':<8} {'N':<6} {'Log-loss':<10} {'Brier':<10} "
          f"{'N(FF only)':<12} {'LL(FF)':<10} {'Brier(FF)':<10}")
    print("-" * 80)
    for r in sweep_results:
        ll_ff = f"{r['log_loss_with_factors_only']:.4f}" if r['log_loss_with_factors_only'] is not None else "—"
        bs_ff = f"{r['brier_score_with_factors_only']:.4f}" if r['brier_score_with_factors_only'] is not None else "—"
        print(
            f"{r['kappa']:<8.2f} {r['n_predictions']:<6} "
            f"{r['log_loss']:<10.4f} {r['brier_score']:<10.4f} "
            f"{r['n_predictions_with_factors']:<12} "
            f"{ll_ff:<10} {bs_ff:<10}"
        )

    print()
    print(f"Calibration at best KAPPA = {best['kappa']}, prior_regression={args.prior_regression}")
    print(format_calibration_table(buckets))

    # Step 5: write outputs
    args.output.write_text(json.dumps({
        "seasons": args.seasons,
        "kappas_swept": args.kappas,
        "prior_regression": args.prior_regression,
        "sweep_results": sweep_results,
        "best_kappa": best["kappa"],
        "best_log_loss": best["log_loss"],
        "calibration_at_best": [
            {
                "bucket_low": b.bucket_low,
                "bucket_high": b.bucket_high,
                "n": b.n,
                "avg_predicted": b.avg_predicted,
                "fraction_actual": b.fraction_actual,
            }
            for b in buckets
        ],
        "events": [event_to_dict(e) for e in best_events],
    }, indent=2))
    log.info(f"Wrote raw results to {args.output}")

    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    plot_calibration_curve(
        buckets,
        str(args.plot_output),
        title=f"Calibration — best KAPPA = {best['kappa']}",
        extra_label=(
            f"n predictions: {best['n_predictions']}\n"
            f"log-loss: {best['log_loss']:.4f}\n"
            f"brier: {best['brier_score']:.4f}\n"
            f"seasons: {', '.join(args.seasons)}"
        ),
    )
    log.info(f"Wrote calibration plot to {args.plot_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
