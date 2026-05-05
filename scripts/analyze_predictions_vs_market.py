"""
Analyze model probability vs Kalshi market price over time.

Pulls all observations from Supabase (joined to their run timestamps),
groups by series, and produces:

  1. Per-series time-series plots showing model probability and market
     price evolution. Game-result moments are marked as vertical lines.
  2. A small-multiples grid of all series, for at-a-glance comparison.
  3. A console summary with per-series stats: average edge, edge
     volatility, model-market correlation, max edge moment.

Usage
-----
    python scripts/analyze_predictions_vs_market.py
    python scripts/analyze_predictions_vs_market.py --series DEN_MIN
    python scripts/analyze_predictions_vs_market.py --output-dir docs/timeseries

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) in .env.

Design notes
------------
- We use the service_role key locally because we want to read everything
  (RLS allows anon read too, but consistency with how the bot authenticates).
- "Edge" is defined as model_p - market_p. Positive = model thinks higher
  seed is more likely than market does.
- Market price is fetched as `market_yes_mid` (the mid of bid/ask). When
  one side is missing we fall back to last-trade.
- We don't try to identify when games happened from the observations alone
  — instead we detect "moments where favorite_wins or underdog_wins changed
  between consecutive observations" as game-result anchors.
- Model probability changes only at game boundaries (the bot is deterministic
  given series state). Market price changes continuously. The visible
  "step function" of model vs "smooth curve" of market is informative —
  it shows when the market moved without the model moving.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class Observation:
    """One row from the observations table, joined to its run's timestamp."""
    run_id: int
    timestamp: datetime
    series_key: str
    favorite: str
    underdog: str
    favorite_wins: int
    underdog_wins: int
    model_p: float                  # p_fav_series_raw
    model_p_adj: Optional[float]    # p_fav_series_tail_adj
    market_p: Optional[float]       # market_yes_mid
    edge_raw: Optional[float]       # model - market
    edge_adj: Optional[float]


def load_env() -> Tuple[str, str]:
    """Load Supabase credentials from .env, falling back to env vars."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set "
            "(in .env or environment)"
        )
    return url, key


def fetch_observations(url: str, key: str, log) -> List[Observation]:
    """
    Pull all observations from Supabase, joined to their run timestamps.

    Uses the supabase-py client (already in requirements.txt). We page
    through results because Supabase caps at 1000 rows by default.
    """
    from supabase import create_client
    client = create_client(url, key)

    # Step 1: pull all runs (we need their started_at for timestamps)
    runs_by_id: Dict[int, datetime] = {}
    page = 0
    page_size = 1000
    while True:
        resp = (client.table("runs")
                .select("id, started_at")
                .order("id")
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute())
        rows = resp.data or []
        for r in rows:
            ts = r.get("started_at")
            if ts:
                # Supabase returns ISO8601 strings
                runs_by_id[r["id"]] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if len(rows) < page_size:
            break
        page += 1
    log.info(f"Loaded {len(runs_by_id)} run timestamps")

    # Step 2: pull all observations
    obs_list: List[Observation] = []
    page = 0
    while True:
        resp = (client.table("observations")
                .select("run_id, series_key, favorite, underdog, "
                        "favorite_wins, underdog_wins, "
                        "p_fav_series_raw, p_fav_series_tail_adj, "
                        "market_yes_mid, edge_raw, edge_tail_adj")
                .order("run_id")
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute())
        rows = resp.data or []
        for r in rows:
            run_id = r.get("run_id")
            if run_id is None or run_id not in runs_by_id:
                continue
            try:
                obs_list.append(Observation(
                    run_id=run_id,
                    timestamp=runs_by_id[run_id],
                    series_key=r["series_key"],
                    favorite=r["favorite"],
                    underdog=r.get("underdog", "?"),
                    favorite_wins=int(r.get("favorite_wins") or 0),
                    underdog_wins=int(r.get("underdog_wins") or 0),
                    model_p=float(r["p_fav_series_raw"]) if r.get("p_fav_series_raw") is not None else None,
                    model_p_adj=float(r["p_fav_series_tail_adj"]) if r.get("p_fav_series_tail_adj") is not None else None,
                    market_p=float(r["market_yes_mid"]) if r.get("market_yes_mid") is not None else None,
                    edge_raw=float(r["edge_raw"]) if r.get("edge_raw") is not None else None,
                    edge_adj=float(r["edge_tail_adj"]) if r.get("edge_tail_adj") is not None else None,
                ))
            except (KeyError, ValueError, TypeError) as e:
                # Defensive: don't let one bad row kill the analysis
                continue
        if len(rows) < page_size:
            break
        page += 1
    log.info(f"Loaded {len(obs_list)} observations")
    return obs_list


def detect_game_result_moments(obs: List[Observation]) -> List[Tuple[datetime, str]]:
    """
    Find moments where the series score changed (= a game result happened).

    Returns list of (timestamp, label) where label is e.g. "fav win G3" or
    "und win G2". We use the first observation that *shows* the new score
    as the game-result moment (won't be more accurate than the cron cadence
    but it's reliable).
    """
    if not obs:
        return []
    obs_sorted = sorted(obs, key=lambda o: o.timestamp)
    moments: List[Tuple[datetime, str]] = []
    prev_score = (obs_sorted[0].favorite_wins, obs_sorted[0].underdog_wins)
    for o in obs_sorted[1:]:
        score = (o.favorite_wins, o.underdog_wins)
        if score != prev_score:
            game_num = score[0] + score[1]
            if score[0] > prev_score[0]:
                label = f"fav G{game_num}"
            else:
                label = f"und G{game_num}"
            moments.append((o.timestamp, label))
            prev_score = score
    return moments


def per_series_summary(obs: List[Observation]) -> Dict[str, Any]:
    """Compute aggregate stats for one series's observations."""
    obs_with_market = [o for o in obs
                        if o.model_p is not None and o.market_p is not None]
    if not obs_with_market:
        return {
            "n_observations": len(obs),
            "n_with_market": 0,
        }
    edges = [o.edge_raw for o in obs_with_market if o.edge_raw is not None]
    model_ps = [o.model_p for o in obs_with_market]
    market_ps = [o.market_p for o in obs_with_market]

    # Pearson correlation (manual, no numpy dependency)
    n = len(model_ps)
    if n > 1:
        mx, my = mean(model_ps), mean(market_ps)
        cov = sum((m - mx) * (k - my) for m, k in zip(model_ps, market_ps)) / n
        sx = pstdev(model_ps) or 1e-9
        sy = pstdev(market_ps) or 1e-9
        corr = cov / (sx * sy)
    else:
        corr = None

    max_edge_obs = max(obs_with_market, key=lambda o: abs(o.edge_raw or 0))

    final_obs = obs_with_market[-1]
    final_score = f"{final_obs.favorite_wins}-{final_obs.underdog_wins}"
    series_resolved = max(final_obs.favorite_wins, final_obs.underdog_wins) >= 4
    favorite_won = final_obs.favorite_wins >= 4

    return {
        "n_observations": len(obs),
        "n_with_market": len(obs_with_market),
        "first_seen": obs_with_market[0].timestamp.isoformat(),
        "last_seen": obs_with_market[-1].timestamp.isoformat(),
        "final_score": final_score,
        "series_resolved": series_resolved,
        "favorite_won": favorite_won if series_resolved else None,
        "mean_model_p": mean(model_ps),
        "mean_market_p": mean(market_ps),
        "mean_edge": mean(edges) if edges else None,
        "edge_stdev": pstdev(edges) if len(edges) > 1 else 0.0,
        "max_abs_edge": abs(max_edge_obs.edge_raw or 0),
        "max_edge_at": max_edge_obs.timestamp.isoformat(),
        "max_edge_score": f"{max_edge_obs.favorite_wins}-{max_edge_obs.underdog_wins}",
        "model_market_corr": corr,
    }


def make_per_series_plot(obs: List[Observation], series_key: str,
                          summary: Dict[str, Any], output_path: Path) -> None:
    """Render the per-series time-series plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if not obs:
        return
    obs_sorted = sorted(obs, key=lambda o: o.timestamp)
    times = [o.timestamp for o in obs_sorted]
    model_ps = [o.model_p for o in obs_sorted]
    market_ps = [o.market_p for o in obs_sorted]

    moments = detect_game_result_moments(obs_sorted)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 7), dpi=110,
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # Top: model vs market
    ax_top.plot(times, model_ps, label="Model P(fav wins)",
                color="#3a6ea5", linewidth=2, drawstyle="steps-post")
    # Market line: only plot points where we have a price
    market_x = [t for t, p in zip(times, market_ps) if p is not None]
    market_y = [p for p in market_ps if p is not None]
    ax_top.plot(market_x, market_y, label="Kalshi market mid",
                color="#a05050", linewidth=1.5, alpha=0.85)

    # Game-result vertical lines
    for ts, label in moments:
        color = "#3a6ea5" if "fav" in label else "#a05050"
        ax_top.axvline(ts, linestyle=":", color=color, alpha=0.6, linewidth=1.2)
        ax_top.text(ts, 0.97, label, color=color, fontsize=8, ha="left",
                    va="top", rotation=90, alpha=0.85,
                    transform=ax_top.get_xaxis_transform())

    # Reference lines
    ax_top.axhline(0.5, linestyle="--", color="#888", linewidth=0.8, alpha=0.5)
    ax_top.set_ylabel("P(favorite wins series)")
    ax_top.set_ylim(0, 1)
    ax_top.legend(loc="upper left", fontsize=10)
    ax_top.grid(True, alpha=0.3)

    # Title with summary
    favorite = obs_sorted[0].favorite
    underdog = obs_sorted[0].underdog
    final = summary.get("final_score", "?")
    resolved = summary.get("series_resolved")
    fav_won = summary.get("favorite_won")
    if resolved:
        outcome = f"final {final}, {'favorite' if fav_won else 'underdog'} won"
    else:
        outcome = f"current {final}"
    ax_top.set_title(
        f"{series_key}: {favorite} (fav) vs {underdog} ({outcome})",
        fontsize=12,
    )

    # Bottom: edge over time
    edges = [(t, o.edge_raw) for t, o in zip(times, obs_sorted) if o.edge_raw is not None]
    if edges:
        edge_times = [t for t, _ in edges]
        edge_vals = [e for _, e in edges]
        ax_bot.plot(edge_times, edge_vals, color="#666", linewidth=1.2)
        ax_bot.fill_between(edge_times, edge_vals, 0,
                             where=[e > 0 for e in edge_vals],
                             color="#3a6ea5", alpha=0.3, step="post",
                             label="Model > Market")
        ax_bot.fill_between(edge_times, edge_vals, 0,
                             where=[e < 0 for e in edge_vals],
                             color="#a05050", alpha=0.3, step="post",
                             label="Model < Market")
        ax_bot.axhline(0, linestyle="-", color="#444", linewidth=0.8)
    ax_bot.set_ylabel("Edge\n(model - market)", fontsize=9)
    ax_bot.legend(loc="upper left", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    # Format x-axis as dates
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def make_small_multiples_plot(by_series: Dict[str, List[Observation]],
                                output_path: Path) -> None:
    """Grid of small per-series plots for at-a-glance comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not by_series:
        return
    n = len(by_series)
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.2 * rows), dpi=110)
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]

    for i, (key, obs) in enumerate(sorted(by_series.items())):
        ax = axes_flat[i]
        obs_sorted = sorted(obs, key=lambda o: o.timestamp)
        times = [o.timestamp for o in obs_sorted]
        model_ps = [o.model_p for o in obs_sorted]
        market_ps = [(o.timestamp, o.market_p) for o in obs_sorted if o.market_p is not None]

        ax.plot(times, model_ps, color="#3a6ea5", linewidth=1.5,
                drawstyle="steps-post", label="model")
        if market_ps:
            ax.plot([t for t, _ in market_ps],
                    [p for _, p in market_ps],
                    color="#a05050", linewidth=1.2, alpha=0.85,
                    label="market")

        # Game moments
        for ts, label in detect_game_result_moments(obs_sorted):
            color = "#3a6ea5" if "fav" in label else "#a05050"
            ax.axvline(ts, linestyle=":", color=color, alpha=0.5, linewidth=0.8)

        favorite = obs_sorted[0].favorite if obs_sorted else "?"
        underdog = obs_sorted[0].underdog if obs_sorted else "?"
        final_score = f"{obs_sorted[-1].favorite_wins}-{obs_sorted[-1].underdog_wins}" if obs_sorted else ""
        ax.set_title(f"{key} ({favorite} vs {underdog}, final {final_score})",
                     fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, linestyle="--", color="#888", linewidth=0.5, alpha=0.4)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", fontsize=7)

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Model probability vs Kalshi market price, all series",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def print_summary_table(summaries: Dict[str, Dict[str, Any]]) -> None:
    """Console report."""
    print()
    print("=" * 100)
    print("PER-SERIES SUMMARY")
    print("=" * 100)
    print(f"{'Series':<10} {'Final':<6} {'Won?':<6} "
          f"{'mean E':>8} {'σ E':>6} {'max |E|':>8} "
          f"{'corr':>6} {'n':>4}")
    print("-" * 100)
    for key in sorted(summaries.keys()):
        s = summaries[key]
        if s.get("n_with_market", 0) == 0:
            print(f"{key:<10} (no market data, n_obs={s['n_observations']})")
            continue

        won_label = ""
        if s.get("series_resolved"):
            won_label = "fav" if s.get("favorite_won") else "und"
        else:
            won_label = "TBD"

        mean_e = s.get("mean_edge")
        std_e = s.get("edge_stdev")
        max_e = s.get("max_abs_edge")
        corr = s.get("model_market_corr")

        print(
            f"{key:<10} {s.get('final_score', '?'):<6} {won_label:<6} "
            f"{mean_e:>+8.3f} {std_e:>6.3f} {max_e:>8.3f} "
            f"{corr:>+6.2f} {s.get('n_with_market', 0):>4}"
        )

    # Highlight max-edge moments — these are the "interesting" data points
    print()
    print("=" * 100)
    print("BIGGEST MISPRICING MOMENTS (one per series)")
    print("=" * 100)
    print(f"{'Series':<10} {'When':<20} {'Score':<6} {'|edge|':>8}")
    print("-" * 100)
    for key in sorted(summaries.keys()):
        s = summaries[key]
        if s.get("n_with_market", 0) == 0:
            continue
        ts = s.get("max_edge_at", "")[:16].replace("T", " ")
        print(f"{key:<10} {ts:<20} {s.get('max_edge_score', ''):<6} "
              f"{s.get('max_abs_edge', 0):>8.3f}")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    log = logging.getLogger("analyze_predictions_vs_market")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--series", nargs="+", default=None,
                        help="Only analyze specific series_keys (default: all)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "docs" / "timeseries")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation, just print the summary")
    parser.add_argument("--summary-output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "timeseries_summary.json")
    args = parser.parse_args()

    url, key = load_env()
    log.info(f"Connecting to Supabase ({url[:40]}...)")
    obs_list = fetch_observations(url, key, log)

    by_series: Dict[str, List[Observation]] = defaultdict(list)
    for o in obs_list:
        by_series[o.series_key].append(o)

    if args.series:
        by_series = {k: v for k, v in by_series.items() if k in args.series}
        log.info(f"Filtered to series: {list(by_series.keys())}")

    log.info(f"Series with data: {len(by_series)}")
    summaries = {key: per_series_summary(obs) for key, obs in by_series.items()}

    print_summary_table(summaries)

    if not args.no_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing plots to {args.output_dir}")

        # Per-series plots
        for key, obs in by_series.items():
            path = args.output_dir / f"timeseries_{key}.png"
            make_per_series_plot(obs, key, summaries[key], path)
            log.info(f"  wrote {path.name}")

        # Small multiples
        grid_path = args.output_dir / "timeseries_all.png"
        make_small_multiples_plot(by_series, grid_path)
        log.info(f"  wrote {grid_path.name}")

    args.summary_output.write_text(json.dumps(summaries, indent=2, default=str))
    log.info(f"Wrote summary to {args.summary_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
