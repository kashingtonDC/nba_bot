"""
NBA series prediction bot — main entry point.

Flow:
  1. Start a run (insert into Supabase `runs` table).
  2. Fetch open Kalshi NBA series markets.
  3. For each configured series, find the matching Kalshi market, compute
     the model probability, build an observation row.
  4. Bulk-insert observations and close out the run.

Run with:  python bot.py

This is logging-only. It does not place trades.
"""
from __future__ import annotations
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import config
import factors
import kalshi
import model as M
import db


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("bot")


def evaluate_one_series(
    series: config.SeriesState,
    market: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute model + market data for one series and return an observations row.

    `market` may be None if no matching Kalshi market was found for this
    series — we still log the model output, just without market comparison.
    """
    fav_nrtg = config.TEAM_NET_RATINGS.get(series.favorite)
    und_nrtg = config.TEAM_NET_RATINGS.get(series.underdog)
    if fav_nrtg is None or und_nrtg is None:
        raise ValueError(
            f"Net rating missing for {series.series_key}: "
            f"{series.favorite}={fav_nrtg}, {series.underdog}={und_nrtg}"
        )

    games_for_model = [(g.margin, g.favorite_was_home) for g in series.completed_games]

    # Compute Four Factors-implied expected margin per game (None where
    # box-score data isn't available). When KAPPA = 0 these have no effect
    # on the model — but we compute and log them anyway so we can see the
    # divergence between actual and expected margins.
    expected_margins: List[Optional[float]] = []
    for g in series.completed_games:
        em = None
        if g.fav_box and g.und_box:
            fav_ff = factors.four_factors_from_box(g.fav_box, g.und_box)
            und_ff = factors.four_factors_from_box(g.und_box, g.fav_box)
            if fav_ff and und_ff:
                em = factors.expected_margin(fav_ff, und_ff)
                # Sign convention check: expected_margin is from the favorite's
                # perspective. If the favorite was on the road, HCA shaved
                # points off — but we already model HCA separately in the
                # update, so we don't double-count it here.
        expected_margins.append(em)

    prices = kalshi.extract_prices(market) if market else {
        "yes_bid": None, "yes_ask": None, "yes_last": None,
        "yes_mid": None, "volume": None, "volume_24h": None,
    }

    # Orderbook depth: one extra request per matched market. Uses the
    # throttled _get under the hood so we stay polite even at peak.
    ob_summary = {
        "ob_best_bid": None, "ob_best_bid_size": None,
        "ob_best_ask": None, "ob_best_ask_size": None,
        "ob_spread": None, "ob_mid": None, "ob_depth_top": None,
    }
    if market and market.get("ticker"):
        raw_ob = kalshi.get_orderbook(market["ticker"])
        if raw_ob:
            book = kalshi.reconstruct_yes_orderbook(raw_ob)
            ob_summary = kalshi.summarize_orderbook(book)

    # Prefer the orderbook mid over the legacy bid/ask mid: it's strictly
    # fresher (orderbook is a real-time snapshot, while market endpoint
    # fields can lag) and uses the same definition.
    market_price = ob_summary["ob_mid"] or prices["yes_mid"] or prices["yes_last"]

    obs = M.evaluate_series(
        fav_net_rating=fav_nrtg,
        und_net_rating=und_nrtg,
        fav_wins=series.favorite_wins,
        und_wins=series.underdog_wins,
        completed_games=games_for_model,
        favorite_has_home_court=series.favorite_has_home_court,
        market_price=market_price,
        sigma_game=config.SIGMA_GAME,
        sigma_theta=config.SIGMA_THETA,
        hca=config.HCA,
        tail_magnitude_pp=config.TAIL_CORRECTION_PP,
        expected_margins=expected_margins,
        kappa=config.KAPPA,
        prior_regression=config.PRIOR_REGRESSION,
    )

    # Determine next-game home team for logging
    games_played = series.favorite_wins + series.underdog_wins
    next_game_number = games_played + 1
    if next_game_number <= 7:
        higher_seed_home = M.HIGHER_SEED_HOME_BY_GAME[next_game_number]
        fav_home = higher_seed_home if series.favorite_has_home_court else (not higher_seed_home)
        next_game_home = "favorite" if fav_home else "underdog"
    else:
        next_game_number = None
        next_game_home = None

    edge_raw = None
    edge_tail = None
    if market_price is not None:
        edge_raw = obs.p_fav_series_raw - market_price
        if obs.p_fav_series_tail_adj is not None:
            edge_tail = obs.p_fav_series_tail_adj - market_price

    # Four Factors summary stats over the games we have data for. We log
    # these even when KAPPA = 0, so we can see how predictive they would
    # have been if/when we turn the weighting on.
    n_games_with_factors = sum(1 for em in expected_margins if em is not None)
    avg_observed_margin = (
        sum(g.margin for g in series.completed_games) / len(series.completed_games)
        if series.completed_games else None
    )
    avg_expected_margin = (
        sum(em for em in expected_margins if em is not None) / n_games_with_factors
        if n_games_with_factors > 0 else None
    )
    avg_margin_divergence = None
    if avg_observed_margin is not None and avg_expected_margin is not None:
        avg_margin_divergence = avg_observed_margin - avg_expected_margin

    row = {
        "series_key": series.series_key,
        "kalshi_ticker": (market or {}).get("ticker"),
        "favorite": series.favorite,
        "underdog": series.underdog,
        "favorite_wins": series.favorite_wins,
        "underdog_wins": series.underdog_wins,
        "next_game_home": next_game_home,
        "next_game_number": next_game_number,
        "fav_net_rating": fav_nrtg,
        "und_net_rating": und_nrtg,
        "net_rating_diff": obs.posterior_diff_mean,
        "posterior_uncertainty": obs.posterior_diff_std,
        "p_fav_home": obs.p_fav_home,
        "p_fav_road": obs.p_fav_road,
        "p_fav_series_raw": obs.p_fav_series_raw,
        "p_fav_series_tail_adj": obs.p_fav_series_tail_adj,
        "market_yes_bid": prices["yes_bid"],
        "market_yes_ask": prices["yes_ask"],
        "market_yes_last": prices["yes_last"],
        "market_yes_mid": prices["yes_mid"],
        "market_volume": prices["volume"],
        "market_volume_24h": prices["volume_24h"],
        "edge_raw": edge_raw,
        "edge_tail_adj": edge_tail,
        "ob_best_bid": ob_summary["ob_best_bid"],
        "ob_best_bid_size": ob_summary["ob_best_bid_size"],
        "ob_best_ask": ob_summary["ob_best_ask"],
        "ob_best_ask_size": ob_summary["ob_best_ask_size"],
        "ob_spread": ob_summary["ob_spread"],
        "ob_mid": ob_summary["ob_mid"],
        "ob_depth_top": ob_summary["ob_depth_top"],
        "kappa": config.KAPPA,
        "prior_regression": config.PRIOR_REGRESSION,
        "n_games_with_factors": n_games_with_factors,
        "avg_observed_margin": avg_observed_margin,
        "avg_expected_margin": avg_expected_margin,
        "avg_margin_divergence": avg_margin_divergence,
        "raw_market_payload": market,
    }
    return row


def log_summary(row: Dict[str, Any]) -> None:
    """Pretty-print one observation row for the console."""
    fav = row["favorite"]
    und = row["underdog"]
    fw = row["favorite_wins"]
    uw = row["underdog_wins"]
    raw = row["p_fav_series_raw"]
    mid = row["market_yes_mid"]
    edge = row["edge_raw"]
    spread = row.get("ob_spread")
    depth = row.get("ob_depth_top")

    line = f"  {fav} vs {und}  ({fw}-{uw})  model={raw:.1%}"
    if mid is not None:
        line += f"  market={mid:.1%}  edge={edge:+.1%}"
    else:
        line += "  market=—  (no Kalshi match)"

    if spread is not None and depth is not None:
        line += f"  spread={spread*100:.0f}c  depth={depth}"

    log.info(line)


def main() -> int:
    load_dotenv()

    log.info("Starting bot run")
    log.info(f"Configured series: {len(config.SERIES)}")

    # Fetch live Kalshi markets
    try:
        markets = kalshi.list_open_nba_series_markets()
    except kalshi.KalshiError as e:
        log.error(f"Could not fetch Kalshi markets: {e}")
        markets = []

    log.info(f"Fetched {len(markets)} open Kalshi NBA series markets")
    if markets:
        sample_tickers = [m.get("ticker") for m in markets[:5]]
        log.info(f"Sample tickers: {sample_tickers}")

    # Connect to Supabase and start a run
    try:
        client = db.get_client()
        run_id = db.start_run(client, notes="logging-only v0")
        log.info(f"Started run {run_id}")
    except Exception as e:
        log.error(f"Could not connect to Supabase: {e}")
        log.warning("Continuing in dry-run mode (will not write to DB)")
        client = None
        run_id = None

    # Build observation rows
    rows: List[Dict[str, Any]] = []
    log.info("Per-series results:")
    for series in config.SERIES:
        market = None
        if series.kalshi_ticker_match:
            market = kalshi.find_market_by_substring(markets, series.kalshi_ticker_match)
            if market is None:
                log.warning(
                    f"  No Kalshi match for {series.series_key} "
                    f"(searching for '{series.kalshi_ticker_match}')"
                )

        try:
            row = evaluate_one_series(series, market)
            row["run_id"] = run_id
            rows.append(row)
            log_summary(row)
        except Exception as e:
            log.error(f"Failed to evaluate {series.series_key}: {e}")

    # Write to DB
    if client is not None and rows:
        try:
            # Strip raw_market_payload to JSON-serializable dict only
            for r in rows:
                if r.get("raw_market_payload") is not None:
                    # Supabase client handles dict -> jsonb, but make sure it's serializable
                    try:
                        json.dumps(r["raw_market_payload"])
                    except (TypeError, ValueError):
                        r["raw_market_payload"] = None
            db.insert_observations(client, rows)
            db.finish_run(client, run_id, n_markets=len(rows))
            log.info(f"Run {run_id} complete: {len(rows)} observations written")
        except Exception as e:
            log.error(f"Failed to write observations: {e}")
            return 2

    if not rows:
        log.warning("No observations produced")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
