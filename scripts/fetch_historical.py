"""
Fetch historical NBA playoff data for backtesting.

This is a one-time-ish operation — runs once per season, results are cached
to backtest/data/<season>/. Subsequent invocations skip already-cached
seasons unless --force is given.

Usage
-----
    python scripts/fetch_historical.py                           # default 5 seasons
    python scripts/fetch_historical.py --seasons 2024-25         # one season
    python scripts/fetch_historical.py --seasons 2023-24 2024-25 # explicit list
    python scripts/fetch_historical.py --force                   # re-fetch even if cached
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `import backtest.*` work when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.historical_data import HistoricalDataFetcher


DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log = logging.getLogger("fetch_historical")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help=f"Seasons in YYYY-YY format. Default: {' '.join(DEFAULT_SEASONS)}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if a season's data is already cached.",
    )
    args = parser.parse_args()

    log.info(f"Fetching {len(args.seasons)} season(s): {args.seasons}")
    log.info("This will take ~5-10 minutes per season due to rate-limited API calls.")

    fetcher = HistoricalDataFetcher(seasons=args.seasons)
    try:
        results = fetcher.fetch_all(force=args.force)
    except KeyboardInterrupt:
        log.warning("Interrupted. Partial data may be cached.")
        return 130

    # Per-season status
    print()
    print(f"{'Season':<10} {'Status'}")
    print("-" * 50)
    for season, status in results.items():
        print(f"{season:<10} {status}")

    # Summary table for cached/successful seasons
    print()
    print(f"{'Season':<10} {'Series':<8} {'Games':<8} {'Teams w/ ratings'}")
    print("-" * 50)
    for season, series_payload, ratings in fetcher.load_all():
        series = series_payload.get("series", [])
        n_games = sum(len(s.get("games", [])) for s in series)
        print(f"{season:<10} {len(series):<8} {n_games:<8} {len(ratings)}")

    print()
    print(f"Cache dir: {fetcher.cache_dir}")

    failures = [s for s, status in results.items() if status.startswith("failed")]
    if failures:
        log.warning(f"{len(failures)} season(s) failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
