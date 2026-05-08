"""
Generate a compact outcomes file for the dashboard's "Completed series" tab.

Reads `series_state.json` (the source of truth maintained by refresh_series.py)
and produces `docs/outcomes.json` containing only what the dashboard needs:

  - per-series final score (max wins reached, even if logging stopped early)
  - per-series complete flag
  - per-series round number (inferred from ESPN notes when available)
  - per-game dates and margins (for plotting game-result vertical lines on
    the time-series charts)

This file is committed by the cron workflow to docs/, where GitHub Pages
serves it. The dashboard fetches it from `outcomes.json` at the same origin
as index.html.

Why a compact view rather than committing series_state.json directly?
  - series_state.json contains box scores and other data the dashboard
    doesn't need; serving the full ~50KB file every page load is wasteful.
  - A purpose-built schema means the dashboard doesn't have to know about
    refresh_series.py's internal structure.
  - We can add fields without breaking the upstream JSON shape.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _round_num_from_label(label: Optional[str]) -> Optional[int]:
    """
    Map ESPN round-label strings to round numbers 1-4.

    Duplicated from refresh_series.py to keep this script standalone — we
    don't want to import refresh_series.py just for one helper.
    """
    if not label:
        return None
    lower = label.lower()
    patterns = [
        ("first round", 1),
        ("quarterfinal", 1),
        ("semifinal", 2),
        ("conference semifinal", 2),
        ("conference final", 3),
        ("nba final", 4),
        ("the finals", 4),
    ]
    for pattern, num in patterns:
        if pattern in lower:
            return num
    return None


# Round 1 fallback set — for the 8 manually-configured round-1 series.
# These don't go through ESPN's auto-discovery and so don't get a round
# label from refresh_series.py. Hardcoded here as a fallback.
ROUND1_SERIES_KEYS = {
    "OKC_PHX", "LAL_HOU", "DEN_MIN", "SAS_POR",
    "BOS_PHI", "DET_ORL", "CLE_TOR", "NYK_ATL",
}


def derive_round(series_key: str, series_data: Dict[str, Any]) -> int:
    """
    Determine the round for a series. Tries (in order):
      1. Round label stored on the series (auto-discovered series have this
         from ESPN notes)
      2. Hardcoded round-1 set
      3. Default to round 2 (anything else we know about must be later than
         round 1, since round 1 is the only manually-configured round)
    """
    label = series_data.get("round_label")
    if label:
        n = _round_num_from_label(label)
        if n is not None:
            return n
    if series_key in ROUND1_SERIES_KEYS:
        return 1
    return 2


def build_outcomes(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert series_state.json's payload into the compact dashboard format.

    Input shape (from refresh_series.py):
      {
        "fetched_at": "...", "date_range": [...], "source": "...",
        "series": {
          "DEN_MIN": {
            "favorite": "DEN", "underdog": "MIN",
            "favorite_wins": 1, "underdog_wins": 4,
            "completed_games": [{"date": "...", "margin": ..., ...}, ...],
            "favorite_has_home_court": true,
            "kalshi_ticker_match": "...",
            "complete": true,
            "round_label": "Western Conference First Round",  // optional
          }, ...
        }
      }

    Output shape (compact for dashboard):
      {
        "fetched_at": "...",
        "series": {
          "DEN_MIN": {
            "favorite": "DEN", "underdog": "MIN",
            "favorite_wins": 1, "underdog_wins": 4,
            "complete": true,
            "round": 1,
            "games": [
              {"date": "2026-04-19", "favorite_was_home": true, "margin": 11.0},
              ...
            ]
          }, ...
        }
      }
    """
    out_series: Dict[str, Any] = {}
    for key, s in (state.get("series") or {}).items():
        # Compact game records — just date, home, margin
        games_compact = []
        for g in s.get("completed_games") or []:
            games_compact.append({
                "date": g.get("date"),
                "favorite_was_home": bool(g.get("favorite_was_home")),
                "margin": float(g.get("margin")) if g.get("margin") is not None else None,
            })

        out_series[key] = {
            "favorite": s.get("favorite"),
            "underdog": s.get("underdog"),
            "favorite_wins": int(s.get("favorite_wins") or 0),
            "underdog_wins": int(s.get("underdog_wins") or 0),
            "complete": bool(s.get("complete", False)),
            "round": derive_round(key, s),
            "games": games_compact,
        }

    return {
        # Note: deliberately no fetched_at field. The cron commits this file
        # back to main only when content actually changed; including a
        # timestamp would mean every cron tick produces a "different" file
        # and commits get spammed. The series data itself is the truth.
        "series": out_series,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")
    log = logging.getLogger("generate_outcomes")

    parser = argparse.ArgumentParser(description="Generate docs/outcomes.json from series_state.json")
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent / "series_state.json",
        help="Input series_state.json path",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "outcomes.json",
        help="Output outcomes.json path (defaults to docs/ for GitHub Pages serving)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        return 1

    state = json.loads(args.input.read_text())
    outcomes = build_outcomes(state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcomes, indent=2))

    n_series = len(outcomes["series"])
    n_complete = sum(1 for s in outcomes["series"].values() if s["complete"])
    log.info(f"Wrote {n_series} series ({n_complete} complete) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
