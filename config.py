"""
Configuration for the NBA series prediction bot.

Edit this file to update:
  - Series state after each game completes (SERIES)
  - Model constants if you want to tune the calibration

Net ratings are loaded from `ratings.json` (written by
`scripts/refresh_ratings.py`) if that file exists, falling back to the
hardcoded `_FALLBACK_NET_RATINGS` dict below if it doesn't.

Refresh ratings from basketball-reference:
    python scripts/refresh_ratings.py
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

log = logging.getLogger(__name__)


# --- Model constants --------------------------------------------------------

HCA = 2.5            # home court advantage (points)
SIGMA_GAME = 11.5    # std dev of single-game margin (points)
SIGMA_THETA = 2.0    # prior uncertainty on team net rating
TAIL_CORRECTION_PP = 0.02  # magnitude of tail-bias adjustment

# Four Factors variance-weighting strength. When > 0, games where the actual
# margin diverges from the Four Factors-implied margin contribute less to
# the posterior (their likelihood variance is inflated). 0 disables the
# weighting entirely and preserves the original constant-variance behavior.
#
# Starting value 0.0 = ship the plumbing tonight, leave the model unchanged.
# Tomorrow, sweep this and find the value with the best calibration.
KAPPA = 0.0

# Prior regression coefficient. Multiplies the regular-season NRtg
# differential before forming the Bayesian prior. Empirically calibrated
# from a 5-season backtest (see README "Calibration" section): c=0.6 reduces
# expected calibration error from ~0.064 to ~0.044 with negligible log-loss
# cost. Interpretation: regular-season NRtg overstates playoff team-strength
# gaps by ~40%, so we shrink the gap before forming the prior.
PRIOR_REGRESSION = 0.6


# --- Team net ratings -------------------------------------------------------

# Hardcoded fallback values (placeholders flagged with "TODO_VERIFY"). These
# are used if `ratings.json` doesn't exist or fails to load. Run
# `python scripts/refresh_ratings.py` to populate ratings.json from
# basketball-reference, which will override these values.

_FALLBACK_NET_RATINGS = {
    # West playoff teams
    "OKC": 11.1,    # confirmed
    "LAL": 4.5,     # TODO_VERIFY
    "DEN": 5.2,     # TODO_VERIFY
    "MIN": 3.0,     # TODO_VERIFY
    "SAS": 4.0,     # TODO_VERIFY
    "HOU": 1.5,     # TODO_VERIFY
    "POR": -0.5,    # TODO_VERIFY
    "PHX": -1.0,    # TODO_VERIFY

    # East playoff teams
    "BOS": 7.0,     # TODO_VERIFY
    "DET": 6.0,     # TODO_VERIFY
    "CLE": 4.0,     # TODO_VERIFY
    "NYK": 3.5,     # TODO_VERIFY
    "TOR": 0.5,     # TODO_VERIFY
    "ATL": 1.0,     # TODO_VERIFY
    "PHI": 0.0,     # TODO_VERIFY
    "ORL": 1.5,     # TODO_VERIFY
}


def _load_ratings() -> Dict[str, float]:
    """
    Load team net ratings, preferring ratings.json over the fallback dict.

    Looks for ratings.json in the project root (next to this file). If the
    file exists, reads it and merges with the fallback so we still have
    coverage for any team missing from the JSON. If it doesn't exist or
    fails to parse, returns the fallback.
    """
    json_path = Path(__file__).resolve().parent / "ratings.json"
    if not json_path.exists():
        log.info(f"ratings.json not found at {json_path}; using fallback values")
        return dict(_FALLBACK_NET_RATINGS)

    try:
        payload = json.loads(json_path.read_text())
        loaded = payload.get("ratings", {})
        if not isinstance(loaded, dict):
            raise ValueError("'ratings' is not a dict")
        # Merge: loaded values override fallback, but fallback fills any gaps
        merged = dict(_FALLBACK_NET_RATINGS)
        merged.update({k: float(v) for k, v in loaded.items()})
        log.info(
            f"Loaded {len(loaded)} ratings from ratings.json "
            f"(metric={payload.get('metric', '?')}, "
            f"fetched_at={payload.get('fetched_at', '?')})"
        )
        return merged
    except (ValueError, OSError) as e:
        log.warning(f"Failed to load ratings.json ({e}); using fallback values")
        return dict(_FALLBACK_NET_RATINGS)


TEAM_NET_RATINGS = _load_ratings()


# --- Current series state ---------------------------------------------------
@dataclass
class CompletedGame:
    """One completed game in a series."""
    margin: float           # positive if favorite won, negative if underdog won
    favorite_was_home: bool
    # Optional advanced data — present when refresh_series.py was run with
    # the summary endpoint (default since v0.4). Used for Four Factors.
    fav_box: Optional[Dict[str, Any]] = None
    und_box: Optional[Dict[str, Any]] = None


@dataclass
class SeriesState:
    """
    Current state of one playoff series.

    The "favorite" is the team with home-court advantage (typically the
    higher seed). `favorite_has_home_court` is almost always True; only set
    False if for some reason the lower seed has home-court (rare).

    `kalshi_market_match` is a substring used to find this series' market in
    the list of open Kalshi markets. The bot fetches all KXNBASERIES markets
    and matches by ticker substring.
    """
    series_key: str                              # e.g. "DEN_MIN"
    favorite: str                                # 3-letter abbrev (must be in TEAM_NET_RATINGS)
    underdog: str
    favorite_wins: int
    underdog_wins: int
    completed_games: List[CompletedGame] = field(default_factory=list)
    favorite_has_home_court: bool = True
    # Substring of the Kalshi ticker for the FAVORITE's YES market.
    # Kalshi lists two markets per series — one per team. We want the
    # favorite's market so its YES price is directly "P(favorite wins)".
    # Pattern observed: "KXNBASERIES-26{LOSER_SEED_TEAM}{HIGHER_SEED_TEAM}R1-{TEAM}"
    # Match on the suffix "R1-{FAVORITE}" to disambiguate.
    kalshi_ticker_match: Optional[str] = None


# As of 2026-04-26 (Sunday). Update after each completed game.
# Note: we track game margins from the favorite's perspective (positive when
# the favorite won), and whether the favorite was home for that game.
#
# Series ordering follows the 2-2-1-1-1 pattern: G1, G2 at favorite's home;
# G3, G4 at underdog's home; G5 favorite home; G6 underdog home; G7 favorite home.

# Hardcoded fallback series state (placeholders — the truth is loaded from
# series_state.json written by `scripts/refresh_series.py`). These values are
# only used if the JSON is missing.
_FALLBACK_SERIES: List[SeriesState] = [
    # West
    SeriesState(
        series_key="OKC_PHX",
        favorite="OKC", underdog="PHX",
        favorite_wins=3, underdog_wins=0,
        completed_games=[
            # TODO_VERIFY game margins — placeholders consistent with a sweep
            CompletedGame(margin=15.0, favorite_was_home=True),
            CompletedGame(margin=10.0, favorite_was_home=True),
            CompletedGame(margin=8.0,  favorite_was_home=False),
        ],
        kalshi_ticker_match="PHXOKCR1-OKC",
    ),
    SeriesState(
        series_key="LAL_HOU",
        favorite="LAL", underdog="HOU",
        favorite_wins=3, underdog_wins=0,
        completed_games=[
            # TODO_VERIFY — placeholders
            CompletedGame(margin=8.0,  favorite_was_home=True),
            CompletedGame(margin=12.0, favorite_was_home=True),
            CompletedGame(margin=5.0,  favorite_was_home=False),
        ],
        kalshi_ticker_match="HOULALR1-LAL",
    ),
    SeriesState(
        series_key="DEN_MIN",
        favorite="DEN", underdog="MIN",
        favorite_wins=1, underdog_wins=3,
        completed_games=[
            CompletedGame(margin=11.0,  favorite_was_home=True),   # G1: DEN won by 11 home
            CompletedGame(margin=-5.0,  favorite_was_home=True),   # G2: DEN lost by 5 home
            CompletedGame(margin=-17.0, favorite_was_home=False),  # G3: DEN lost by 17 road
            CompletedGame(margin=-2.0,  favorite_was_home=False),  # G4: DEN lost (close, TODO_VERIFY margin)
        ],
        kalshi_ticker_match="MINDENR1-DEN",
    ),
    SeriesState(
        series_key="SAS_POR",
        favorite="SAS", underdog="POR",
        favorite_wins=2, underdog_wins=1,
        completed_games=[
            # TODO_VERIFY all
            CompletedGame(margin=10.0,  favorite_was_home=True),
            CompletedGame(margin=8.0,   favorite_was_home=True),
            CompletedGame(margin=-4.0,  favorite_was_home=False),
        ],
        kalshi_ticker_match="PORSASR1-SAS",
    ),

    # East
    SeriesState(
        series_key="BOS_PHI",
        favorite="BOS", underdog="PHI",
        favorite_wins=2, underdog_wins=1,
        completed_games=[
            # TODO_VERIFY all
            CompletedGame(margin=12.0,  favorite_was_home=True),
            CompletedGame(margin=8.0,   favorite_was_home=True),
            CompletedGame(margin=-3.0,  favorite_was_home=False),
        ],
        kalshi_ticker_match="PHIBOSR1-BOS",
    ),
    SeriesState(
        series_key="DET_ORL",
        favorite="DET", underdog="ORL",
        favorite_wins=1, underdog_wins=2,
        completed_games=[
            # TODO_VERIFY all
            CompletedGame(margin=6.0,   favorite_was_home=True),
            CompletedGame(margin=-4.0,  favorite_was_home=True),
            CompletedGame(margin=-8.0,  favorite_was_home=False),
        ],
        kalshi_ticker_match="ORLDETR1-DET",
    ),
    SeriesState(
        series_key="CLE_TOR",
        favorite="CLE", underdog="TOR",
        favorite_wins=2, underdog_wins=1,
        completed_games=[
            # TODO_VERIFY all
            CompletedGame(margin=10.0,  favorite_was_home=True),
            CompletedGame(margin=8.0,   favorite_was_home=True),
            CompletedGame(margin=-22.0, favorite_was_home=False),  # large blowout per earlier discussion
        ],
        kalshi_ticker_match="TORCLER1-CLE",
    ),
    SeriesState(
        series_key="NYK_ATL",
        favorite="NYK", underdog="ATL",
        favorite_wins=2, underdog_wins=2,
        completed_games=[
            # TODO_VERIFY all
            CompletedGame(margin=5.0,   favorite_was_home=True),
            CompletedGame(margin=-3.0,  favorite_was_home=True),
            CompletedGame(margin=-6.0,  favorite_was_home=False),
            CompletedGame(margin=4.0,   favorite_was_home=False),  # NYK won G4 today
        ],
        kalshi_ticker_match="ATLNYKR1-NYK",
    ),
]


def _load_series() -> List[SeriesState]:
    """
    Load current series state, preferring series_state.json over the
    hardcoded fallback.

    The JSON is written by `scripts/refresh_series.py`. It contains a
    `series` dict keyed by series_key with current win counts and the
    full `completed_games` list. The fallback list (above) provides
    `kalshi_ticker_match` and `favorite`/`underdog` identity, which the
    JSON merges into.
    """
    json_path = Path(__file__).resolve().parent / "series_state.json"
    if not json_path.exists():
        log.info(f"series_state.json not found at {json_path}; using fallback values")
        return list(_FALLBACK_SERIES)

    try:
        payload = json.loads(json_path.read_text())
        loaded = payload.get("series", {})
        if not isinstance(loaded, dict):
            raise ValueError("'series' is not a dict")

        out: List[SeriesState] = []
        for fallback in _FALLBACK_SERIES:
            data = loaded.get(fallback.series_key)
            if data is None:
                # No live data for this series; keep fallback
                out.append(fallback)
                continue

            completed = [
                CompletedGame(
                    margin=float(g["margin"]),
                    favorite_was_home=bool(g["favorite_was_home"]),
                    fav_box=g.get("fav_box"),
                    und_box=g.get("und_box"),
                )
                for g in data.get("completed_games", [])
            ]
            out.append(SeriesState(
                series_key=fallback.series_key,
                favorite=fallback.favorite,
                underdog=fallback.underdog,
                favorite_wins=int(data.get("favorite_wins", 0)),
                underdog_wins=int(data.get("underdog_wins", 0)),
                completed_games=completed,
                favorite_has_home_court=fallback.favorite_has_home_court,
                kalshi_ticker_match=fallback.kalshi_ticker_match,
            ))

        log.info(
            f"Loaded series state for {sum(1 for k in loaded.keys() if k in {s.series_key for s in _FALLBACK_SERIES})} "
            f"series from series_state.json (fetched_at={payload.get('fetched_at', '?')})"
        )
        return out
    except (ValueError, OSError, KeyError) as e:
        log.warning(f"Failed to load series_state.json ({e}); using fallback values")
        return list(_FALLBACK_SERIES)


SERIES: List[SeriesState] = _load_series()


def get_series(key: str) -> Optional[SeriesState]:
    for s in SERIES:
        if s.series_key == key:
            return s
    return None
