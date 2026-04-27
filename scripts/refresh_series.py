"""
Refresh playoff series state from ESPN's scoreboard API.

Writes to `series_state.json`. The bot's config will read from this file if
it exists, falling back to hardcoded values otherwise.

Usage
-----
    python scripts/refresh_series.py                      # last 14 days through today
    python scripts/refresh_series.py --since 2026-04-15   # explicit start date
    python scripts/refresh_series.py --until 2026-04-26   # explicit end date

Design notes
------------
* Source: site.api.espn.com (free, public, no auth, returns clean JSON).
  Documented in many places, e.g. github.com/pseudo-r/Public-ESPN-API.
  Endpoint: /apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD

* We hit the endpoint once per day in the date range. ESPN doesn't publish
  rate limits but the search results suggest "be respectful." For typical
  use (run once per bot invocation, every 5 minutes max), this is fine —
  we make ~14 calls per refresh, well within reasonable bounds.

* We respect `config.SERIES` for which team is the favorite in each
  matchup. Margins are computed from the favorite's perspective. This
  matches the "Option A" decision: config declares the favorite, the
  script just respects it.

* We only count completed games (status STATUS_FINAL). Live and scheduled
  games are skipped. Bayesian updating must use final results only.

* We match games to series by team-pair (order-independent). If a game's
  two teams match a configured series, it belongs to that series.

* Output schema (series_state.json):
    {
      "fetched_at": "...",
      "source": "site.api.espn.com",
      "date_range": ["2026-04-15", "2026-04-26"],
      "series": {
        "DEN_MIN": {
          "favorite_wins": 1,
          "underdog_wins": 3,
          "completed_games": [
            {"margin": 11.0, "favorite_was_home": true,
             "date": "2026-04-18", "fav_score": 116, "und_score": 105},
            ...
          ]
        },
        ...
      }
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Make `import config` work whether run from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)

# Status values we treat as "this game is done." ESPN occasionally uses
# STATUS_FINAL_OT etc. for overtime games, but the prefix STATUS_FINAL is
# what matters.
FINAL_STATUSES = {"STATUS_FINAL", "STATUS_FINAL_OT"}

# Browser-like UA to be polite. ESPN's API is permissive but identifying
# yourself avoids being lumped in with anonymous traffic.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Polite throttle between day-fetches (seconds)
INTER_REQUEST_DELAY = 0.25

# ESPN's NBA scoreboard endpoint uses abbreviations that differ from the
# ones we (and Kalshi, and basketball-reference) use for two-letter-state
# teams. Confirmed via diagnostic on 2026-04-26: ESPN returns "NY" / "SA"
# for the Knicks / Spurs while the rest of the league lives at three
# letters. We normalize on the way in so all downstream code sees the
# canonical three-letter form.
ESPN_TO_CONFIG_ABBREV = {
    "NY":  "NYK",
    "SA":  "SAS",
    "GS":  "GSW",
    "NO":  "NOP",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def normalize_abbrev(espn_abbrev: str) -> str:
    """Map ESPN's abbreviation to the canonical config form."""
    return ESPN_TO_CONFIG_ABBREV.get(espn_abbrev, espn_abbrev)


log = logging.getLogger("refresh_series")


# --- Helpers ---------------------------------------------------------------

def daterange(start: date, end: date) -> List[date]:
    """Inclusive list of dates from start to end."""
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def fetch_one_day(d: date, timeout: float = 15.0) -> List[Dict[str, Any]]:
    """Return the list of events (games) for one day from ESPN."""
    url = f"{ESPN_SCOREBOARD}?dates={d.strftime('%Y%m%d')}"
    log.debug(f"  Fetching {url}")
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("events", []) or []


# --- Game extraction -------------------------------------------------------

def extract_completed_game(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pull the data we care about from one ESPN event, IF it's completed.

    Returns None for live or scheduled games. Also returns None for
    malformed entries (defensive).
    """
    competition = (event.get("competitions") or [{}])[0]

    # Status check
    status = competition.get("status", {}).get("type", {}).get("name")
    if status not in FINAL_STATUSES:
        return None

    competitors = competition.get("competitors") or []
    if len(competitors) != 2:
        return None

    # Pull home and away
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if home is None or away is None:
        return None

    try:
        home_abbrev = normalize_abbrev(home["team"]["abbreviation"])
        away_abbrev = normalize_abbrev(away["team"]["abbreviation"])
        home_score = int(home["score"])
        away_score = int(away["score"])
    except (KeyError, TypeError, ValueError):
        return None

    # Date in ISO YYYY-MM-DD form
    iso_date = (event.get("date") or "")[:10]

    return {
        "home": home_abbrev,
        "away": away_abbrev,
        "home_score": home_score,
        "away_score": away_score,
        "date": iso_date,
        "espn_event_id": event.get("id"),
    }


# --- Match games to series -------------------------------------------------

def build_series_lookup() -> Dict[Tuple[str, str], config.SeriesState]:
    """
    Build a lookup keyed by sorted team-pair tuple -> SeriesState.

    This makes matching order-independent: a DEN-MIN game matches the
    DEN_MIN series regardless of who's home.
    """
    out = {}
    for s in config.SERIES:
        key = tuple(sorted([s.favorite, s.underdog]))
        out[key] = s
    return out


def assign_to_series(
    game: Dict[str, Any],
    lookup: Dict[Tuple[str, str], config.SeriesState],
) -> Optional[config.SeriesState]:
    """Find the series this game belongs to, or None if not playoff-relevant."""
    key = tuple(sorted([game["home"], game["away"]]))
    return lookup.get(key)


def game_to_completed_game_record(
    game: Dict[str, Any],
    series: config.SeriesState,
) -> Dict[str, Any]:
    """
    Convert an ESPN game to our internal completed-game record.

    Margins are computed from the favorite's perspective:
      - positive if the favorite won
      - negative if the underdog won
    """
    fav = series.favorite
    if game["home"] == fav:
        favorite_was_home = True
        fav_score = game["home_score"]
        und_score = game["away_score"]
    elif game["away"] == fav:
        favorite_was_home = False
        fav_score = game["away_score"]
        und_score = game["home_score"]
    else:
        # Shouldn't reach here because lookup key matched on team pair
        raise RuntimeError(f"Favorite {fav} not in game {game}")

    margin = float(fav_score - und_score)
    return {
        "margin": margin,
        "favorite_was_home": favorite_was_home,
        "date": game["date"],
        "fav_score": fav_score,
        "und_score": und_score,
        "espn_event_id": game.get("espn_event_id"),
    }


# --- Main pipeline ---------------------------------------------------------

def collect_games(
    since: date, until: date,
) -> List[Dict[str, Any]]:
    """Fetch all ESPN events in [since, until], filtered to completed games."""
    all_completed = []
    days = daterange(since, until)
    log.info(f"Fetching {len(days)} days of scoreboard data ({since} to {until})")
    for d in days:
        try:
            events = fetch_one_day(d)
        except requests.HTTPError as e:
            log.warning(f"  {d}: HTTP {e.response.status_code if e.response else '?'} — skipping")
            continue
        except requests.RequestException as e:
            log.warning(f"  {d}: network error ({e}) — skipping")
            continue

        completed = [g for g in (extract_completed_game(e) for e in events) if g is not None]
        if completed:
            log.info(f"  {d}: {len(completed)} completed game(s)")
        time.sleep(INTER_REQUEST_DELAY)
        all_completed.extend(completed)

    return all_completed


def build_series_state(games: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Group completed games by series and produce the output structure.

    Games are sorted by date within each series.
    """
    lookup = build_series_lookup()
    by_series: Dict[str, List[Dict[str, Any]]] = {}
    unmatched_pairs = set()

    for g in games:
        series = assign_to_series(g, lookup)
        if series is None:
            unmatched_pairs.add(tuple(sorted([g["home"], g["away"]])))
            continue
        record = game_to_completed_game_record(g, series)
        by_series.setdefault(series.series_key, []).append(record)

    if unmatched_pairs:
        log.info(
            f"  Skipped {sum(1 for g in games if assign_to_series(g, lookup) is None)} "
            f"non-playoff games across {len(unmatched_pairs)} matchups"
        )

    out: Dict[str, Dict[str, Any]] = {}
    for s in config.SERIES:
        recs = sorted(by_series.get(s.series_key, []), key=lambda r: r["date"])
        fav_wins = sum(1 for r in recs if r["margin"] > 0)
        und_wins = sum(1 for r in recs if r["margin"] < 0)
        out[s.series_key] = {
            "favorite": s.favorite,
            "underdog": s.underdog,
            "favorite_wins": fav_wins,
            "underdog_wins": und_wins,
            "completed_games": recs,
        }
    return out


def write_state_json(
    output_path: Path,
    state: Dict[str, Dict[str, Any]],
    since: date,
    until: date,
) -> None:
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "site.api.espn.com",
        "date_range": [since.isoformat(), until.isoformat()],
        "series": state,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    log.info(f"Wrote state for {len(state)} series to {output_path}")


def print_summary(state: Dict[str, Dict[str, Any]]) -> None:
    print()
    print(f"{'Series':<12} {'Score':<6} Recent games")
    print("-" * 78)
    for key, s in state.items():
        score = f"{s['favorite_wins']}-{s['underdog_wins']}"
        recent = s["completed_games"][-3:]  # last few
        recent_str = "  ".join(
            f"G{i + 1 + len(s['completed_games']) - len(recent)} "
            f"{'+' if r['margin'] >= 0 else ''}{r['margin']:.0f} "
            f"({'H' if r['favorite_was_home'] else 'R'})"
            for i, r in enumerate(recent)
        )
        if not recent_str:
            recent_str = "(no games)"
        print(f"{key:<12} {score:<6} {recent_str}")
    print()


# --- CLI -------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Refresh playoff series state from ESPN."
    )
    parser.add_argument(
        "--since", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today() - timedelta(days=14),
        help="Start date (YYYY-MM-DD). Default: today - 14 days.",
    )
    parser.add_argument(
        "--until", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today(),
        help="End date inclusive (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "series_state.json",
        help="Output path. Default: <repo>/series_state.json",
    )
    args = parser.parse_args()

    games = collect_games(args.since, args.until)
    log.info(f"Collected {len(games)} completed games in window")

    state = build_series_state(games)
    write_state_json(args.output, state, args.since, args.until)
    print_summary(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
