"""
Refresh playoff series state from ESPN's scoreboard + summary APIs.

Writes to `series_state.json`. The bot's config will read from this file if
it exists, falling back to hardcoded values otherwise.

Usage
-----
    python scripts/refresh_series.py                      # last 14 days through today
    python scripts/refresh_series.py --since 2026-04-15   # explicit start date
    python scripts/refresh_series.py --until 2026-04-26   # explicit end date
    python scripts/refresh_series.py --no-advanced        # skip per-game summary fetch

Design notes
------------
* Source: site.api.espn.com / site.web.api.espn.com (free, public, no auth).

* Two endpoints:
    1. Scoreboard: /apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
       One call per day; returns scores, status, team abbrevs, event_id.
    2. Summary:  /apis/site/v2/sports/basketball/nba/summary?event=<id>
       One call per completed playoff game. Returns full box scores
       (for Four Factors), per-quarter scoring, top players by minutes,
       attendance, OT flag, and ESPN's pregame win probability.

* We only fetch the summary endpoint for games we identify as playoff games
  (both teams are in our SERIES config). This bounds the cost. For a full
  first round, that's ~50 extra API calls, throttled at 0.25s each.

* We respect `config.SERIES` for which team is the favorite in each
  matchup. Margins and box scores are computed from the favorite's
  perspective.

* We only count completed games (status STATUS_FINAL or STATUS_FINAL_OT).

* Output schema (series_state.json) — see code below for the full nested
  shape. Each completed_games entry now includes optional `fav_box`,
  `und_box`, `fav_quarters`, `und_quarters`, `is_overtime`, `attendance`,
  `fav_top_minutes`, `und_top_minutes`, `espn_pregame_home_win_prob`. The
  base fields (margin, favorite_was_home, date, scores) remain intact for
  backward compatibility — the bot's model code only needs those.
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
ESPN_SUMMARY = (
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
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


# --- Summary endpoint (per-game advanced stats) ----------------------------

# Box-score stat labels we care about. ESPN reports these as a list of
# {"name": "fieldGoalsMade-fieldGoalsAttempted", ...} entries; we map the
# canonical labels to clean keys.
BOX_STAT_LABELS = {
    "fieldGoalsMade-fieldGoalsAttempted":     ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted":     ("ftm", "fta"),
    "totalRebounds":                          ("reb",),
    "offensiveRebounds":                      ("oreb",),
    "defensiveRebounds":                      ("dreb",),
    "assists":                                ("ast",),
    "steals":                                 ("stl",),
    "blocks":                                 ("blk",),
    "turnovers":                              ("tov",),
    "fouls":                                  ("pf",),
}


def fetch_summary(event_id: str, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
    """
    Fetch the full game-summary payload for one ESPN event. Returns None on
    error rather than raising — advanced stats are optional and we don't
    want to break the whole refresh if one game's summary endpoint hiccups.
    """
    url = f"{ESPN_SUMMARY}?event={event_id}"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.warning(f"  Summary fetch failed for event {event_id}: {e}")
        return None


def _parse_box_stat(value: str, slot_count: int) -> List[Optional[int]]:
    """
    Convert a stat string to integers.

    ESPN reports paired stats like "12-28" for FGM-FGA in a single string.
    Single-value stats come as "12". slot_count tells us how many ints we
    expect (1 or 2). Missing/malformed entries return [None] * slot_count.
    """
    if not isinstance(value, str):
        return [None] * slot_count
    parts = value.split("-")
    out: List[Optional[int]] = []
    for p in parts[:slot_count]:
        try:
            out.append(int(p))
        except (ValueError, TypeError):
            out.append(None)
    while len(out) < slot_count:
        out.append(None)
    return out


def parse_team_box(team_stats_node: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """
    Extract box-score totals for one team from the summary payload's
    boxscore.teams[i].statistics list.
    """
    out: Dict[str, Optional[int]] = {
        "fgm": None, "fga": None, "fg3m": None, "fg3a": None,
        "ftm": None, "fta": None, "oreb": None, "dreb": None,
        "reb": None, "ast": None, "stl": None, "blk": None,
        "tov": None, "pf": None,
    }
    stats = team_stats_node.get("statistics") or []
    for entry in stats:
        name = entry.get("name") or ""
        value = entry.get("displayValue") or entry.get("value")
        if name not in BOX_STAT_LABELS:
            continue
        keys = BOX_STAT_LABELS[name]
        parsed = _parse_box_stat(str(value) if value is not None else "", len(keys))
        for k, v in zip(keys, parsed):
            out[k] = v
    return out


def parse_quarters(competitor: Dict[str, Any]) -> List[Optional[int]]:
    """Return the per-quarter scoring list for one competitor, or [] if missing."""
    line_scores = competitor.get("linescores") or []
    out: List[Optional[int]] = []
    for ls in line_scores:
        try:
            out.append(int(ls.get("value")))
        except (TypeError, ValueError):
            out.append(None)
    return out


def parse_top_minutes(boxscore_players: List[Dict[str, Any]],
                      team_abbrev: str,
                      n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the top-N players by minutes for the given team.

    The summary's `boxscore.players` is a list with one entry per team. Each
    entry has `team.abbreviation` and a list of `statistics` groupings.
    Player rows live under statistics[0].athletes (or similar — schema
    varies). We hunt for the right node, then sort by minutes descending.
    """
    for team_block in boxscore_players:
        team_node = team_block.get("team") or {}
        if (team_node.get("abbreviation") or "").upper() != team_abbrev.upper():
            continue

        # The "statistics" list contains stat groupings; the first one is
        # usually the main one with all athletes.
        stat_groups = team_block.get("statistics") or []
        if not stat_groups:
            return []
        athletes_block = stat_groups[0].get("athletes") or []
        labels = stat_groups[0].get("labels") or []  # e.g. ["MIN","FG","3PT","FT","OREB",...]

        # Find indices of the columns we want
        def col(label: str) -> Optional[int]:
            try:
                return labels.index(label)
            except ValueError:
                return None

        idx_min = col("MIN")
        idx_pts = col("PTS")
        idx_reb = col("REB")
        idx_ast = col("AST")

        rows = []
        for ath in athletes_block:
            stats = ath.get("stats") or []
            athlete_node = ath.get("athlete") or {}
            name = athlete_node.get("displayName") or athlete_node.get("shortName")
            if not name:
                continue

            def _i(idx: Optional[int]) -> Optional[int]:
                if idx is None or idx >= len(stats):
                    return None
                try:
                    return int(stats[idx])
                except (ValueError, TypeError):
                    return None

            mins = _i(idx_min)
            if mins is None or mins == 0:
                continue
            rows.append({
                "name": name,
                "min": mins,
                "pts": _i(idx_pts),
                "reb": _i(idx_reb),
                "ast": _i(idx_ast),
            })

        rows.sort(key=lambda r: -(r["min"] or 0))
        return rows[:n]
    return []


def parse_pregame_home_win_prob(payload: Dict[str, Any]) -> Optional[float]:
    """
    Extract ESPN's own pregame home-win probability if available.

    ESPN's predictor field looks like:
      "predictor": {"homeTeam": {"gameProjection": "62.4"}, ...}
    where gameProjection is a percent string.
    """
    pred = payload.get("predictor") or {}
    home = pred.get("homeTeam") or {}
    val = home.get("gameProjection") or home.get("teamChanceLoss")
    if val is None:
        return None
    try:
        return float(val) / 100.0
    except (TypeError, ValueError):
        return None


def extract_summary_data(payload: Dict[str, Any],
                         home_abbrev: str,
                         away_abbrev: str) -> Dict[str, Any]:
    """
    Pull the rich per-game data we want from a summary payload. Returns a
    dict keyed by "home_*" / "away_*" — caller flips to fav/und based on
    which team is the favorite for that series.

    All fields are optional. Anything we can't extract returns None or [].
    """
    out: Dict[str, Any] = {
        "home_box": {}, "away_box": {},
        "home_quarters": [], "away_quarters": [],
        "home_top_minutes": [], "away_top_minutes": [],
        "is_overtime": None,
        "attendance": None,
        "espn_pregame_home_win_prob": None,
    }

    boxscore = payload.get("boxscore") or {}
    teams_block = boxscore.get("teams") or []
    for team_node in teams_block:
        team_info = team_node.get("team") or {}
        ab = (team_info.get("abbreviation") or "").upper()
        # ESPN uses NY/SA — normalize via the same map our scoreboard parser uses
        ab = ESPN_TO_CONFIG_ABBREV.get(ab, ab)
        box = parse_team_box(team_node)
        if ab == home_abbrev.upper():
            out["home_box"] = box
        elif ab == away_abbrev.upper():
            out["away_box"] = box

    # Per-quarter scoring + OT flag from header.competitions[0].competitors
    header = payload.get("header") or {}
    competitions = header.get("competitions") or [{}]
    competition = competitions[0]
    competitors = competition.get("competitors") or []
    for c in competitors:
        team_info = c.get("team") or {}
        ab = (team_info.get("abbreviation") or "").upper()
        ab = ESPN_TO_CONFIG_ABBREV.get(ab, ab)
        quarters = parse_quarters(c)
        if (c.get("homeAway") or "") == "home":
            out["home_quarters"] = quarters
        elif (c.get("homeAway") or "") == "away":
            out["away_quarters"] = quarters

    # Overtime: any team has more than 4 quarter scores
    home_q = out["home_quarters"] or []
    away_q = out["away_quarters"] or []
    out["is_overtime"] = (len(home_q) > 4) or (len(away_q) > 4) if (home_q or away_q) else None

    # Attendance lives in gameInfo
    game_info = payload.get("gameInfo") or {}
    try:
        att = game_info.get("attendance")
        out["attendance"] = int(att) if att is not None else None
    except (TypeError, ValueError):
        out["attendance"] = None

    # Top players by minutes
    boxscore_players = boxscore.get("players") or []
    out["home_top_minutes"] = parse_top_minutes(boxscore_players, home_abbrev)
    out["away_top_minutes"] = parse_top_minutes(boxscore_players, away_abbrev)

    # Pregame home win prob (ESPN's own model)
    out["espn_pregame_home_win_prob"] = parse_pregame_home_win_prob(payload)

    return out


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

    # Playoff detection. ESPN tags postseason games with season.type == 3
    # AND a `notes` array describing the round (e.g., "Eastern Conference
    # Semifinals"). We use both signals: season.type to gate on postseason,
    # notes for the round name (used by auto-discovery to label the series).
    season = event.get("season") or {}
    is_postseason = season.get("type") == 3
    notes_list = competition.get("notes") or []
    notes_headline = next(
        (n.get("headline") for n in notes_list if n.get("headline")),
        None,
    )

    return {
        "home": home_abbrev,
        "away": away_abbrev,
        "home_score": home_score,
        "away_score": away_score,
        "date": iso_date,
        "espn_event_id": event.get("id"),
        "is_postseason": is_postseason,
        "round_label": notes_headline,
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
    summary_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert an ESPN game to our internal completed-game record.

    Margins are computed from the favorite's perspective:
      - positive if the favorite won
      - negative if the underdog won

    If `summary_payload` is provided, this also folds in the rich per-game
    data (box score, per-quarter scoring, top players, OT flag, attendance,
    ESPN's pregame win prob). Box and quarter data are flipped to fav/und
    perspective. The base fields (margin, favorite_was_home, date, scores)
    are set whether or not summary data is available — model code only
    needs those.
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
    record: Dict[str, Any] = {
        "margin": margin,
        "favorite_was_home": favorite_was_home,
        "date": game["date"],
        "fav_score": fav_score,
        "und_score": und_score,
        "espn_event_id": game.get("espn_event_id"),
    }

    if summary_payload is not None:
        # extract_summary_data keys results by home/away. Flip to fav/und
        # based on which side the favorite was on for this game.
        rich = extract_summary_data(
            summary_payload,
            home_abbrev=game["home"],
            away_abbrev=game["away"],
        )
        if favorite_was_home:
            record["fav_box"] = rich["home_box"]
            record["und_box"] = rich["away_box"]
            record["fav_quarters"] = rich["home_quarters"]
            record["und_quarters"] = rich["away_quarters"]
            record["fav_top_minutes"] = rich["home_top_minutes"]
            record["und_top_minutes"] = rich["away_top_minutes"]
        else:
            record["fav_box"] = rich["away_box"]
            record["und_box"] = rich["home_box"]
            record["fav_quarters"] = rich["away_quarters"]
            record["und_quarters"] = rich["home_quarters"]
            record["fav_top_minutes"] = rich["away_top_minutes"]
            record["und_top_minutes"] = rich["home_top_minutes"]
        record["is_overtime"] = rich["is_overtime"]
        record["attendance"] = rich["attendance"]
        # ESPN's pregame win prob is from the home team's perspective.
        # Don't flip — we log it as-is so future analysis can see it.
        record["espn_pregame_home_win_prob"] = rich["espn_pregame_home_win_prob"]

    return record


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


def discover_new_series(
    games: List[Dict[str, Any]],
    existing_lookup: Dict[Tuple[str, str], config.SeriesState],
    ratings: Dict[str, float],
) -> Dict[Tuple[str, str], config.SeriesState]:
    """
    Find postseason matchups in `games` that aren't already in `existing_lookup`,
    and synthesize SeriesState records for them.

    Favorite assignment: higher NRtg wins. If neither team has NRtg data,
    we skip the matchup (can't model without ratings).

    Home court assignment: whoever was home in the EARLIEST observed game
    of this matchup is treated as having homecourt advantage. In the NBA
    playoffs the higher seed always hosts G1, so this is reliable as long
    as we've seen at least one game.

    Kalshi ticker pattern: in our experience, Kalshi tickers follow the
    pattern KXNBASERIES-YY{LOSER_FIRST}{WINNER_FIRST}R{N}-{TEAM}, where
    R{N} matches the round. We extract the round from the ESPN notes
    (e.g., "Eastern Conference Semifinals" -> R2). We construct a best-
    effort substring match; if Kalshi varies the format, the bot's "no
    match" warning will surface that.

    Returns a dict keyed by sorted-team-pair, mapping to new SeriesState
    objects. Series_key is "{FAV}_{UND}" for consistency with manual config.
    """
    discovered: Dict[Tuple[str, str], config.SeriesState] = {}

    # Group games by matchup, capturing earliest-game info
    by_matchup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for g in games:
        if not g.get("is_postseason"):
            continue
        key = tuple(sorted([g["home"], g["away"]]))
        by_matchup.setdefault(key, []).append(g)

    for matchup_key, matchup_games in by_matchup.items():
        if matchup_key in existing_lookup:
            continue   # Already configured; skip auto-discovery

        team_a, team_b = matchup_key
        rating_a = ratings.get(team_a)
        rating_b = ratings.get(team_b)
        if rating_a is None or rating_b is None:
            log.warning(
                f"Auto-discovery: matchup {team_a}-{team_b} has missing NRtg "
                f"(a={rating_a}, b={rating_b}); skipping"
            )
            continue

        # Higher NRtg = favorite
        if rating_a >= rating_b:
            favorite, underdog = team_a, team_b
        else:
            favorite, underdog = team_b, team_a

        # Determine homecourt: who was home in the earliest game?
        earliest = min(matchup_games, key=lambda g: g["date"])
        favorite_has_home_court = earliest["home"] == favorite

        # Round detection from ESPN notes (best-effort)
        round_label = next(
            (g.get("round_label") for g in matchup_games if g.get("round_label")),
            None,
        )
        round_num = _round_num_from_label(round_label)
        round_str = f"R{round_num}" if round_num else "R1"

        # Kalshi ticker substring. The exact format matches KXNBASERIES-26{X}{Y}{R}-{TEAM}.
        # We don't know which side is X vs Y in the actual ticker; we just match
        # on "{R}-{FAVORITE}" which is reliable across both orderings.
        kalshi_match = f"{round_str}-{favorite}"

        series_key = f"{favorite}_{underdog}"
        discovered[matchup_key] = config.SeriesState(
            series_key=series_key,
            favorite=favorite,
            underdog=underdog,
            favorite_wins=0,         # filled in by build_series_state
            underdog_wins=0,
            completed_games=[],
            favorite_has_home_court=favorite_has_home_court,
            kalshi_ticker_match=kalshi_match,
        )
        log.info(
            f"Auto-discovered series: {series_key} "
            f"(round={round_str}, fav-home={favorite_has_home_court}, "
            f"kalshi-match='{kalshi_match}')"
        )

    return discovered


_ROUND_LABEL_PATTERNS = [
    # First Round / Quarterfinals: round 1
    ("first round", 1),
    ("quarterfinal", 1),
    # Semifinals: round 2
    ("semifinal", 2),
    ("conference semifinal", 2),
    # Conference Finals: round 3
    ("conference final", 3),
    # NBA Finals: round 4
    ("nba final", 4),
    ("the finals", 4),
]


def _round_num_from_label(label: Optional[str]) -> Optional[int]:
    """Map ESPN round-label strings to round numbers 1-4."""
    if not label:
        return None
    lower = label.lower()
    for pattern, num in _ROUND_LABEL_PATTERNS:
        if pattern in lower:
            return num
    return None


def build_series_state(
    games: List[Dict[str, Any]],
    fetch_advanced: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Group completed games by series and produce the output structure.

    Process:
    1. Build lookup from config.SERIES (manually configured matchups).
    2. Auto-discover any postseason matchups in `games` not in the lookup,
       creating new SeriesState records for them with NRtg-based favorite
       assignment. This is what enables round 2+ tracking without manual
       config edits.
    3. Match each game to its (configured or discovered) series.
    4. Optionally fetch ESPN summary endpoint for each matched game to
       enrich with box scores etc.
    5. Build the output with win counts and a `complete` flag (true when
       either team has 4 wins).

    Games are sorted by date within each series.
    """
    ratings = config.TEAM_NET_RATINGS
    lookup = build_series_lookup()

    # Auto-discovery pass: extend the lookup with any postseason matchups
    # we observe in the games list that aren't already configured.
    discovered = discover_new_series(games, lookup, ratings)
    lookup.update(discovered)

    by_series: Dict[str, List[Dict[str, Any]]] = {}
    series_by_key: Dict[str, config.SeriesState] = {
        s.series_key: s for s in lookup.values()
    }

    unmatched_count = 0
    unmatched_pairs = set()
    matched: List[Tuple[Dict[str, Any], config.SeriesState]] = []
    for g in games:
        series = assign_to_series(g, lookup)
        if series is None:
            unmatched_count += 1
            unmatched_pairs.add(tuple(sorted([g["home"], g["away"]])))
            continue
        matched.append((g, series))

    if unmatched_pairs:
        log.info(f"  Skipped {unmatched_count} non-playoff games across {len(unmatched_pairs)} matchups")

    # Second pass: enrich with summary data and build records
    if fetch_advanced and matched:
        log.info(f"  Fetching summary endpoint for {len(matched)} playoff games (1 request each, throttled)")

    for g, series in matched:
        summary = None
        if fetch_advanced and g.get("espn_event_id"):
            summary = fetch_summary(g["espn_event_id"])
            time.sleep(INTER_REQUEST_DELAY)
        record = game_to_completed_game_record(g, series, summary_payload=summary)
        by_series.setdefault(series.series_key, []).append(record)

    out: Dict[str, Dict[str, Any]] = {}
    # Iterate over the union of configured + discovered, not just config.SERIES.
    for series_key, s in series_by_key.items():
        recs = sorted(by_series.get(series_key, []), key=lambda r: r["date"])
        fav_wins = sum(1 for r in recs if r["margin"] > 0)
        und_wins = sum(1 for r in recs if r["margin"] < 0)
        complete = fav_wins >= 4 or und_wins >= 4
        out[series_key] = {
            "favorite": s.favorite,
            "underdog": s.underdog,
            "favorite_wins": fav_wins,
            "underdog_wins": und_wins,
            "completed_games": recs,
            "favorite_has_home_court": s.favorite_has_home_court,
            "kalshi_ticker_match": s.kalshi_ticker_match,
            "complete": complete,
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
        default=date.today() - timedelta(days=30),
        help="Start date (YYYY-MM-DD). Default: today - 30 days. Wide enough "
             "to capture a full series even at the back end of a round.",
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
    parser.add_argument(
        "--no-advanced", action="store_true",
        help="Skip the per-game summary endpoint fetch. Faster, but the "
             "output won't contain box scores, quarter splits, top players, "
             "OT flag, attendance, or ESPN's pregame win probability.",
    )
    args = parser.parse_args()

    games = collect_games(args.since, args.until)
    log.info(f"Collected {len(games)} completed games in window")

    state = build_series_state(games, fetch_advanced=not args.no_advanced)
    write_state_json(args.output, state, args.since, args.until)
    print_summary(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
