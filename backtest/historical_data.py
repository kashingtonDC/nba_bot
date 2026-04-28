"""
Historical NBA playoff data fetcher (basketball-reference scraper).

Pulls past playoff series for backtesting:
  - End-of-regular-season net ratings (priors)
  - Playoff series structure (who played whom, who won, in how many games)
  - Per-game team-level Four Factors (eFG%, TOV%, ORB%, FT-rate)

Why basketball-reference and not nba_api / balldontlie?
  - NBA Stats API silently times out from many networks (we saw this)
  - balldontlie now requires a free API key but doesn't expose net ratings
  - We already have working BR scrapers in refresh_ratings.py
  - This data is HISTORICAL — we run the fetch ONCE, cache forever.
    Future BR HTML changes don't affect us because we already have the cache.

All fetched data is cached to backtest/data/<season>/. Subsequent runs
hit the cache instead of re-scraping.

Politeness
----------
basketball-reference doesn't publish rate limits but their robots.txt asks
crawlers to be respectful. We throttle at 3.5 seconds between requests —
slower than feels necessary but it's a one-time fetch and we'd rather
finish without getting blocked. 5 seasons * (~80 games + 2 index pages) ~=
410 requests, ~25 minutes total.

Schema (one cached file per season)
-----------------------------------
backtest/data/<season>/series.json:
    {
      "season": "2023-24",
      "fetched_at": "...",
      "source": "basketball-reference.com",
      "series": [
        {
          "higher_seed": "BOS",
          "lower_seed": "MIA",
          "higher_seed_won": true,
          "higher_seed_wins": 4,
          "lower_seed_wins": 1,
          "round": 1,
          "games": [
            {
              "game_id": "202404210BOS",
              "date": "2024-04-21",
              "higher_seed_was_home": true,
              "higher_seed_score": 114,
              "lower_seed_score": 94,
              "margin": 20.0,
              "higher_seed_box": {fgm, fga, fg3m, ...},
              "lower_seed_box":  {fgm, fga, fg3m, ...}
            },
            ...
          ]
        },
        ...
      ]
    }
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Comment

log = logging.getLogger(__name__)

REQUEST_DELAY = 3.5
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

CACHE_DIR = Path(__file__).resolve().parent / "data"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

BR_TO_CONFIG_ABBREV = {
    "BRK": "BKN",
    "CHO": "CHA",
    "PHO": "PHX",
}

# data-stat attribute on BR's box-score basic table -> our internal key
BOXSCORE_FIELD_MAP = {
    "fg":     "fgm",
    "fga":    "fga",
    "fg3":    "fg3m",
    "fg3a":   "fg3a",
    "ft":     "ftm",
    "fta":    "fta",
    "orb":    "oreb",
    "drb":    "dreb",
    "trb":    "reb",
    "ast":    "ast",
    "stl":    "stl",
    "blk":    "blk",
    "tov":    "tov",
    "pf":     "pf",
    "pts":    "pts",
}


# --- HTTP helpers -----------------------------------------------------------

def _br_url_to_year(season: str) -> int:
    """'2023-24' -> 2024 (BR uses the season-end year in URLs)."""
    return int(season.split("-")[0]) + 1


def _polite_get(url: str, timeout: float = REQUEST_TIMEOUT) -> str:
    """GET with retries on transient failures. Caller throttles."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                wait = 30 * attempt
                log.warning(f"      429 from BR; backing off {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = 5 * attempt
                log.warning(f"      {url} attempt {attempt} failed ({e}); retry in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def _normalize_abbrev(br_abbrev: str) -> str:
    return BR_TO_CONFIG_ABBREV.get(br_abbrev, br_abbrev)


def _find_table(soup_or_html, table_id: str):
    """Find a table by id, including ones embedded inside HTML comments (BR-ism)."""
    if isinstance(soup_or_html, str):
        soup = BeautifulSoup(soup_or_html, "html.parser")
    else:
        soup = soup_or_html

    table = soup.find("table", id=table_id)
    if table is not None:
        return table

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if f'id="{table_id}"' in comment:
            inner = BeautifulSoup(comment, "html.parser")
            t = inner.find("table", id=table_id)
            if t is not None:
                return t
    return None


# --- Ratings parsing --------------------------------------------------------

def parse_ratings_html(html: str) -> Dict[str, float]:
    """
    Parse end-of-season Net Rating (adjusted) from BR's ratings page.

    BR's data-stat names (confirmed via diagnostic 2026-04-27):
      - team_name      : team identifier cell with link to /teams/<ABBREV>/...
      - net_rtg_adj    : strength-of-schedule-adjusted net rating
      - net_rtg        : raw net rating (fallback)
    """
    table = _find_table(html, "ratings")
    if table is None:
        raise RuntimeError("Could not find ratings table")

    ratings: Dict[str, float] = {}
    tbody = table.find("tbody") or table
    for row in tbody.find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue

        # Team cell: try team_name (current) and team (legacy)
        team_cell = (
            row.find("td", {"data-stat": "team_name"})
            or row.find("th", {"data-stat": "team_name"})
            or row.find("td", {"data-stat": "team"})
            or row.find("th", {"data-stat": "team"})
        )
        if team_cell is None:
            continue
        anchor = team_cell.find("a")
        if anchor is None or not anchor.get("href"):
            continue
        m = re.search(r"/teams/([A-Z]{3})/", anchor["href"])
        if not m:
            continue
        ab = _normalize_abbrev(m.group(1))

        # Adjusted Net Rating: net_rtg_adj (current), n_rtg_a (legacy)
        rating_cell = (
            row.find("td", {"data-stat": "net_rtg_adj"})
            or row.find("td", {"data-stat": "n_rtg_a"})
        )
        if rating_cell is None:
            continue
        text = rating_cell.get_text(strip=True)
        if not text:
            continue
        try:
            ratings[ab] = float(text)
        except ValueError:
            continue

    return ratings


# --- Playoff schedule parsing ----------------------------------------------

@dataclass
class _RawGame:
    game_id: str
    date: str
    home_abbrev: str
    away_abbrev: str
    home_score: int
    away_score: int


def parse_playoff_schedule(html: str) -> List[_RawGame]:
    """
    Parse all playoff games from BR's NBA_<year>_games.html page.

    BR's data-stat columns (confirmed via diagnostic 2026-04-27):
      - date_game            : date as text
      - box_score_text       : link to /boxscores/<game_id>.html (this is where
                               the game_id lives — NOT inside date_game)
      - visitor_team_name    : away team cell with /teams/<ABBREV>/ link
      - visitor_pts          : away score
      - home_team_name       : home team cell with /teams/<ABBREV>/ link
      - home_pts             : home score
    """
    table = _find_table(html, "schedule")
    if table is None:
        raise RuntimeError("Could not find schedule table")

    games: List[_RawGame] = []
    tbody = table.find("tbody") or table
    for row in tbody.find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue

        # Date — could be in a <th> or <td>
        date_cell = (
            row.find("th", {"data-stat": "date_game"})
            or row.find("td", {"data-stat": "date_game"})
        )
        if date_cell is None:
            continue
        date_text = date_cell.get_text(strip=True)

        # Box-score link is in its own cell (data-stat="box_score_text")
        box_cell = row.find("td", {"data-stat": "box_score_text"})
        if box_cell is None:
            continue
        box_anchor = box_cell.find("a")
        if box_anchor is None or not box_anchor.get("href"):
            continue
        m = re.search(r"/boxscores/([0-9A-Za-z]+)\.html", box_anchor["href"])
        if not m:
            continue
        game_id = m.group(1)

        # Parse the date — BR uses formats like "Sat, Apr 20, 2024"
        try:
            date_iso = datetime.strptime(date_text, "%a, %b %d, %Y").date().isoformat()
        except ValueError:
            try:
                date_iso = datetime.strptime(game_id[:8], "%Y%m%d").date().isoformat()
            except ValueError:
                log.warning(f"Could not parse date for {game_id}; skipping")
                continue

        away_team_cell = row.find("td", {"data-stat": "visitor_team_name"})
        home_team_cell = row.find("td", {"data-stat": "home_team_name"})
        away_pts_cell = row.find("td", {"data-stat": "visitor_pts"})
        home_pts_cell = row.find("td", {"data-stat": "home_pts"})

        if not all([away_team_cell, home_team_cell, away_pts_cell, home_pts_cell]):
            continue

        away_anchor = away_team_cell.find("a")
        home_anchor = home_team_cell.find("a")
        if not away_anchor or not home_anchor:
            continue
        away_match = re.search(r"/teams/([A-Z]{3})/", away_anchor.get("href", ""))
        home_match = re.search(r"/teams/([A-Z]{3})/", home_anchor.get("href", ""))
        if not away_match or not home_match:
            continue

        try:
            away_score = int(away_pts_cell.get_text(strip=True))
            home_score = int(home_pts_cell.get_text(strip=True))
        except ValueError:
            continue

        games.append(_RawGame(
            game_id=game_id,
            date=date_iso,
            home_abbrev=_normalize_abbrev(home_match.group(1)),
            away_abbrev=_normalize_abbrev(away_match.group(1)),
            home_score=home_score,
            away_score=away_score,
        ))

    return games


# --- Box-score parsing -----------------------------------------------------

def parse_box_score(html: str, home_abbrev: str, away_abbrev: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Parse a BR box-score page and return per-team aggregated stats.

    BR has two tables per game with id "box-<TEAM>-game-basic". The team
    totals row is in <tfoot>.
    """
    soup = BeautifulSoup(html, "html.parser")

    def _team_totals(team_abbrev: str) -> Optional[Dict[str, Any]]:
        candidates = [team_abbrev]
        for br_ab, cfg_ab in BR_TO_CONFIG_ABBREV.items():
            if cfg_ab == team_abbrev:
                candidates.append(br_ab)

        for ab in candidates:
            table = _find_table(soup, f"box-{ab}-game-basic")
            if table is None:
                continue

            tfoot = table.find("tfoot")
            if tfoot is None:
                continue
            row = tfoot.find("tr")
            if row is None:
                continue

            out: Dict[str, Any] = {}
            for src, dst in BOXSCORE_FIELD_MAP.items():
                cell = row.find("td", {"data-stat": src})
                if cell is None:
                    out[dst] = None
                    continue
                text = cell.get_text(strip=True)
                if not text:
                    out[dst] = None
                    continue
                try:
                    out[dst] = int(text)
                except ValueError:
                    try:
                        out[dst] = float(text)
                    except ValueError:
                        out[dst] = None
            return out
        return None

    home_box = _team_totals(home_abbrev)
    away_box = _team_totals(away_abbrev)
    if home_box is None or away_box is None:
        return None

    return {"home": home_box, "away": away_box}


# --- Series construction ----------------------------------------------------

def build_series(games: List[_RawGame]) -> List[Dict[str, Any]]:
    """
    Group raw games into playoff series, oriented from higher seed's POV.

    Convention:
      - The team that hosts G1 of a series is the "higher seed"
      - All games are oriented from the higher seed's perspective
      - Margin is positive when the higher seed won
    """
    games_sorted = sorted(games, key=lambda g: g.date)

    by_pair: Dict[Tuple[str, str], List[_RawGame]] = {}
    for g in games_sorted:
        key = tuple(sorted([g.home_abbrev, g.away_abbrev]))
        by_pair.setdefault(key, []).append(g)

    series_list: List[Dict[str, Any]] = []
    for pair, pair_games in by_pair.items():
        pair_games_sorted = sorted(pair_games, key=lambda g: g.date)
        g1 = pair_games_sorted[0]
        higher_seed = g1.home_abbrev
        lower_seed = pair[0] if higher_seed == pair[1] else pair[1]

        higher_wins = 0
        oriented_games = []
        for g in pair_games_sorted:
            if g.home_abbrev == higher_seed:
                higher_was_home = True
                higher_score, lower_score = g.home_score, g.away_score
            else:
                higher_was_home = False
                higher_score, lower_score = g.away_score, g.home_score

            if higher_score > lower_score:
                higher_wins += 1

            oriented_games.append({
                "game_id": g.game_id,
                "date": g.date,
                "higher_seed_was_home": higher_was_home,
                "higher_seed_score": higher_score,
                "lower_seed_score": lower_score,
                "margin": float(higher_score - lower_score),
            })

        lower_wins = len(pair_games_sorted) - higher_wins

        series_list.append({
            "higher_seed": higher_seed,
            "lower_seed": lower_seed,
            "higher_seed_won": higher_wins > lower_wins,
            "higher_seed_wins": higher_wins,
            "lower_seed_wins": lower_wins,
            "games": oriented_games,
        })

    series_list.sort(key=lambda s: s["games"][0]["date"])
    for idx, s in enumerate(series_list):
        if idx < 8:
            s["round"] = 1
        elif idx < 12:
            s["round"] = 2
        elif idx < 14:
            s["round"] = 3
        else:
            s["round"] = 4

    return series_list


# --- Main fetcher class -----------------------------------------------------

class HistoricalDataFetcher:
    """Fetches and caches historical NBA playoff data from basketball-reference."""

    def __init__(
        self,
        seasons: List[str],
        cache_dir: Path = CACHE_DIR,
        request_delay: float = REQUEST_DELAY,
    ):
        self.seasons = seasons
        self.cache_dir = cache_dir
        self.request_delay = request_delay
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(self, force: bool = False, continue_on_error: bool = True) -> Dict[str, str]:
        """
        Fetch all seasons. Skips seasons already cached unless force=True.
        Returns {season: status} where status is "cached", "fetched", or
        "failed: <reason>".
        """
        results: Dict[str, str] = {}
        for season in self.seasons:
            if not force and self._series_cache_path(season).exists():
                log.info(f"  {season}: cached, skipping")
                results[season] = "cached"
                continue
            log.info(f"  {season}: fetching")
            try:
                self._fetch_season(season)
                results[season] = "fetched"
            except Exception as e:
                log.error(f"  {season}: fetch failed ({type(e).__name__}: {e})")
                results[season] = f"failed: {e}"
                if not continue_on_error:
                    raise
        return results

    def load_series(self, season: str) -> Dict[str, Any]:
        path = self._series_cache_path(season)
        if not path.exists():
            raise FileNotFoundError(f"No cached series for {season}")
        return json.loads(path.read_text())

    def load_ratings(self, season: str) -> Dict[str, float]:
        path = self._ratings_cache_path(season)
        if not path.exists():
            raise FileNotFoundError(f"No cached ratings for {season}")
        return json.loads(path.read_text()).get("ratings", {})

    def load_all(self) -> List[Tuple[str, Dict[str, Any], Dict[str, float]]]:
        out = []
        for season in self.seasons:
            try:
                series = self.load_series(season)
                ratings = self.load_ratings(season)
                out.append((season, series, ratings))
            except FileNotFoundError as e:
                log.warning(f"  {e}")
        return out

    def _series_cache_path(self, season: str) -> Path:
        d = self.cache_dir / season
        d.mkdir(parents=True, exist_ok=True)
        return d / "series.json"

    def _ratings_cache_path(self, season: str) -> Path:
        d = self.cache_dir / season
        d.mkdir(parents=True, exist_ok=True)
        return d / "ratings.json"

    def _fetch_season(self, season: str) -> None:
        year = _br_url_to_year(season)

        # Step 1: ratings
        log.info(f"    Fetching ratings ({year})")
        time.sleep(self.request_delay)
        ratings_html = _polite_get(
            f"https://www.basketball-reference.com/leagues/NBA_{year}_ratings.html"
        )
        ratings = parse_ratings_html(ratings_html)
        self._ratings_cache_path(season).write_text(json.dumps({
            "season": season,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "basketball-reference.com",
            "metric": "NRtg/A",
            "ratings": dict(sorted(ratings.items())),
        }, indent=2))
        log.info(f"      Got {len(ratings)} team ratings")

        # Step 2: playoff schedule
        log.info(f"    Fetching playoff schedule ({year})")
        time.sleep(self.request_delay)
        schedule_html = _polite_get(
            f"https://www.basketball-reference.com/playoffs/NBA_{year}_games.html"
        )
        raw_games = parse_playoff_schedule(schedule_html)
        log.info(f"      Got {len(raw_games)} playoff games")

        # Step 3: per-game box scores (the bulk of the work)
        log.info(f"    Fetching {len(raw_games)} box scores")
        boxes_by_game: Dict[str, Dict[str, Any]] = {}
        for idx, g in enumerate(raw_games, 1):
            time.sleep(self.request_delay)
            try:
                box_html = _polite_get(
                    f"https://www.basketball-reference.com/boxscores/{g.game_id}.html"
                )
                boxes = parse_box_score(box_html, g.home_abbrev, g.away_abbrev)
                if boxes:
                    boxes_by_game[g.game_id] = boxes
            except Exception as e:
                log.warning(f"      box {g.game_id} failed: {e}")
            if idx % 10 == 0:
                log.info(f"      {idx}/{len(raw_games)} box scores fetched")

        # Step 4: build series, attach box scores from higher-seed POV
        series_list = build_series(raw_games)
        for s in series_list:
            for game in s["games"]:
                box = boxes_by_game.get(game["game_id"])
                if box is None:
                    continue
                if game["higher_seed_was_home"]:
                    game["higher_seed_box"] = box["home"]
                    game["lower_seed_box"] = box["away"]
                else:
                    game["higher_seed_box"] = box["away"]
                    game["lower_seed_box"] = box["home"]

        self._series_cache_path(season).write_text(json.dumps({
            "season": season,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "basketball-reference.com",
            "series": series_list,
        }, indent=2))
        log.info(f"      Wrote {len(series_list)} series")
