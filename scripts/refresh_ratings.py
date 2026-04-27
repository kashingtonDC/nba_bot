"""
Refresh team net ratings from basketball-reference.

Writes to `ratings.json` in the project root. The bot's config will read
from this file if it exists, falling back to hardcoded values otherwise.

Usage
-----
    python scripts/refresh_ratings.py                # default: current season
    python scripts/refresh_ratings.py --season 2026  # explicit season year
    python scripts/refresh_ratings.py --raw          # use raw NRtg, not NRtg/A

Design notes
------------
* Source: https://www.basketball-reference.com/leagues/NBA_<YEAR>_ratings.html
  The HTML page contains a table with team-level offensive, defensive, and
  net ratings, both raw and strength-of-schedule-adjusted (NRtg/A).
  We default to adjusted because it's more predictive.

* basketball-reference's robots.txt is permissive for human-rate access but
  not bot-friendly. This script makes a SINGLE request per invocation,
  presents a real browser User-Agent, and is designed to be run nightly at
  most. Don't increase the cadence without checking BR's policy.

* Team name -> abbreviation mapping comes from BR's team URLs in the page,
  so we don't need to hardcode it. BR uses the same 3-letter abbreviations
  in the HTML that we use in config (with two exceptions: BR uses CHO for
  the Hornets and BRK for the Nets; NBA uses CHA and BKN. We normalize.)

* Output schema (ratings.json):
    {
      "season": 2026,
      "fetched_at": "2026-04-26T...",
      "source": "basketball-reference.com",
      "metric": "NRtg/A",                  # or "NRtg" if --raw
      "ratings": {
        "OKC": 11.1,
        "BOS":  6.8,
        ...
      }
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup


# --- Constants --------------------------------------------------------------

BR_URL_TEMPLATE = "https://www.basketball-reference.com/leagues/NBA_{season}_ratings.html"

# Browser-like UA. BR serves 403 to default requests UA.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# BR uses BRK and CHO; we normalize to BKN and CHA to match our config.
BR_TO_CONFIG_ABBREV = {
    "BRK": "BKN",
    "CHO": "CHA",
    "PHO": "PHX",  # PHO appears in some BR pages, PHX in others
}


log = logging.getLogger("refresh_ratings")


# --- HTTP fetch -------------------------------------------------------------

def fetch_ratings_html(season: int, timeout: float = 20.0) -> str:
    """
    Fetch the raw HTML of BR's team ratings page for the given season.

    `season` is the year the season ENDS in (e.g., 2026 for 2025-26).
    Returns the full HTML as a string. Raises requests.HTTPError on non-200.
    """
    url = BR_URL_TEMPLATE.format(season=season)
    log.info(f"Fetching {url}")
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=timeout,
    )
    resp.raise_for_status()
    log.info(f"Got {len(resp.text)} bytes")
    return resp.text


# --- Parser -----------------------------------------------------------------

def parse_ratings(html: str, metric: str = "NRtg/A") -> Dict[str, float]:
    """
    Parse the BR ratings page HTML and return {abbrev: rating}.

    `metric` is one of "NRtg" (raw) or "NRtg/A" (adjusted). The HTML uses
    the data-stat attribute "n_rtg" for raw and "n_rtg_a" for adjusted on
    each <td>.
    """
    if metric not in ("NRtg", "NRtg/A"):
        raise ValueError(f"metric must be 'NRtg' or 'NRtg/A', got {metric!r}")

    data_stat = "n_rtg_a" if metric == "NRtg/A" else "n_rtg"

    soup = BeautifulSoup(html, "html.parser")

    # BR sometimes wraps the actual data table inside an HTML comment to
    # work around its own CMS. Look for <table id="ratings"> in raw HTML
    # and inside comments.
    table = soup.find("table", id="ratings")
    if table is None:
        # Fall back to scanning HTML comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if 'id="ratings"' in comment:
                inner = BeautifulSoup(comment, "html.parser")
                table = inner.find("table", id="ratings")
                if table is not None:
                    break

    if table is None:
        raise RuntimeError("Could not find ratings table in BR HTML")

    ratings: Dict[str, float] = {}

    tbody = table.find("tbody") or table
    for row in tbody.find_all("tr"):
        # Skip header rows that BR injects mid-table.
        if "thead" in (row.get("class") or []):
            continue

        # Team abbreviation is in the team cell's anchor href: /teams/LAL/2026.html
        team_cell = row.find("td", {"data-stat": "team"}) or row.find("th", {"data-stat": "team"})
        if team_cell is None:
            continue

        anchor = team_cell.find("a")
        if anchor is None or not anchor.get("href"):
            continue

        m = re.search(r"/teams/([A-Z]{3})/", anchor["href"])
        if not m:
            continue
        abbrev = m.group(1)
        abbrev = BR_TO_CONFIG_ABBREV.get(abbrev, abbrev)

        # Find the rating cell
        rating_cell = row.find("td", {"data-stat": data_stat})
        if rating_cell is None:
            continue

        text = rating_cell.get_text(strip=True)
        if not text:
            continue

        try:
            ratings[abbrev] = float(text)
        except ValueError:
            log.warning(f"Could not parse rating for {abbrev!r}: {text!r}")

    if len(ratings) < 25:
        # NBA has 30 teams; if we got fewer than 25 something went wrong with parsing
        raise RuntimeError(
            f"Only parsed {len(ratings)} teams from ratings page; expected ~30. "
            f"BR may have changed page structure."
        )

    log.info(f"Parsed ratings for {len(ratings)} teams")
    return ratings


# --- Output -----------------------------------------------------------------

def write_ratings_json(
    output_path: Path,
    ratings: Dict[str, float],
    season: int,
    metric: str,
) -> None:
    """Write the ratings dict to a JSON file with metadata."""
    payload = {
        "season": season,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "basketball-reference.com",
        "metric": metric,
        "ratings": dict(sorted(ratings.items())),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    log.info(f"Wrote {len(ratings)} ratings to {output_path}")


# --- CLI --------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--season", type=int, default=2026,
        help="Season-end year (e.g. 2026 for the 2025-26 season). Default: 2026.",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Use raw NRtg instead of strength-of-schedule-adjusted NRtg/A. "
             "Default is adjusted, which is more predictive.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "ratings.json",
        help="Where to write the JSON. Default: <repo>/ratings.json",
    )
    args = parser.parse_args()

    metric = "NRtg" if args.raw else "NRtg/A"

    try:
        html = fetch_ratings_html(args.season)
    except requests.HTTPError as e:
        log.error(f"HTTP error fetching BR: {e}")
        log.error("If this is 403/429, basketball-reference may be blocking. "
                  "Manual fallback: copy values from the page into config.TEAM_NET_RATINGS.")
        return 2
    except requests.RequestException as e:
        log.error(f"Network error fetching BR: {e}")
        return 2

    try:
        ratings = parse_ratings(html, metric=metric)
    except RuntimeError as e:
        log.error(f"Parse error: {e}")
        log.error("Saving the raw HTML to ratings_debug.html for inspection")
        Path("ratings_debug.html").write_text(html)
        return 3

    write_ratings_json(args.output, ratings, args.season, metric)

    # Print a summary the user can sanity-check
    print()
    print(f"Top 10 by {metric}:")
    for team, val in sorted(ratings.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {team:5s} {val:+.2f}")
    print()
    print(f"Bottom 5:")
    for team, val in sorted(ratings.items(), key=lambda kv: kv[1])[:5]:
        print(f"  {team:5s} {val:+.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
