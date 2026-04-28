"""
Diagnostic: fetch one BR page and report what's actually in it.

Usage:
    python scripts/inspect_br.py ratings 2024
    python scripts/inspect_br.py schedule 2024
    python scripts/inspect_br.py box 202404200BOS

Saves the raw HTML to /tmp/br_<kind>_<id>.html so you can inspect manually.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup, Comment

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch(url: str) -> str:
    print(f"Fetching: {url}")
    resp = requests.get(
        url,
        headers={"User-Agent": UA, "Accept": "text/html"},
        timeout=30,
    )
    print(f"  Status: {resp.status_code}")
    print(f"  Length: {len(resp.text):,} bytes")
    return resp.text


def report(html: str, kind: str, ident: str) -> None:
    out_path = f"/tmp/br_{kind}_{ident}.html"
    Path(out_path).write_text(html)
    print(f"  Saved raw HTML to: {out_path}")

    soup = BeautifulSoup(html, "html.parser")

    # 1. Tables found directly in the document
    direct_tables = soup.find_all("table")
    print(f"\n  Direct <table> tags found: {len(direct_tables)}")
    for t in direct_tables[:10]:
        tid = t.get("id") or "(no id)"
        n_rows = len(t.find_all("tr"))
        print(f"    id={tid!r}  rows={n_rows}")

    # 2. Tables hidden inside HTML comments (BR's known trick)
    commented_tables = []
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            inner = BeautifulSoup(c, "html.parser")
            for t in inner.find_all("table"):
                tid = t.get("id") or "(no id)"
                n_rows = len(t.find_all("tr"))
                commented_tables.append((tid, n_rows))
    print(f"\n  Tables inside HTML comments: {len(commented_tables)}")
    for tid, n_rows in commented_tables[:15]:
        print(f"    id={tid!r}  rows={n_rows}")

    # 3. Sample data-stat attributes from the first interesting table
    print(f"\n  data-stat attributes seen (sampling first 3 tables with rows):")
    seen_stats = set()
    all_tables = direct_tables[:]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            inner = BeautifulSoup(c, "html.parser")
            all_tables.extend(inner.find_all("table"))

    for t in all_tables[:3]:
        rows = t.find_all("tr")
        if not rows:
            continue
        for cell in rows[0].find_all(["td", "th"]):
            ds = cell.get("data-stat")
            if ds:
                seen_stats.add(ds)
        for row in rows[1:3]:
            for cell in row.find_all(["td", "th"]):
                ds = cell.get("data-stat")
                if ds:
                    seen_stats.add(ds)
    print(f"    {sorted(seen_stats)[:50]}")

    # 4. Any <a> hrefs pointing to /teams/<3-letter>/
    import re
    team_links = re.findall(r'/teams/([A-Z]{3})/', html)
    print(f"\n  Team-page links found: {len(team_links)} total, "
          f"{len(set(team_links))} unique")
    print(f"    First few: {sorted(set(team_links))[:10]}")

    # 5. Box-score links (only relevant for schedule pages)
    box_links = re.findall(r'/boxscores/([0-9A-Za-z]+)\.html', html)
    if box_links:
        print(f"\n  Box-score links: {len(box_links)} total, "
              f"{len(set(box_links))} unique")
        print(f"    First few: {sorted(set(box_links))[:5]}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    kind = sys.argv[1]
    ident = sys.argv[2]

    if kind == "ratings":
        url = f"https://www.basketball-reference.com/leagues/NBA_{ident}_ratings.html"
    elif kind == "schedule":
        url = f"https://www.basketball-reference.com/playoffs/NBA_{ident}_games.html"
    elif kind == "box":
        url = f"https://www.basketball-reference.com/boxscores/{ident}.html"
    else:
        print(f"Unknown kind: {kind}; use 'ratings', 'schedule', or 'box'")
        return 1

    html = fetch(url)
    report(html, kind, ident)
    return 0


if __name__ == "__main__":
    sys.exit(main())
