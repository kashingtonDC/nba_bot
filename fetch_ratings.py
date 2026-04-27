# scripts/fetch_ratings.py — one-off helper
import requests
from html.parser import HTMLParser

resp = requests.get(
    "https://www.basketball-reference.com/leagues/NBA_2026_ratings.html",
    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"},
    timeout=15,
)
print(resp.status_code, len(resp.text))
# Save raw HTML so we can parse it
with open("ratings.html", "w") as f:
    f.write(resp.text)