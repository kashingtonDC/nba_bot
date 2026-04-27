"""
Read-only Kalshi public API client.

Uses unauthenticated endpoints — no API key required for market data,
orderbooks, or trade history. The "elections" subdomain is legacy; it serves
all Kalshi markets, not just elections.

Docs: https://docs.kalshi.com/getting_started/quick_start_market_data
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
NBA_SERIES_TICKER = "KXNBASERIES"


class KalshiError(RuntimeError):
    """Raised when the Kalshi API returns a non-200 or malformed response."""


def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    """GET with simple error handling."""
    url = f"{BASE}{path}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise KalshiError(f"Network error fetching {url}: {e}") from e
    if resp.status_code != 200:
        raise KalshiError(f"{url} returned {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as e:
        raise KalshiError(f"Bad JSON from {url}: {e}") from e


def list_open_nba_series_markets() -> List[Dict[str, Any]]:
    """
    Return all open markets in the KXNBASERIES series.

    Each market dict contains keys like `ticker`, `event_ticker`, `yes_bid`,
    `yes_ask`, `last_price`, `volume`, `volume_24h`, etc. Field names follow
    Kalshi's fixed-point migration; both legacy fields and `_dollars`/`_fp`
    variants may be present.
    """
    out: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    # Paginate fully.
    while True:
        params = {
            "series_ticker": NBA_SERIES_TICKER,
            "status": "open",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        data = _get("/markets", params=params)
        page = data.get("markets") or []
        out.extend(page)
        cursor = data.get("cursor")
        if not cursor or not page:
            break
    log.info(f"Found {len(out)} open NBA series markets")
    return out


def find_market_by_substring(markets: List[Dict[str, Any]], substring: str) -> Optional[Dict[str, Any]]:
    """Return the first market whose ticker contains the substring (case-insensitive)."""
    needle = substring.upper()
    for m in markets:
        ticker = (m.get("ticker") or "").upper()
        if needle in ticker:
            return m
    return None


def extract_prices(market: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Pull the price fields we care about, normalized to dollars (0..1).

    Kalshi has been migrating field names. We look for both the new
    `_dollars` fields and the legacy cents-as-int fields. Returns None for
    any field we can't read.
    """
    def from_dollars(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def from_cents(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v) / 100.0
        except (TypeError, ValueError):
            return None

    yes_bid = from_dollars(market.get("yes_bid_dollars")) or from_cents(market.get("yes_bid"))
    yes_ask = from_dollars(market.get("yes_ask_dollars")) or from_cents(market.get("yes_ask"))
    last = from_dollars(market.get("last_price_dollars")) or from_cents(market.get("last_price"))

    yes_mid = None
    if yes_bid is not None and yes_ask is not None:
        yes_mid = (yes_bid + yes_ask) / 2.0

    volume = market.get("volume_fp") or market.get("volume")
    volume_24h = market.get("volume_24h_fp") or market.get("volume_24h")
    try:
        volume = float(volume) if volume is not None else None
    except (TypeError, ValueError):
        volume = None
    try:
        volume_24h = float(volume_24h) if volume_24h is not None else None
    except (TypeError, ValueError):
        volume_24h = None

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_last": last,
        "yes_mid": yes_mid,
        "volume": volume,
        "volume_24h": volume_24h,
    }
