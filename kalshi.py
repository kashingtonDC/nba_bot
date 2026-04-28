"""
Read-only Kalshi public API client.

Uses unauthenticated endpoints — no API key required for market data,
orderbooks, or trade history. The "elections" subdomain is legacy; it serves
all Kalshi markets, not just elections.

Docs: https://docs.kalshi.com/getting_started/quick_start_market_data
"""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
NBA_SERIES_TICKER = "KXNBASERIES"

# Polite throttle between consecutive Kalshi API requests, in seconds.
# 100ms is a comfortable spacing for an unauthenticated public endpoint —
# even at our peak (one /markets call + N orderbook calls per bot run),
# total wait is well under a second.
INTER_REQUEST_DELAY = 0.1
_last_request_time: float = 0.0


class KalshiError(RuntimeError):
    """Raised when the Kalshi API returns a non-200 or malformed response."""


def _throttle() -> None:
    """Sleep enough that consecutive requests are spaced by INTER_REQUEST_DELAY."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < INTER_REQUEST_DELAY:
        time.sleep(INTER_REQUEST_DELAY - elapsed)
    _last_request_time = time.monotonic()


def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    """GET with simple error handling and a polite inter-request delay."""
    _throttle()
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


# --- Orderbook --------------------------------------------------------------

def get_orderbook(ticker: str) -> Dict[str, Any]:
    """
    Fetch the raw orderbook for a market.

    Kalshi's orderbook only exposes bids on each side (yes/no). This is a
    consequence of how binary contracts work: a YES ask at 92¢ is the same
    object as a NO bid at 8¢ (price + complement = $1). The exchange chooses
    one canonical representation.

    To get a full YES-side bid/ask view, see `reconstruct_yes_orderbook`.

    Kalshi's response uses the key "orderbook_fp" (fixed-point) in the
    current API version. Older docs sometimes refer to plain "orderbook";
    we accept either to be defensive against future API changes.

    The inner dict has keys "yes_dollars"/"no_dollars" (current) or
    "yes"/"no" (legacy) — see reconstruct_yes_orderbook for the format.

    Returns an empty dict on missing/malformed data rather than raising —
    orderbooks for settled or zero-liquidity markets may legitimately be
    empty.
    """
    try:
        payload = _get(f"/markets/{ticker}/orderbook")
    except KalshiError as e:
        log.warning(f"orderbook fetch failed for {ticker}: {e}")
        return {}

    inner = payload.get("orderbook_fp") or payload.get("orderbook")
    if not isinstance(inner, dict):
        return {}
    return inner


def reconstruct_yes_orderbook(orderbook: Dict[str, Any]) -> Dict[str, List[Tuple[float, int]]]:
    """
    Convert Kalshi's bid-only-on-both-sides format into a unified YES-side
    bid/ask view.

    Two input formats are supported:

    Current (fixed-point dollar strings):
        {
          "yes_dollars": [["0.5000", "1200.00"], ["0.4900", "800.00"], ...],
          "no_dollars":  [["0.4800", "1500.00"], ...],
        }

    Legacy (integer cents):
        {
          "yes": [[50, 1200], [49, 800], ...],
          "no":  [[48, 1500], ...],
        }

    Output format:
        {
          "bids": [(0.50, 1200), (0.49, 800), ...],  # YES bids, sorted desc
          "asks": [(0.52, 1500), ...],               # YES asks (= 1 - NO bid), sorted asc
        }

    Sizes are floored to int — Kalshi sometimes returns fractional sizes
    like "93093.39" on fractional-trading-enabled markets. We round down
    to be conservative about depth.
    """
    # Detect format by which keys are present. Default to legacy if neither.
    has_dollars = "yes_dollars" in orderbook or "no_dollars" in orderbook

    if has_dollars:
        yes_bids_raw = orderbook.get("yes_dollars") or []
        no_bids_raw = orderbook.get("no_dollars") or []
    else:
        yes_bids_raw = orderbook.get("yes") or []
        no_bids_raw = orderbook.get("no") or []

    def _parse_price(p: Any) -> Optional[float]:
        """Returns price in dollars [0,1]."""
        try:
            v = float(p)
        except (TypeError, ValueError):
            return None
        # Heuristic: if value > 1, must be cents (legacy format)
        return v if v <= 1.0 else v / 100.0

    def _parse_size(s: Any) -> Optional[int]:
        try:
            n = int(float(s))
            return n if n > 0 else None
        except (TypeError, ValueError):
            return None

    bids: List[Tuple[float, int]] = []
    for entry in yes_bids_raw:
        try:
            price = _parse_price(entry[0])
            size = _parse_size(entry[1])
            if price is None or size is None:
                continue
            bids.append((price, size))
        except (TypeError, IndexError):
            continue
    bids.sort(key=lambda x: -x[0])

    asks: List[Tuple[float, int]] = []
    for entry in no_bids_raw:
        try:
            no_price = _parse_price(entry[0])
            size = _parse_size(entry[1])
            if no_price is None or size is None:
                continue
            asks.append((1.0 - no_price, size))
        except (TypeError, IndexError):
            continue
    asks.sort(key=lambda x: x[0])

    return {"bids": bids, "asks": asks}


def summarize_orderbook(book: Dict[str, List[Tuple[float, int]]]) -> Dict[str, Optional[float]]:
    """
    Compute summary stats from a reconstructed YES-side orderbook.

    Returns the columns we'll log to the database: best bid/ask + sizes,
    spread, mid, and depth at the top of the book (sum of contracts at the
    best bid and best ask, a coarse measure of immediate liquidity).
    """
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    best_bid_price, best_bid_size = (bids[0] if bids else (None, None))
    best_ask_price, best_ask_size = (asks[0] if asks else (None, None))

    spread = None
    mid = None
    if best_bid_price is not None and best_ask_price is not None:
        spread = best_ask_price - best_bid_price
        mid = (best_bid_price + best_ask_price) / 2.0

    # Depth at top of book: contracts available at the best bid + best ask.
    # A coarse but meaningful liquidity proxy.
    depth_top = None
    if best_bid_size is not None or best_ask_size is not None:
        depth_top = (best_bid_size or 0) + (best_ask_size or 0)

    return {
        "ob_best_bid": best_bid_price,
        "ob_best_bid_size": best_bid_size,
        "ob_best_ask": best_ask_price,
        "ob_best_ask_size": best_ask_size,
        "ob_spread": spread,
        "ob_mid": mid,
        "ob_depth_top": depth_top,
    }
