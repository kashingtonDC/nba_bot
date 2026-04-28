"""
Unit tests for kalshi orderbook reconstruction.

These tests exercise the math of converting Kalshi's bid-only-on-both-sides
format into a unified YES-side bid/ask view. No network — pure logic.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock requests before importing kalshi (kalshi imports requests at top)
class _FakeReq:
    class RequestException(Exception):
        pass
    @staticmethod
    def get(*a, **k):
        raise NotImplementedError("Network not available in tests")

if "requests" not in sys.modules:
    sys.modules["requests"] = _FakeReq()

import kalshi as K


def test_reconstruct_basic_bid_ask():
    """Standard case (legacy integer cents): bids on both sides, computes YES bids and asks."""
    raw = {
        "yes": [[50, 1200], [49, 800]],
        "no":  [[48, 1500], [47, 900]],
    }
    book = K.reconstruct_yes_orderbook(raw)
    # YES bids stay as-is, sorted descending by price
    assert book["bids"][0] == (0.50, 1200)
    assert book["bids"][1] == (0.49, 800)
    # NO bids invert: NO@48 -> YES ask@52, NO@47 -> YES ask@53
    # YES asks sorted ascending by price
    assert book["asks"][0] == (0.52, 1500)
    assert book["asks"][1] == (0.53, 900)


def test_reconstruct_dollar_format():
    """Current Kalshi format: dollar strings."""
    raw = {
        "yes_dollars": [["0.5000", "1200.00"], ["0.4900", "800.00"]],
        "no_dollars":  [["0.4800", "1500.00"], ["0.4700", "900.00"]],
    }
    book = K.reconstruct_yes_orderbook(raw)
    assert book["bids"][0] == (0.50, 1200)
    assert book["bids"][1] == (0.49, 800)
    assert book["asks"][0] == (0.52, 1500)
    assert book["asks"][1] == (0.53, 900)


def test_reconstruct_dollar_format_with_fractional_sizes():
    """Fractional sizes like '93093.39' are floored to int."""
    raw = {
        "yes_dollars": [["0.0800", "93093.39"]],
        "no_dollars":  [["0.0700", "136495.00"]],
    }
    book = K.reconstruct_yes_orderbook(raw)
    assert book["bids"] == [(0.08, 93093)]   # floored
    # 1 - 0.07 = 0.9299999... in float; compare with tolerance
    assert len(book["asks"]) == 1
    assert abs(book["asks"][0][0] - 0.93) < 1e-9
    assert book["asks"][0][1] == 136495


def test_reconstruct_empty_book():
    """Both sides empty — common for settled or zero-liquidity markets."""
    book = K.reconstruct_yes_orderbook({"yes": [], "no": []})
    assert book == {"bids": [], "asks": []}


def test_reconstruct_one_sided():
    """Only YES bids, no NO bids — bids exist but no asks."""
    book = K.reconstruct_yes_orderbook({"yes": [[50, 100]], "no": []})
    assert book["bids"] == [(0.50, 100)]
    assert book["asks"] == []


def test_reconstruct_skips_malformed_entries():
    """Defensive: bad entries should be skipped, not crash."""
    raw = {
        "yes": [[50, 1200], "garbage", [None, 100], [49]],
        "no":  [[48, 500]],
    }
    book = K.reconstruct_yes_orderbook(raw)
    # Only the well-formed entry survives on the bids side
    assert book["bids"] == [(0.50, 1200)]
    assert book["asks"] == [(0.52, 500)]


def test_reconstruct_handles_missing_keys():
    """Defensive: missing yes/no keys treated as empty arrays."""
    book = K.reconstruct_yes_orderbook({})
    assert book == {"bids": [], "asks": []}

    book2 = K.reconstruct_yes_orderbook({"yes": [[50, 100]]})  # no 'no' key
    assert book2["bids"] == [(0.50, 100)]
    assert book2["asks"] == []


def test_summarize_basic():
    """Summary of a standard book produces sensible spread/mid/depth."""
    book = {
        "bids": [(0.50, 1200), (0.49, 800)],
        "asks": [(0.52, 1500), (0.53, 900)],
    }
    s = K.summarize_orderbook(book)
    assert s["ob_best_bid"] == 0.50
    assert s["ob_best_bid_size"] == 1200
    assert s["ob_best_ask"] == 0.52
    assert s["ob_best_ask_size"] == 1500
    assert abs(s["ob_spread"] - 0.02) < 1e-9
    assert abs(s["ob_mid"] - 0.51) < 1e-9
    assert s["ob_depth_top"] == 2700  # 1200 + 1500


def test_summarize_empty():
    """Empty book gives all-None summary."""
    s = K.summarize_orderbook({"bids": [], "asks": []})
    assert s["ob_best_bid"] is None
    assert s["ob_best_ask"] is None
    assert s["ob_spread"] is None
    assert s["ob_mid"] is None
    assert s["ob_depth_top"] is None


def test_summarize_one_sided_bids_only():
    """Bids only — spread/mid undefined, depth uses what we have."""
    s = K.summarize_orderbook({"bids": [(0.50, 1200)], "asks": []})
    assert s["ob_best_bid"] == 0.50
    assert s["ob_best_ask"] is None
    assert s["ob_spread"] is None      # can't compute without ask
    assert s["ob_mid"] is None
    assert s["ob_depth_top"] == 1200   # just the bid side


def test_summarize_one_sided_asks_only():
    s = K.summarize_orderbook({"bids": [], "asks": [(0.52, 800)]})
    assert s["ob_best_bid"] is None
    assert s["ob_best_ask"] == 0.52
    assert s["ob_spread"] is None
    assert s["ob_mid"] is None
    assert s["ob_depth_top"] == 800
