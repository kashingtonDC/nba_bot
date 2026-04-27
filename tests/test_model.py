"""
Unit tests for the probability model.

Run with: pytest tests/
"""
from __future__ import annotations
import math
import sys
import os

# Allow `import model` from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import model as M


# --- phi (standard normal CDF) ---------------------------------------------

def test_phi_zero():
    assert abs(M.phi(0.0) - 0.5) < 1e-12


def test_phi_symmetry():
    for z in [0.5, 1.0, 1.5, 2.0]:
        assert abs(M.phi(z) + M.phi(-z) - 1.0) < 1e-12


def test_phi_known_values():
    # phi(1) ~ 0.8413, phi(2) ~ 0.9772
    assert abs(M.phi(1.0) - 0.8413) < 1e-3
    assert abs(M.phi(2.0) - 0.9772) < 1e-3


# --- per_game_win_prob ------------------------------------------------------

def test_equal_teams_neutral_court_is_50pct():
    p = M.per_game_win_prob(5.0, 5.0, favorite_is_home=True, hca=0.0)
    assert abs(p - 0.5) < 1e-12


def test_home_advantage_increases_win_prob():
    p_home = M.per_game_win_prob(5.0, 5.0, favorite_is_home=True)
    p_road = M.per_game_win_prob(5.0, 5.0, favorite_is_home=False)
    assert p_home > 0.5 > p_road
    # By symmetry around the equal-strength case, p_home + p_road should == 1
    assert abs(p_home + p_road - 1.0) < 1e-12


def test_better_team_more_likely_to_win_at_home():
    p = M.per_game_win_prob(7.0, 2.0, favorite_is_home=True)
    # 5-pt diff + 2.5 HCA = 7.5 expected margin / 11.5 sigma -> phi(0.652) ~ 0.743
    assert 0.70 < p < 0.78


def test_den_min_sanity():
    """DEN +5.2, MIN +3.0 -> matches the manual calc we did earlier."""
    p_home = M.per_game_win_prob(5.2, 3.0, favorite_is_home=True)
    p_road = M.per_game_win_prob(5.2, 3.0, favorite_is_home=False)
    # Expected: ~0.66 home, ~0.49 road
    assert 0.63 < p_home < 0.69
    assert 0.46 < p_road < 0.52


# --- Bayesian updating ------------------------------------------------------

def test_initial_belief_mean_is_diff():
    b = M.initial_belief(5.0, 2.0, sigma_theta=2.0)
    assert b.mean == 3.0
    # var = 2 * sigma_theta^2 = 8
    assert abs(b.var - 8.0) < 1e-12


def test_bayes_update_pulls_toward_observation():
    """A surprising blowout should pull the posterior toward that result."""
    prior = M.initial_belief(5.0, 5.0)  # equal strength
    # Favorite wins by 20 at home -> posterior should shift positive
    post = M.bayes_update(prior, observed_margin=20.0, favorite_was_home=True)
    assert post.mean > prior.mean


def test_bayes_update_reduces_variance():
    prior = M.initial_belief(5.0, 5.0)
    post = M.bayes_update(prior, observed_margin=5.0, favorite_was_home=True)
    assert post.var < prior.var
    assert post.var > 0


def test_bayes_update_accounts_for_hca():
    """A 3-point home win is weaker evidence than a 3-point road win."""
    prior = M.initial_belief(5.0, 5.0)
    post_home_win = M.bayes_update(prior, 3.0, favorite_was_home=True)
    post_road_win = M.bayes_update(prior, 3.0, favorite_was_home=False)
    # Road win is stronger evidence for fav being better
    assert post_road_win.mean > post_home_win.mean


def test_bayes_update_many_chains():
    """
    Three blowout road wins should shift the posterior up, but only modestly
    relative to a tight prior. This is desirable: σ_game = 11.5 is large
    relative to σ_theta = 2.0, so individual games are noisy evidence and
    the prior is sticky. This avoids overreacting to small samples — the
    exact pathology we discussed when I underestimated DEN's series chances
    after 3 playoff games.
    """
    prior = M.initial_belief(5.0, 5.0)  # equal strength, mean diff = 0
    games = [(10.0, False), (10.0, False), (10.0, False)]
    post = M.bayes_update_many(prior, games)
    # Posterior should shift upward but by a modest amount (~2 points).
    assert post.mean > 0
    assert post.mean < 5.0  # NOT a wild swing
    # Variance should shrink (we've gathered information).
    assert post.var < prior.var


# --- Series win probability -------------------------------------------------

def test_series_already_won():
    assert M.series_win_prob(0.5, 0.5, fav_wins=4, und_wins=0) == 1.0
    assert M.series_win_prob(0.5, 0.5, fav_wins=4, und_wins=3) == 1.0


def test_series_already_lost():
    assert M.series_win_prob(0.5, 0.5, fav_wins=0, und_wins=4) == 0.0


def test_series_coinflip_at_tied_with_no_hca():
    """Equal teams, no HCA effect (50/50 home and road) -> 50/50 from any tied state."""
    p = M.series_win_prob(0.5, 0.5, fav_wins=0, und_wins=0)
    assert abs(p - 0.5) < 1e-12
    p2 = M.series_win_prob(0.5, 0.5, fav_wins=2, und_wins=2)
    assert abs(p2 - 0.5) < 1e-12


def test_series_invalid_state_raises():
    with pytest.raises(ValueError):
        M.series_win_prob(0.5, 0.5, fav_wins=-1, und_wins=0)
    with pytest.raises(ValueError):
        M.series_win_prob(0.5, 0.5, fav_wins=0, und_wins=5)


def test_series_pre_play_with_real_probs():
    """DEN-MIN style: ~0.66 home, ~0.49 road. Pre-play favorite probability."""
    p = M.series_win_prob(0.66, 0.49, fav_wins=0, und_wins=0)
    # Should be well above 50% but not extreme
    assert 0.55 < p < 0.70


def test_series_down_3_1_road_team_dire():
    """Teams down 3-1 with no HCA edge should be in single digits."""
    # Favorite trailing 1-3, with home court (G5, G7 home) but only 50/50 winners
    p = M.series_win_prob(0.50, 0.32, fav_wins=1, und_wins=3)
    # Path: must win G5(H), G6(R), G7(H) = 0.5 * 0.32 * 0.5 = 0.08
    assert abs(p - 0.08) < 1e-9


def test_series_up_3_1_dominant():
    """Up 3-1, at home for G5 should be strong favorite."""
    p = M.series_win_prob(0.50, 0.32, fav_wins=3, und_wins=1)
    # win in G5 (0.5) OR lose G5 then win G6 OR lose G5,G6 then win G7
    # = 0.5 + 0.5*0.32 + 0.5*0.68*0.5 = 0.5 + 0.16 + 0.17 = 0.83
    assert abs(p - 0.83) < 0.01


def test_lakers_3_0_near_certain():
    """3-0 with reasonable home/road probs gives ~96-99%."""
    p = M.series_win_prob(0.65, 0.45, fav_wins=3, und_wins=0)
    # opponent must win 4 straight: 0.55 * 0.35 * 0.55 * 0.35 = ~0.037
    # so favorite ~0.963
    assert 0.95 < p < 0.98


def test_series_probabilities_sum_to_one():
    """For any state, P(fav wins) + P(und wins) should be ~1."""
    p_fav = M.series_win_prob(0.6, 0.4, fav_wins=2, und_wins=1)
    # If we flipped roles: undergod's home prob is fav's road prob, etc.
    # From underdog's perspective with 1-2 record, home_court_for_underdog=False
    p_und = M.series_win_prob(
        p_fav_home=1 - 0.4,  # underdog at their home = fav on road, fav wins 0.4 -> und 0.6
        p_fav_road=1 - 0.6,  # underdog on road = at fav's home, fav wins 0.6 -> und 0.4
        fav_wins=1, und_wins=2,  # underdog has 1 win, fav has 2
        favorite_has_home_court=False,  # underdog doesn't have home court
    )
    assert abs(p_fav + p_und - 1.0) < 1e-9


# --- Tail correction --------------------------------------------------------

def test_tail_correction_zero_in_middle():
    for price in [0.20, 0.40, 0.50, 0.60, 0.80]:
        assert M.tail_correction(price) == 0.0


def test_tail_correction_positive_for_high_prices():
    assert M.tail_correction(0.92) > 0
    # At exactly 1.0, full magnitude
    assert abs(M.tail_correction(1.0) - 0.02) < 1e-12
    # At exactly threshold, zero
    assert M.tail_correction(0.85) == 0.0


def test_tail_correction_negative_for_low_prices():
    assert M.tail_correction(0.05) < 0
    assert abs(M.tail_correction(0.0) + 0.02) < 1e-12
    assert M.tail_correction(0.15) == 0.0


def test_tail_correction_linear_ramp():
    """Should ramp linearly between threshold and extreme."""
    # Halfway between 0.85 and 1.0 is 0.925, should give half magnitude
    assert abs(M.tail_correction(0.925) - 0.01) < 1e-9


def test_tail_correction_invalid_input():
    with pytest.raises(ValueError):
        M.tail_correction(1.5)
    with pytest.raises(ValueError):
        M.tail_correction(-0.1)


# --- evaluate_series end-to-end --------------------------------------------

def test_evaluate_series_no_games():
    obs = M.evaluate_series(
        fav_net_rating=5.2, und_net_rating=3.0,
        fav_wins=0, und_wins=0,
    )
    assert 0.55 < obs.p_fav_series_raw < 0.70
    assert obs.p_fav_series_tail_adj is None  # no market price provided


def test_evaluate_series_with_completed_games():
    """DEN-MIN: 1-1 with the 17-pt blowout in G3 (DEN lost on road)."""
    games = [
        (11.0, True),    # G1: DEN won by 11 at home
        (-5.0, True),    # G2: DEN lost by 5 at home
        (-17.0, False),  # G3: DEN lost by 17 on road
    ]
    obs = M.evaluate_series(
        fav_net_rating=5.2, und_net_rating=3.0,
        fav_wins=1, und_wins=2,
        completed_games=games,
    )
    # Posterior should have shifted negative (DEN underperforming prior)
    assert obs.posterior_diff_mean < (5.2 - 3.0)
    # Series probability for DEN should be modest (down 1-2)
    assert 0.20 < obs.p_fav_series_raw < 0.45


def test_evaluate_series_market_price_drives_tail_adjustment():
    obs = M.evaluate_series(
        fav_net_rating=8.0, und_net_rating=2.0,
        fav_wins=3, und_wins=0,
        market_price=0.92,
    )
    # 3-0 lead with strong fav -> raw probability should be very high
    assert obs.p_fav_series_raw > 0.95
    # Market at 0.92 is in the tail -> tail-adjusted should be raw + correction
    assert obs.p_fav_series_tail_adj is not None
    assert obs.p_fav_series_tail_adj > obs.p_fav_series_raw


def test_evaluate_series_clamps_to_unit_interval():
    """Tail correction should never push probability above 1.0."""
    obs = M.evaluate_series(
        fav_net_rating=15.0, und_net_rating=-5.0,
        fav_wins=3, und_wins=0,
        market_price=0.99,
    )
    assert obs.p_fav_series_raw <= 1.0
    assert 0.0 <= (obs.p_fav_series_tail_adj or 0) <= 1.0
