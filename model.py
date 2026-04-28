"""
Probability model for NBA playoff series outcomes.

All functions in this module are pure: same inputs -> same outputs, no I/O,
no side effects. This makes the model independently testable and reusable.

Design overview
---------------
1. Per-game probability comes from a normal-distribution model:
       expected_margin = (fav_NRtg - und_NRtg) + HCA_signed
       P(fav wins) = Phi(expected_margin / SIGMA_GAME)

2. Bayesian updating after each completed game treats the team-strength
   differential as N(prior_mean, prior_var) and each observed game margin as
       observed_margin ~ N(diff + HCA_signed, SIGMA_GAME^2)
   This is a standard normal-normal conjugate update.

3. Series probability enumerates all paths through the remaining games
   given current series state and the home/away pattern (2-2-1-1-1).

4. Tail correction applies a small structural adjustment near the price tails
   (>=85c, <=15c) reflecting the documented longshot/favorite bias on
   prediction markets. This is logged separately so its value can be tested.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import erf, sqrt
from typing import Iterable, List, Optional, Tuple


# --- Model constants (overridable via config) -------------------------------

DEFAULT_HCA = 2.5            # home court advantage in points
DEFAULT_SIGMA_GAME = 11.5    # std dev of single-game margin
DEFAULT_SIGMA_THETA = 2.0    # prior std dev on each team's true net rating
DEFAULT_KAPPA = 0.0          # Four Factors variance-weighting strength
                             # (0.0 = current behavior, ignore Four Factors)
DEFAULT_PRIOR_REGRESSION = 1.0  # multiplier on the rating differential
                                # (1.0 = use NRtg as-is; 0.6 shrinks the gap by 40%)


# --- Standard normal CDF (no scipy dependency) ------------------------------

def phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# --- Per-game probability ---------------------------------------------------

def per_game_win_prob(
    fav_net_rating: float,
    und_net_rating: float,
    favorite_is_home: bool,
    sigma_game: float = DEFAULT_SIGMA_GAME,
    hca: float = DEFAULT_HCA,
) -> float:
    """
    Probability that the favorite wins a single game.

    Uses the standard NBA-analytics mapping from net-rating differential to
    win probability via a normal CDF on expected margin.
    """
    diff = fav_net_rating - und_net_rating
    hca_signed = hca if favorite_is_home else -hca
    expected_margin = diff + hca_signed
    return phi(expected_margin / sigma_game)


# --- Bayesian update --------------------------------------------------------

@dataclass(frozen=True)
class BeliefDiff:
    """Posterior on the (fav - und) net-rating differential."""
    mean: float
    var: float

    @property
    def std(self) -> float:
        return sqrt(self.var)


def initial_belief(
    fav_net_rating: float,
    und_net_rating: float,
    sigma_theta: float = DEFAULT_SIGMA_THETA,
) -> BeliefDiff:
    """
    Prior belief over the (fav - und) net-rating differential.

    Each team's true rating is treated as N(observed, sigma_theta^2). The
    differential is therefore N(observed_diff, 2 * sigma_theta^2).
    """
    return BeliefDiff(
        mean=fav_net_rating - und_net_rating,
        var=2.0 * sigma_theta * sigma_theta,
    )


def bayes_update(
    belief: BeliefDiff,
    observed_margin: float,
    favorite_was_home: bool,
    sigma_game: float = DEFAULT_SIGMA_GAME,
    hca: float = DEFAULT_HCA,
    sigma_game_squared: Optional[float] = None,
) -> BeliefDiff:
    """
    Single Bayesian update on the differential after one observed game.

    Each game gives an observation:
        observed_margin = diff + HCA_signed + epsilon,  epsilon ~ N(0, sigma^2)
    so we update toward (observed_margin - HCA_signed).

    Standard normal-normal conjugate result:
        posterior_var  = 1 / (1/prior_var + 1/sigma^2)
        posterior_mean = posterior_var * (prior_mean/prior_var + (m - h)/sigma^2)

    The likelihood variance can be specified two ways:
      - `sigma_game` (scalar, default DEFAULT_SIGMA_GAME). Backward-
        compatible mode: one constant for all games.
      - `sigma_game_squared` (per-game). When provided, overrides
        sigma_game. Used by `bayes_update_many` when Four Factors
        variance weighting is active.
    """
    hca_signed = hca if favorite_was_home else -hca
    likelihood_mean = observed_margin - hca_signed

    if sigma_game_squared is None:
        sigma_game_squared = sigma_game * sigma_game

    inv_prior = 1.0 / belief.var
    inv_lik = 1.0 / sigma_game_squared

    posterior_var = 1.0 / (inv_prior + inv_lik)
    posterior_mean = posterior_var * (belief.mean * inv_prior + likelihood_mean * inv_lik)

    return BeliefDiff(mean=posterior_mean, var=posterior_var)


def per_game_sigma_squared(
    observed_margin: float,
    expected_margin_from_factors: Optional[float],
    sigma_game: float = DEFAULT_SIGMA_GAME,
    kappa: float = DEFAULT_KAPPA,
) -> float:
    """
    Compute the likelihood variance for one game, optionally inflated based
    on how far the actual margin diverged from what Four Factors predicted.

    Idea: a game where the favorite won by 17 with sustainable Four Factors
    is strong evidence of team strength. A game where they won by 17 on
    65% eFG (unrepeatable shooting) is weak evidence. The Bayesian update
    should weight the latter less.

    Formula:
        sigma_squared = sigma_game^2 + kappa * (observed - expected)^2

    With kappa = 0 (default) this reduces to the original constant variance.
    With kappa > 0, fluky games get larger variance and contribute less to
    the posterior.

    If `expected_margin_from_factors` is None (Four Factors data missing),
    we fall back to the constant sigma_game^2 — preserving backward
    compatibility for games we have no box score for.
    """
    base = sigma_game * sigma_game
    if expected_margin_from_factors is None or kappa <= 0.0:
        return base
    diff = observed_margin - expected_margin_from_factors
    return base + kappa * diff * diff


def bayes_update_many(
    initial: BeliefDiff,
    games: Iterable[Tuple[float, bool]],
    sigma_game: float = DEFAULT_SIGMA_GAME,
    hca: float = DEFAULT_HCA,
    expected_margins: Optional[Iterable[Optional[float]]] = None,
    kappa: float = DEFAULT_KAPPA,
) -> BeliefDiff:
    """
    Apply Bayesian updates over a sequence of completed games.

    `games` is an iterable of (observed_margin, favorite_was_home) tuples.

    `expected_margins`, if provided, is an iterable of the same length
    giving the Four Factors-implied margin for each game (or None for
    games where the box score is unavailable). When provided alongside
    kappa > 0, the per-game likelihood variance is inflated based on the
    divergence between observed and expected margin — fluky games are
    downweighted.

    With kappa = 0 (the default) or expected_margins = None, this reduces
    to the original constant-variance update.
    """
    games_list = list(games)
    if expected_margins is None:
        em_list: list = [None] * len(games_list)
    else:
        em_list = list(expected_margins)
        if len(em_list) != len(games_list):
            raise ValueError(
                f"expected_margins length ({len(em_list)}) must match "
                f"games length ({len(games_list)})"
            )

    belief = initial
    for (margin, fav_home), em in zip(games_list, em_list):
        sigma2 = per_game_sigma_squared(margin, em, sigma_game=sigma_game, kappa=kappa)
        belief = bayes_update(
            belief, margin, fav_home,
            hca=hca,
            sigma_game_squared=sigma2,
        )
    return belief


def per_game_win_prob_from_belief(
    belief: BeliefDiff,
    favorite_is_home: bool,
    sigma_game: float = DEFAULT_SIGMA_GAME,
    hca: float = DEFAULT_HCA,
) -> float:
    """
    P(favorite wins) for a single upcoming game, integrating over the belief
    on the differential.

    Marginalizing N(diff_mean, diff_var) through the per-game normal model:
        margin = diff + HCA_signed + epsilon,  epsilon ~ N(0, sigma_game^2)
        P(margin > 0) = Phi( (diff_mean + HCA_signed) / sqrt(diff_var + sigma_game^2) )
    """
    hca_signed = hca if favorite_is_home else -hca
    expected = belief.mean + hca_signed
    total_std = sqrt(belief.var + sigma_game * sigma_game)
    return phi(expected / total_std)


# --- Series-level probability via path enumeration --------------------------

# Best-of-7 with the 2-2-1-1-1 home pattern. The home team for each game
# index (1..7) depends on which team holds home-court advantage. From the
# higher seed's perspective: True if higher seed is at home that game.
HIGHER_SEED_HOME_BY_GAME = {
    1: True,   # G1 at higher seed
    2: True,   # G2 at higher seed
    3: False,  # G3 at lower seed
    4: False,  # G4 at lower seed
    5: True,   # G5 at higher seed
    6: False,  # G6 at lower seed
    7: True,   # G7 at higher seed
}


def series_win_prob(
    p_fav_home: float,
    p_fav_road: float,
    fav_wins: int,
    und_wins: int,
    favorite_has_home_court: bool = True,
) -> float:
    """
    Probability that the favorite wins the series, given current state and
    per-game win probabilities at home and on the road.

    `fav_wins` and `und_wins` are wins so far. The next game number is
    fav_wins + und_wins + 1. The home pattern is 2-2-1-1-1 starting with the
    higher-seeded team at home.

    If `favorite_has_home_court` is True (the typical case — higher seed is
    favored), HIGHER_SEED_HOME_BY_GAME applies directly. If False (rare:
    a lower-seeded team is the favorite), the pattern is mirrored.
    """
    if fav_wins < 0 or und_wins < 0 or fav_wins > 4 or und_wins > 4:
        raise ValueError(f"Invalid series state: {fav_wins}-{und_wins}")
    if fav_wins == 4 and und_wins == 4:
        raise ValueError("Both teams cannot have 4 wins")
    if fav_wins == 4:
        return 1.0
    if und_wins == 4:
        return 0.0

    next_game = fav_wins + und_wins + 1

    def is_fav_home(game_number: int) -> bool:
        higher_seed_home = HIGHER_SEED_HOME_BY_GAME[game_number]
        return higher_seed_home if favorite_has_home_court else (not higher_seed_home)

    # Recursive enumeration with memoization.
    cache: dict = {}

    def recurse(fw: int, uw: int, game_num: int) -> float:
        if fw >= 4:
            return 1.0
        if uw >= 4:
            return 0.0
        key = (fw, uw, game_num)
        if key in cache:
            return cache[key]
        p_win = p_fav_home if is_fav_home(game_num) else p_fav_road
        result = (
            p_win * recurse(fw + 1, uw, game_num + 1)
            + (1.0 - p_win) * recurse(fw, uw + 1, game_num + 1)
        )
        cache[key] = result
        return result

    return recurse(fav_wins, und_wins, next_game)


# --- Tail correction --------------------------------------------------------

def tail_correction(
    market_price: float,
    magnitude_pp: float = 0.02,
    inner_threshold_low: float = 0.15,
    inner_threshold_high: float = 0.85,
) -> float:
    """
    Probability adjustment to apply at the tails of the market price.

    Returns a positive value when buying YES near 100c (market underprices
    near-certainties) and a negative value when buying YES near 0c (market
    overprices longshots). The adjustment ramps linearly from 0 at the inner
    threshold to ±magnitude_pp at the outer extreme (0 or 1).

    Both raw and tail-adjusted model probabilities are logged separately so
    the value of this correction can be measured against ground truth.
    """
    if not 0.0 <= market_price <= 1.0:
        raise ValueError(f"market_price must be in [0,1], got {market_price}")

    if market_price >= inner_threshold_high:
        # Underpriced near-certainty -> nudge our probability up
        ramp = (market_price - inner_threshold_high) / (1.0 - inner_threshold_high)
        return +magnitude_pp * ramp
    if market_price <= inner_threshold_low:
        # Overpriced longshot -> nudge our probability down
        ramp = (inner_threshold_low - market_price) / inner_threshold_low
        return -magnitude_pp * ramp
    return 0.0


# --- Top-level convenience: series probability from inputs ------------------

@dataclass(frozen=True)
class SeriesObservation:
    """Result of running the full model for one series at one moment."""
    fav_net_rating: float
    und_net_rating: float
    posterior_diff_mean: float
    posterior_diff_std: float
    p_fav_home: float
    p_fav_road: float
    p_fav_series_raw: float
    p_fav_series_tail_adj: Optional[float]  # None if no market price provided


def evaluate_series(
    fav_net_rating: float,
    und_net_rating: float,
    fav_wins: int,
    und_wins: int,
    completed_games: Optional[List[Tuple[float, bool]]] = None,
    favorite_has_home_court: bool = True,
    market_price: Optional[float] = None,
    sigma_game: float = DEFAULT_SIGMA_GAME,
    sigma_theta: float = DEFAULT_SIGMA_THETA,
    hca: float = DEFAULT_HCA,
    tail_magnitude_pp: float = 0.02,
    expected_margins: Optional[List[Optional[float]]] = None,
    kappa: float = DEFAULT_KAPPA,
    prior_regression: float = DEFAULT_PRIOR_REGRESSION,
) -> SeriesObservation:
    """
    Run the full model for a single series.

    Parameters
    ----------
    completed_games : list of (margin, fav_was_home)
        Games already played. `margin` is positive if the favorite won.
        Used for Bayesian updating of the differential.
    market_price : float or None
        If provided, also computes the tail-adjusted series probability.
    expected_margins : list of float-or-None, optional
        Same length as completed_games. Four Factors-implied margin per
        game, used to weight the per-game variance in the Bayesian update.
        Pass None for any game where Four Factors data isn't available.
        Has no effect when `kappa = 0`.
    kappa : float
        Variance inflation coefficient for Four Factors weighting. 0 (the
        default) preserves the original constant-variance behavior.
    prior_regression : float
        Multiplier on the rating differential before forming the prior.
        c=1.0 uses regular-season NRtg as-is. c<1.0 shrinks the gap toward
        zero, encoding that regular-season NRtg overstates playoff team-
        strength gaps. Empirically calibrated to ~0.6; see the README's
        "Calibration" section for the methodology.
    """
    completed_games = completed_games or []

    # Apply prior regression: shrink the differential toward zero while
    # preserving the midpoint. With c=0.6 and ratings (+7, +1), the midpoint
    # is +4 and the new ratings are (+5.8, +2.2) — same midpoint, gap shrunk
    # from 6 to 3.6 points.
    if prior_regression != 1.0:
        midpoint = (fav_net_rating + und_net_rating) / 2.0
        fav_net_rating = midpoint + prior_regression * (fav_net_rating - midpoint)
        und_net_rating = midpoint + prior_regression * (und_net_rating - midpoint)

    prior = initial_belief(fav_net_rating, und_net_rating, sigma_theta=sigma_theta)
    posterior = bayes_update_many(
        prior, completed_games,
        sigma_game=sigma_game, hca=hca,
        expected_margins=expected_margins, kappa=kappa,
    )

    p_home = per_game_win_prob_from_belief(posterior, True, sigma_game=sigma_game, hca=hca)
    p_road = per_game_win_prob_from_belief(posterior, False, sigma_game=sigma_game, hca=hca)

    p_series_raw = series_win_prob(
        p_home, p_road, fav_wins, und_wins,
        favorite_has_home_court=favorite_has_home_court,
    )

    if market_price is not None:
        p_series_tail_adj = max(
            0.0,
            min(1.0, p_series_raw + tail_correction(market_price, magnitude_pp=tail_magnitude_pp)),
        )
    else:
        p_series_tail_adj = None

    return SeriesObservation(
        fav_net_rating=fav_net_rating,
        und_net_rating=und_net_rating,
        posterior_diff_mean=posterior.mean,
        posterior_diff_std=posterior.std,
        p_fav_home=p_home,
        p_fav_road=p_road,
        p_fav_series_raw=p_series_raw,
        p_fav_series_tail_adj=p_series_tail_adj,
    )
