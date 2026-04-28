"""
Replay the prediction model through historical playoff series.

Given cached historical data (from backtest/historical_data.py) and a
config (KAPPA, sigma_game, etc.), produce a list of "prediction events" —
one per state in each series where the model can make a prediction.

A "state" is a moment in the series where the model has observed some
prefix of the games and predicts the eventual outcome. For an N-game
series we produce N prediction events:

    state 0: pre-G1, no games observed
    state 1: post-G1, 1 game observed
    ...
    state N-1: post-G(N-1), all but the last game observed

We don't produce a state for the post-final position because the series
is over and there's nothing to predict.

This module is the bridge between historical data and the metrics module.
It does not load data (the caller does that and passes it in) and does
not compute metrics (the metrics module consumes its output).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import factors
import model as M


@dataclass(frozen=True)
class PredictionEvent:
    """One state in one series at which the model made a prediction."""
    season: str
    higher_seed: str
    lower_seed: str
    state_index: int          # 0 = pre-series, 1 = after G1, etc.
    higher_seed_wins_so_far: int
    lower_seed_wins_so_far: int
    n_games_observed: int
    n_games_with_factors: int   # how many of the observed games had Four Factors data

    # Predicted probability of the higher seed eventually winning the series
    p_higher_wins: float

    # Posterior on the differential at this state
    posterior_diff_mean: float
    posterior_diff_std: float

    # Ground truth: did the higher seed win the series?
    higher_seed_won_series: bool


def _build_observed_games(
    games_so_far: List[Dict[str, Any]],
) -> Tuple[List[Tuple[float, bool]], List[Optional[float]], int]:
    """
    Convert raw game records (from historical_data.py) into the format
    the model's bayes_update_many expects.

    Returns:
      - observed_games: list of (margin, was_home) tuples from the higher
        seed's perspective. (`favorite` in our model = `higher seed` here.)
      - expected_margins: list of Four Factors-implied margins, with None
        for any game where box-score data is missing.
      - n_games_with_factors: count of games where we had usable box data.
    """
    observed_games: List[Tuple[float, bool]] = []
    expected_margins: List[Optional[float]] = []
    n_with_factors = 0

    for g in games_so_far:
        margin = float(g["margin"])
        was_home = bool(g.get("higher_seed_was_home", False))
        observed_games.append((margin, was_home))

        em = None
        higher_box = g.get("higher_seed_box")
        lower_box = g.get("lower_seed_box")
        if higher_box and lower_box:
            higher_ff = factors.four_factors_from_box(higher_box, lower_box)
            lower_ff = factors.four_factors_from_box(lower_box, higher_box)
            if higher_ff and lower_ff:
                em = factors.expected_margin(higher_ff, lower_ff)
                n_with_factors += 1
        expected_margins.append(em)

    return observed_games, expected_margins, n_with_factors


def replay_one_series(
    season: str,
    series: Dict[str, Any],
    ratings: Dict[str, float],
    kappa: float = 0.0,
    sigma_game: float = M.DEFAULT_SIGMA_GAME,
    sigma_theta: float = M.DEFAULT_SIGMA_THETA,
    hca: float = M.DEFAULT_HCA,
    prior_regression: float = 1.0,
    rating_lookup: Optional[Any] = None,
) -> List[PredictionEvent]:
    """
    Replay one series and return one PredictionEvent per state.

    `rating_lookup` is an optional callable (team_abbrev -> rating) that
    overrides `ratings`. Used by ELO-based experiments where the prior
    comes from a different rating source.

    `prior_regression` (default 1.0 = no change) multiplies the pre-series
    differential. With c < 1, the prior is shrunk toward zero — i.e., we
    treat regular-season NRtg as overstating true team-strength gaps. Used
    in diagnostics to test whether playoff-quality strength is meaningfully
    different from regular-season NRtg.
    """
    higher = series["higher_seed"]
    lower = series["lower_seed"]
    games = series["games"]
    truth = bool(series["higher_seed_won"])

    # Look up ratings; if a team is missing, we can't run this series.
    def _lookup(team: str) -> Optional[float]:
        if rating_lookup is not None:
            return rating_lookup(team)
        return ratings.get(team)

    higher_rating = _lookup(higher)
    lower_rating = _lookup(lower)
    if higher_rating is None or lower_rating is None:
        return []

    # Apply prior regression to the differential. Done by shifting the
    # ratings symmetrically around their midpoint, which preserves the
    # mean rating but shrinks the gap.
    if prior_regression != 1.0:
        midpoint = (higher_rating + lower_rating) / 2.0
        higher_rating = midpoint + prior_regression * (higher_rating - midpoint)
        lower_rating = midpoint + prior_regression * (lower_rating - midpoint)

    events: List[PredictionEvent] = []

    for state_index in range(len(games)):
        games_so_far = games[:state_index]
        observed, expected_margins, n_with_factors = _build_observed_games(games_so_far)

        higher_wins_so_far = sum(1 for g in games_so_far if g["margin"] > 0)
        lower_wins_so_far = state_index - higher_wins_so_far

        if higher_wins_so_far >= 4 or lower_wins_so_far >= 4:
            continue

        obs = M.evaluate_series(
            fav_net_rating=higher_rating,
            und_net_rating=lower_rating,
            fav_wins=higher_wins_so_far,
            und_wins=lower_wins_so_far,
            completed_games=observed,
            favorite_has_home_court=True,
            sigma_game=sigma_game,
            sigma_theta=sigma_theta,
            hca=hca,
            expected_margins=expected_margins,
            kappa=kappa,
        )

        events.append(PredictionEvent(
            season=season,
            higher_seed=higher,
            lower_seed=lower,
            state_index=state_index,
            higher_seed_wins_so_far=higher_wins_so_far,
            lower_seed_wins_so_far=lower_wins_so_far,
            n_games_observed=state_index,
            n_games_with_factors=n_with_factors,
            p_higher_wins=obs.p_fav_series_raw,
            posterior_diff_mean=obs.posterior_diff_mean,
            posterior_diff_std=obs.posterior_diff_std,
            higher_seed_won_series=truth,
        ))

    return events


def replay_all_seasons(
    seasons_data: List[Tuple[str, Dict[str, Any], Dict[str, float]]],
    kappa: float = 0.0,
    sigma_game: float = M.DEFAULT_SIGMA_GAME,
    sigma_theta: float = M.DEFAULT_SIGMA_THETA,
    hca: float = M.DEFAULT_HCA,
    prior_regression: float = 1.0,
    rating_lookup_factory: Optional[Any] = None,
) -> List[PredictionEvent]:
    """Replay all series across all loaded seasons."""
    all_events: List[PredictionEvent] = []
    for season, series_payload, ratings in seasons_data:
        rating_lookup = None
        if rating_lookup_factory is not None:
            rating_lookup = rating_lookup_factory(season)

        for series in series_payload.get("series", []):
            events = replay_one_series(
                season=season,
                series=series,
                ratings=ratings,
                kappa=kappa,
                sigma_game=sigma_game,
                sigma_theta=sigma_theta,
                hca=hca,
                prior_regression=prior_regression,
                rating_lookup=rating_lookup,
            )
            all_events.extend(events)
    return all_events


def event_to_dict(event: PredictionEvent) -> Dict[str, Any]:
    """For JSON serialization."""
    return asdict(event)
