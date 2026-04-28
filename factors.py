"""
Four Factors — Dean Oliver's basketball efficiency framework.

All functions in this module are pure: same inputs -> same outputs, no I/O,
no side effects. This makes the math independently testable.

Background
----------
Dean Oliver (2002) identified four statistical categories that together
explain most of the variance in team winning:

    1. Effective Field Goal % (eFG%) — accounts for 3-pointers being
       worth more than 2-pointers.
    2. Turnover Rate (TOV%) — share of possessions ending in turnovers.
    3. Offensive Rebound % (ORB%) — share of available offensive rebounds.
    4. Free Throw Rate (FT-rate) — free throws drawn per FGA.

Differences in these four categories between the two teams in a game,
weighted by Oliver's standard coefficients and scaled by pace, give an
"expected margin" — the score margin that would be expected from how the
teams actually played, stripped of variance from random hot/cold shooting.

We use this in two ways:

  - As a margin sanity check (`expected_margin` below).
  - As an input to per-game variance weighting in the Bayesian update:
    games where actual margin diverges sharply from expected margin are
    treated as noisier evidence, so they pull the posterior less.

References
----------
- Oliver, "Basketball on Paper" (2004)
- https://www.basketball-reference.com/about/factors.html
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# Oliver's standard relative weights for the four factors. These are the
# canonical values from the basketball-reference glossary entry. They sum
# to a total, but the relative magnitudes are what matter when computing
# expected margin from differentials.
W_EFG = 1.6      # eFG% — strongest factor
W_TOV = 1.4      # TOV% — second strongest (note: lower TOV is better)
W_ORB = 0.5
W_FTR = 0.5      # FT-rate


@dataclass(frozen=True)
class FourFactors:
    """One team's Four Factors for a single game, plus pace estimate."""
    efg: float        # effective FG%, 0..1
    tov_rate: float   # turnover rate, 0..1 (turnovers per possession)
    orb_rate: float   # offensive rebound %, 0..1
    ft_rate: float    # free throw rate (FTA/FGA), 0..~1
    pace: float       # estimated possessions

    def __post_init__(self):
        # Sanity bounds; bad data should raise rather than silently propagate
        for name, val in [("efg", self.efg), ("tov_rate", self.tov_rate),
                          ("orb_rate", self.orb_rate)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {val}")
        if self.ft_rate < 0:
            raise ValueError(f"ft_rate must be >= 0, got {self.ft_rate}")
        if self.pace <= 0:
            raise ValueError(f"pace must be positive, got {self.pace}")


def estimate_possessions(box: Dict[str, Optional[int]]) -> Optional[float]:
    """
    Estimate possessions from a team's box score.

    Standard formula: POSS ~= FGA + 0.44 * FTA - OREB + TOV

    The 0.44 coefficient on FTA accounts for the fact that not all FTAs
    are end-of-possession events (and-1s, technicals). Returns None if
    any required field is missing.
    """
    try:
        fga = box["fga"]
        fta = box["fta"]
        oreb = box["oreb"]
        tov = box["tov"]
    except KeyError:
        return None
    if any(v is None for v in (fga, fta, oreb, tov)):
        return None
    return float(fga) + 0.44 * float(fta) - float(oreb) + float(tov)


def four_factors_from_box(
    team_box: Dict[str, Optional[int]],
    opp_box: Dict[str, Optional[int]],
) -> Optional[FourFactors]:
    """
    Compute the four factors for one team given its own box score and the
    opponent's (needed for ORB% which depends on opponent DREB).

    Returns None if any required field is missing — propagates None
    upward rather than silently using 0 or NaN.
    """
    try:
        fgm = team_box["fgm"]
        fga = team_box["fga"]
        fg3m = team_box["fg3m"]
        fta = team_box["fta"]
        oreb = team_box["oreb"]
        tov = team_box["tov"]
        opp_dreb = opp_box["dreb"]
    except KeyError:
        return None

    required = (fgm, fga, fg3m, fta, oreb, tov, opp_dreb)
    if any(v is None for v in required):
        return None
    if fga == 0:
        return None  # game without shots? defensive — bail

    poss = estimate_possessions(team_box)
    if poss is None or poss <= 0:
        return None

    # eFG% = (FGM + 0.5 * FG3M) / FGA
    efg = (float(fgm) + 0.5 * float(fg3m)) / float(fga)

    # TOV% = TOV / POSS
    tov_rate = float(tov) / poss

    # ORB% = OREB / (OREB + opp_DREB)
    orb_denom = float(oreb) + float(opp_dreb)
    orb_rate = float(oreb) / orb_denom if orb_denom > 0 else 0.0

    # FT-rate = FTA / FGA  (note: not FTM/FGA — Oliver uses FTA)
    ft_rate = float(fta) / float(fga)

    try:
        return FourFactors(
            efg=efg, tov_rate=tov_rate, orb_rate=orb_rate,
            ft_rate=ft_rate, pace=poss,
        )
    except ValueError:
        # If any value falls outside sanity bounds (e.g. eFG > 1.0 from
        # bad data), bail rather than poisoning the model.
        return None


def expected_margin(
    fav_factors: FourFactors,
    und_factors: FourFactors,
) -> float:
    """
    Oliver's Four Factors -> expected per-game margin.

    Uses the standard weighted differential:

        delta_margin_per_100 = 100 * (1.6 * d_eFG - 1.4 * d_TOV
                                     + 0.5 * d_ORB + 0.5 * d_FTR)

    Then scales by the average pace of the two teams to get per-game
    points. Sign convention: positive when the favorite played better.
    """
    d_efg = fav_factors.efg - und_factors.efg
    d_tov = fav_factors.tov_rate - und_factors.tov_rate
    d_orb = fav_factors.orb_rate - und_factors.orb_rate
    d_ftr = fav_factors.ft_rate - und_factors.ft_rate

    # Per-100-possessions point differential. Note the SIGN on TOV:
    # higher TOV rate is BAD, so a positive d_tov (fav turned ball over
    # more) should subtract from expected margin.
    per_100 = 100.0 * (
        W_EFG * d_efg
        - W_TOV * d_tov
        + W_ORB * d_orb
        + W_FTR * d_ftr
    )

    avg_pace = (fav_factors.pace + und_factors.pace) / 2.0
    return per_100 * avg_pace / 100.0
