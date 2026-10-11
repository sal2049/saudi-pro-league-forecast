"""Elo ratings computed from individual match results.

The league tables give us only ~16 rows per season. Elo squeezes more out of
the same data by updating a club's strength after each of its 30-34 matches,
and it keeps a memory across seasons, which a single season's table cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    """Conventional football Elo settings.

    These are fixed up front and were not tuned against any result in this
    repository; tuning four knobs on six seasons would only fit noise.
    """

    k: float = 20.0               # how far one result moves a rating
    home_advantage: float = 60.0  # rating points added to the home side
    mean: float = 1500.0          # rating of an average top-flight club
    newcomer: float = 1400.0      # first-ever rating of a promoted club
    carryover: float = 0.75       # share of (rating - mean) kept over a summer


DEFAULT_ELO = EloConfig()


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability-like expected result (1 win, 0.5 draw, 0 loss) for side A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def preseason_ratings(
    matches: pd.DataFrame,
    upcoming: dict[str, list[str]] | None = None,
    config: EloConfig = DEFAULT_ELO,
) -> pd.DataFrame:
    """Rating each club holds on the opening day of each season.

    Only matches from earlier seasons feed a pre-season rating, so it is safe
    to use as a feature for predicting that season.

    ``upcoming`` maps a future season to its list of clubs, for seasons that
    have no match data yet. Returns columns ``season, team, elo``.
    """
    ratings: dict[str, float] = {}
    rows: list[dict] = []

    def open_season(season: str, teams: list[str]) -> None:
        # In the very first season nobody has a history, so everyone starts
        # level. After that, a club we have never seen is a promoted side.
        entry_rating = config.newcomer if ratings else config.mean
        # Pull every known club part of the way back to the mean: squads and
        # managers change over the summer, so last year's rating overstates
        # what we know.
        for team, rating in ratings.items():
            ratings[team] = config.mean + config.carryover * (rating - config.mean)
        for team in teams:
            ratings.setdefault(team, entry_rating)
            rows.append({"season": season, "team": team, "elo": ratings[team]})

    for season, games in matches.groupby("season", sort=True):
        open_season(season, sorted(pd.unique(games[["home", "away"]].values.ravel())))

        for match in games.itertuples(index=False):
            expected_home = expected_score(
                ratings[match.home] + config.home_advantage, ratings[match.away]
            )
            if match.home_goals > match.away_goals:
                actual_home = 1.0
            elif match.home_goals == match.away_goals:
                actual_home = 0.5
            else:
                actual_home = 0.0
            shift = config.k * (actual_home - expected_home)
            ratings[match.home] += shift
            ratings[match.away] -= shift

    for season, teams in sorted((upcoming or {}).items()):
        open_season(season, sorted(teams))

    return pd.DataFrame(rows)
