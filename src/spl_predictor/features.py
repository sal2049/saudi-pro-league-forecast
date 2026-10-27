"""Turn each match into two numbers that were known before kick-off.

1. ``last_diff``: home team's goal difference per game last season, minus the
   away team's. How strong each club was over a whole season.
2. ``form_diff``: the same, over each team's last 10 league matches. How
   strong each club has been lately, including this season.

Nothing about a match, or anything played after it, can reach its own
features; ``tests/test_features.py`` checks that.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np
import pandas as pd

from .data import league_tables

FEATURES = ["last_diff", "form_diff"]
FORM_WINDOW = 10   # league matches
FORM_MINIMUM = 3   # fewer than this and form falls back to last season


def last_season_strength(matches: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Goal difference per game last season, for every (season, club).

    A promoted club has no top-flight season to look back on. It gets the
    average of last season's bottom three: "about as good as the clubs it
    replaced".
    """
    tables = league_tables(matches)
    seasons = sorted(matches["season"].unique())
    strength = {}
    for previous, season in pairwise(seasons):
        last = tables[tables["season"] == previous].set_index("team")
        promoted_guess = last.sort_values("position")["gd_pg"].tail(3).mean()
        clubs = pd.unique(matches.loc[matches["season"] == season, ["home", "away"]].values.ravel())
        for club in clubs:
            strength[(season, club)] = last["gd_pg"].get(club, promoted_guess)
    return strength


def add_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add both features to every match after the first season.

    The first season has no "last season", so its matches are dropped. They
    still count towards form.
    """
    strength = last_season_strength(matches)
    out = matches.copy()
    out["home_last"] = [strength.get(k, np.nan) for k in zip(out["season"], out["home"])]
    out["away_last"] = [strength.get(k, np.nan) for k in zip(out["season"], out["away"])]

    # Walk through the matches in date order, reading each club's recent goal
    # differences *before* adding this match to them.
    history: dict[str, list[float]] = defaultdict(list)
    home_form, away_form = [], []
    for match in out.itertuples(index=False):
        for club, column in ((match.home, home_form), (match.away, away_form)):
            recent = history[club][-FORM_WINDOW:]
            column.append(np.mean(recent) if len(recent) >= FORM_MINIMUM else np.nan)
        if match.result is not None:
            history[match.home].append(match.home_goals - match.away_goals)
            history[match.away].append(match.away_goals - match.home_goals)
    out["home_form"] = home_form
    out["away_form"] = away_form
    # A club with too few recent top-flight matches (usually a promoted side)
    # is treated as being in last-season form.
    out["home_form"] = out["home_form"].fillna(out["home_last"])
    out["away_form"] = out["away_form"].fillna(out["away_last"])

    out["last_diff"] = out["home_last"] - out["away_last"]
    out["form_diff"] = out["home_form"] - out["away_form"]
    return out.dropna(subset=["last_diff"]).reset_index(drop=True)
