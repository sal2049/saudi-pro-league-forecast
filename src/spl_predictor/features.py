"""The single feature: how much better the home team has been lately.

``gap`` = home team's goal difference per game over its last 50 league
matches, minus the away team's. Only matches played before the one being
predicted count; ``tests/test_features.py`` checks that.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

FEATURES = ["gap"]
WINDOW = 50         # league matches, about a season and a half
MIN_MATCHES = 3     # with fewer, a club counts as new to the league
NEW_CLUB = -0.5     # goals per game assumed for a club new to the league


def add_gap(matches: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add ``home_form``, ``away_form`` and ``gap`` to every match."""
    history: dict[str, list[float]] = defaultdict(list)
    home_form, away_form = [], []
    # Matches are in date order: read each club's form, then add this result.
    for match in matches.itertuples(index=False):
        for club, column in ((match.home, home_form), (match.away, away_form)):
            recent = history[club][-window:]
            column.append(np.mean(recent) if len(recent) >= MIN_MATCHES else NEW_CLUB)
        if pd.notna(match.result):
            history[match.home].append(match.home_goals - match.away_goals)
            history[match.away].append(match.away_goals - match.home_goals)

    out = matches.assign(home_form=home_form, away_form=away_form)
    out["gap"] = out["home_form"] - out["away_form"]
    # The first season only builds up history; nobody has any before it.
    return out[out["season"] > out["season"].min()].reset_index(drop=True)
