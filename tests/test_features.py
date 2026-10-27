import numpy as np
import pandas as pd

from spl_predictor.data import league_tables
from spl_predictor.features import FEATURES, add_features


def test_features_cannot_see_the_match_or_later(matches):
    """The leak test: rewrite every result from a date on; earlier inputs must not move."""
    cut = pd.Timestamp("2024-01-01")
    tampered = matches.copy()
    later = tampered["date"] >= cut
    tampered.loc[later, ["home_goals", "away_goals"]] = tampered.loc[later, ["away_goals", "home_goals"]].to_numpy()
    tampered.loc[later, "result"] = tampered.loc[later, "result"].map({"H": "A", "A": "H", "D": "D"})

    before, after = add_features(matches), add_features(tampered)
    first_day = before["date"] == before.loc[before["date"] >= cut, "date"].min()
    keep = (before["date"] < cut) | first_day   # the first matches on/after the cut use only earlier results
    pd.testing.assert_frame_equal(before.loc[keep, FEATURES], after.loc[keep, FEATURES])
    assert not before.loc[later, "result"].equals(after.loc[later, "result"])


def test_promoted_club_gets_bottom_three_average(matches, data):
    last = league_tables(matches).query("season == '2024-25'").sort_values("position")
    neom = data[(data["season"] == "2025-26") & (data["home"] == "Neom")]
    assert np.isclose(neom["home_last"].iloc[0], last["gd_pg"].tail(3).mean())


def test_first_season_dropped_and_no_gaps(data):
    assert "2014-15" not in set(data["season"])
    assert not data[FEATURES].isna().any().any()
