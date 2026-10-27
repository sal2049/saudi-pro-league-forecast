import pandas as pd

from spl_predictor.features import NEW_CLUB, add_gap


def test_feature_cannot_see_the_match_or_later(matches):
    """The leak test: flip every result from a date on; earlier gaps must not move."""
    cut = pd.Timestamp("2024-01-01")
    tampered = matches.copy()
    later = tampered["date"] >= cut
    tampered.loc[later, ["home_goals", "away_goals"]] = tampered.loc[later, ["away_goals", "home_goals"]].to_numpy()
    tampered.loc[later, "result"] = tampered.loc[later, "result"].map({"H": "A", "A": "H", "D": "D"})

    before, after = add_gap(matches), add_gap(tampered)
    first_day = before["date"] == before.loc[before["date"] >= cut, "date"].min()
    keep = (before["date"] < cut) | first_day   # the first matches on the cut date use only earlier results
    pd.testing.assert_series_equal(before.loc[keep, "gap"], after.loc[keep, "gap"])
    assert not before.loc[later, "result"].equals(after.loc[later, "result"])


def test_a_club_new_to_the_league_starts_at_the_assumed_value(data):
    neom = data[(data["home"] == "Neom") | (data["away"] == "Neom")].iloc[0]
    side = "home_form" if neom["home"] == "Neom" else "away_form"
    assert neom[side] == NEW_CLUB


def test_first_season_dropped_and_no_gaps(data):
    assert "2014-15" not in set(data["season"])
    assert data["gap"].notna().all()
