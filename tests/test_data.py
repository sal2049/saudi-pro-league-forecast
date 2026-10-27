import pandas as pd

from spl_predictor.data import league_tables


def test_table_from_three_matches():
    matches = pd.DataFrame({
        "season": "2000-01",
        "home": ["A", "B", "C"], "away": ["B", "C", "A"],
        "home_goals": [2, 1, 0], "away_goals": [0, 1, 3],
        "result": ["H", "D", "A"],
    })
    table = league_tables(matches).set_index("team")
    assert table.loc["A", ["points", "goal_diff"]].tolist() == [6, 5]
    # B and C are level on points; B's goal difference (-2) beats C's (-3).
    assert table["position"].to_dict() == {"A": 1, "B": 2, "C": 3}


def test_known_champions(matches):
    champions = league_tables(matches).query("position == 1").set_index("season")
    assert champions.loc["2024-25", ["team", "points"]].tolist() == ["Al-Ittihad", 83]
    assert champions.loc["2025-26", ["team", "points"]].tolist() == ["Al-Nassr", 86]


def test_finished_seasons_are_complete(matches):
    finished = matches[matches["season"] < "2026-27"]
    assert finished["result"].notna().all()
    played = league_tables(finished).groupby("season").agg(games=("played", "min"), clubs=("team", "size"))
    # A double round-robin: every club plays every other club home and away.
    assert (played["games"] == 2 * (played["clubs"] - 1)).all()
