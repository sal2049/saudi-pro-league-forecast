import pandas as pd

from spl_predictor.data import league_tables, next_season


def test_table_from_three_matches():
    matches = pd.DataFrame(
        {
            "season": "2000-01",
            "home": ["A", "B", "C"],
            "away": ["B", "C", "A"],
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 3],
        }
    )
    table = league_tables(matches).set_index("team")

    assert table.loc["A", ["wins", "draws", "losses", "points", "goal_diff"]].tolist() == [2, 0, 0, 6, 5]
    assert table.loc["B", "points"] == 1 and table.loc["C", "points"] == 1
    # B and C are level on points; B's goal difference (-2) beats C's (-3).
    assert table["position"].to_dict() == {"A": 1, "B": 2, "C": 3}


def test_real_tables_match_known_results(matches):
    tables = league_tables(matches)

    champions = tables[tables["position"] == 1].set_index("season")
    assert champions.loc["2024-25", "team"] == "Al-Ittihad"
    assert champions.loc["2024-25", "points"] == 83
    assert champions.loc["2023-24", "team"] == "Al-Hilal"

    # Complete double round-robins: 30 games with 16 clubs, 34 with 18.
    games = tables.groupby("season")["played"].agg(["min", "max", "size"])
    assert (games["min"] == games["max"]).all()
    assert (games["min"] == 2 * (games["size"] - 1)).all()


def test_next_season():
    assert next_season("2024-25") == "2025-26"
    assert next_season("2098-99") == "2099-00"
