"""Load match results and build league tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHES_DIR = REPO_ROOT / "data" / "matches"

RESULTS = ["A", "D", "H"]  # away win, draw, home win (alphabetical, as scikit-learn orders them)


def load_matches(matches_dir: Path = MATCHES_DIR) -> pd.DataFrame:
    """Every fixture, played or not, one row per match, in date order.

    Columns: ``season, week, date, home, away, home_goals, away_goals, result``.
    Goals and result are empty for matches that have not been played yet.
    """
    files = sorted(Path(matches_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No season files found in {matches_dir}")
    raw = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)

    # FBref writes scores as "2–1" with an en dash; unplayed matches are blank.
    goals = raw["score"].str.extract(r"^(\d+)\D(\d+)$").astype(float)
    matches = pd.DataFrame(
        {
            "season": raw["season"],
            "week": raw["week"].astype(int),
            "date": pd.to_datetime(raw["date"]),
            "home": raw["home"],
            "away": raw["away"],
            "home_goals": goals[0],
            "away_goals": goals[1],
        }
    )
    diff = matches["home_goals"] - matches["away_goals"]
    matches["result"] = np.select([diff > 0, diff == 0, diff < 0], ["H", "D", "A"], None)
    return matches.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


def league_tables(matches: pd.DataFrame) -> pd.DataFrame:
    """Final table for every season, from the matches played.

    Returns one row per (season, team) with points, goal difference, the
    per-game rates and the finishing position. Per game, because seasons
    have had 26, 30 and 34 matches per club.
    """
    played = matches.dropna(subset=["result"])
    # Stack each match twice, once from each team's point of view.
    home = pd.DataFrame({"season": played["season"], "team": played["home"],
                         "gf": played["home_goals"], "ga": played["away_goals"]})
    away = pd.DataFrame({"season": played["season"], "team": played["away"],
                         "gf": played["away_goals"], "ga": played["home_goals"]})
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], 0)
    long["played"] = 1

    table = long.groupby(["season", "team"], as_index=False)[["played", "gf", "ga", "points"]].sum()
    table["goal_diff"] = table["gf"] - table["ga"]
    table["ppg"] = table["points"] / table["played"]
    table["gd_pg"] = table["goal_diff"] / table["played"]
    # Ties split by goal difference, then goals scored. The league uses
    # head-to-head first; that has not changed a title or relegation place here.
    table = table.sort_values(["season", "points", "goal_diff", "gf"],
                              ascending=[True, False, False, False], kind="stable")
    table["position"] = table.groupby("season").cumcount() + 1
    return table.reset_index(drop=True)
