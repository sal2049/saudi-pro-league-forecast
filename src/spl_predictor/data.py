"""Load raw match results and turn them into end-of-season league tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHES_DIR = REPO_ROOT / "data" / "matches"
HOLDOUT_DIR = REPO_ROOT / "data" / "holdout"

# The source spells a few clubs with sponsor-style suffixes. Shorten them so
# the names line up with the 2025-26 holdout files and read better in tables.
TEAM_ALIASES = {
    "Al Okhdood SC": "Al-Okhdood",
    "Al Kholood Saudi Club": "Al-Kholood",
    "Al-Riyadh SC": "Al-Riyadh",
}

TABLE_COLUMNS = [
    "played", "wins", "draws", "losses",
    "goals_for", "goals_against", "goal_diff", "points",
]


def load_matches(matches_dir: Path = MATCHES_DIR) -> pd.DataFrame:
    """Read every ``saudi_<season>.csv`` file into one tidy frame.

    Returns one row per match with columns
    ``season, date, home, away, home_goals, away_goals``, sorted by date.
    """
    frames = []
    for path in sorted(Path(matches_dir).glob("saudi_*.csv")):
        raw = pd.read_csv(path)
        goals = raw["FT"].str.split("-", expand=True).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "season": path.stem.removeprefix("saudi_"),
                    "date": pd.to_datetime(raw["Date"], format="%a %b %d %Y"),
                    "home": raw["Team 1"].replace(TEAM_ALIASES),
                    "away": raw["Team 2"].replace(TEAM_ALIASES),
                    "home_goals": goals[0],
                    "away_goals": goals[1],
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No saudi_*.csv files found in {matches_dir}")

    matches = pd.concat(frames, ignore_index=True)
    return matches.sort_values(["season", "date"], kind="stable").reset_index(drop=True)


def league_tables(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matches into one final league table per season.

    Returns one row per (season, team) with the usual table columns, plus
    ``ppg`` and ``gd_pg`` (points and goal difference per game) and
    ``position``. The per-game rates matter because the league grew from 16
    teams (30 games) to 18 teams (34 games) in 2023-24, so season totals are
    not comparable across that change.
    """
    # Stack each match twice, once from each team's point of view.
    home = matches.rename(
        columns={"home": "team", "home_goals": "goals_for", "away_goals": "goals_against"}
    )
    away = matches.rename(
        columns={"away": "team", "away_goals": "goals_for", "home_goals": "goals_against"}
    )
    cols = ["season", "team", "goals_for", "goals_against"]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)

    long["played"] = 1
    long["wins"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["draws"] = (long["goals_for"] == long["goals_against"]).astype(int)
    long["losses"] = (long["goals_for"] < long["goals_against"]).astype(int)

    table = long.groupby(["season", "team"], as_index=False).sum()
    table["goal_diff"] = table["goals_for"] - table["goals_against"]
    table["points"] = 3 * table["wins"] + table["draws"]
    table["ppg"] = table["points"] / table["played"]
    table["gd_pg"] = table["goal_diff"] / table["played"]

    # Ties on points are split by goal difference, then goals scored. The real
    # league uses head-to-head first; it has not changed a position in this data.
    table = table.sort_values(
        ["season", "points", "goal_diff", "goals_for"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    table["position"] = table.groupby("season").cumcount() + 1
    return table.reset_index(drop=True)


def load_upcoming_teams(season: str, holdout_dir: Path = HOLDOUT_DIR) -> list[str]:
    """Clubs taking part in a season we have no match data for yet."""
    return pd.read_csv(Path(holdout_dir) / f"teams_{season}.csv")["team"].tolist()


def load_final_table(season: str, holdout_dir: Path = HOLDOUT_DIR) -> pd.DataFrame:
    """Published final table for a holdout season, used only to score a forecast."""
    table = pd.read_csv(Path(holdout_dir) / f"final_table_{season}.csv")
    table["ppg"] = table["points"] / table["played"]
    return table


def next_season(season: str) -> str:
    """``"2024-25"`` -> ``"2025-26"``."""
    start = int(season[:4]) + 1
    return f"{start}-{(start + 1) % 100:02d}"
