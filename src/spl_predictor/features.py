"""Build the modelling table: one row per club per season.

Every feature describes what was known *before* the season kicked off, and
the targets describe how that season ended. Keeping the two strictly apart is
the whole job of this module (see ``tests/test_features.py`` for the leak test).
"""

from __future__ import annotations

import pandas as pd

from .data import TABLE_COLUMNS, league_tables, next_season
from .elo import DEFAULT_ELO, EloConfig, preseason_ratings

# Last season's table line, carried forward as "prev_<column>".
PREV_COLUMNS = [f"prev_{c}" for c in [*TABLE_COLUMNS, "ppg", "gd_pg"]]

TARGETS = ["ppg", "position"]


def build_dataset(
    matches: pd.DataFrame,
    upcoming_teams: list[str] | None = None,
    elo_config: EloConfig = DEFAULT_ELO,
) -> pd.DataFrame:
    """Join each club-season to its pre-season features.

    Columns: ``season, team, n_teams, promoted, elo, prev_*`` and the targets
    ``ppg`` and ``position``.

    The first season in the data has no "previous season" and is dropped. If
    ``upcoming_teams`` is given, rows for the season after the last one in
    ``matches`` are appended with empty targets, ready to be forecast.
    """
    tables = league_tables(matches)
    seasons = sorted(tables["season"].unique())

    entries = tables[["season", "team", "ppg", "position"]].copy()
    upcoming = {}
    if upcoming_teams:
        upcoming_season = next_season(seasons[-1])
        upcoming = {upcoming_season: list(upcoming_teams)}
        entries = pd.concat(
            [entries, pd.DataFrame({"season": upcoming_season, "team": upcoming_teams})],
            ignore_index=True,
        )
    entries["n_teams"] = entries.groupby("season")["team"].transform("size")

    # Shift every table forward one season so it lines up with the season it
    # is allowed to inform.
    previous = tables.drop(columns="position").copy()
    previous["season"] = previous["season"].map(next_season)
    previous = previous.rename(columns={c: f"prev_{c}" for c in previous.columns[2:]})

    dataset = entries.merge(previous, on=["season", "team"], how="left")
    dataset = dataset[dataset["season"] != seasons[0]]

    # A club with no line in last season's table was promoted. Stand in for
    # its missing history with the average of last season's bottom three,
    # i.e. "assume a promoted side is about as good as the sides it replaced".
    # The `promoted` flag lets a model learn how far off that assumption is.
    dataset["promoted"] = dataset["prev_points"].isna().astype(int)
    bottom_three = (
        tables.sort_values("position")
        .groupby("season")
        .tail(3)
        .groupby("season")[[c.removeprefix("prev_") for c in PREV_COLUMNS]]
        .mean()
        .add_prefix("prev_")
    )
    bottom_three.index = bottom_three.index.map(next_season)
    fill = dataset[["season"]].join(bottom_three, on="season")[PREV_COLUMNS]
    dataset[PREV_COLUMNS] = dataset[PREV_COLUMNS].fillna(fill)

    elo = preseason_ratings(matches, upcoming=upcoming, config=elo_config)
    dataset = dataset.merge(elo, on=["season", "team"], how="left", validate="one_to_one")

    ordered = ["season", "team", "n_teams", "promoted", "elo", *PREV_COLUMNS, *TARGETS]
    return dataset[ordered].sort_values(["season", "team"]).reset_index(drop=True)
