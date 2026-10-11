import pandas as pd

from spl_predictor.features import PREV_COLUMNS, build_dataset

FEATURES = ["promoted", "elo", *PREV_COLUMNS]


def test_features_cannot_see_the_season_they_predict(matches):
    """The leak test: rewrite a season's results and its features must not move."""
    season = "2023-24"
    tampered = matches.copy()
    later = tampered["season"] >= season
    tampered.loc[later, ["home_goals", "away_goals"]] = tampered.loc[
        later, ["away_goals", "home_goals"]
    ].to_numpy()

    original = build_dataset(matches)
    changed = build_dataset(tampered)

    def pick(df):
        return df[df["season"] == season].set_index("team")

    pd.testing.assert_frame_equal(pick(original)[FEATURES], pick(changed)[FEATURES])
    # Sanity check that the tampering did something: the targets must differ.
    assert not pick(original)["ppg"].equals(pick(changed)["ppg"])


def test_first_season_is_dropped_and_nothing_is_missing(dataset):
    assert "2018-19" not in set(dataset["season"])
    assert not dataset[FEATURES].isna().any().any()


def test_promoted_clubs_are_flagged(dataset):
    promoted = dataset[dataset["promoted"] == 1].groupby("season")["team"].apply(set)
    assert promoted["2024-25"] == {"Al-Qadsiah", "Al-Orobah", "Al-Kholood"}
    assert promoted["2025-26"] == {"Neom", "Al-Najma", "Al-Hazem"}


def test_upcoming_season_has_features_but_no_targets(dataset):
    upcoming = dataset[dataset["season"] == "2025-26"]
    assert len(upcoming) == 18
    assert upcoming[["ppg", "position"]].isna().all().all()
