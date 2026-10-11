import pandas as pd
import pytest

from spl_predictor.elo import EloConfig, expected_score, preseason_ratings


def test_expected_score_is_symmetric():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)


def test_everyone_starts_level_in_the_first_season(matches):
    ratings = preseason_ratings(matches)
    assert (ratings.loc[ratings["season"] == "2018-19", "elo"] == 1500).all()


def test_winner_takes_exactly_what_the_loser_gives_up():
    one_match = pd.DataFrame(
        {"season": ["2000-01"], "home": ["A"], "away": ["B"], "home_goals": [1], "away_goals": [0]}
    )
    # No summer regression, so the only thing that moves ratings is the match.
    config = EloConfig(carryover=1.0)
    after = preseason_ratings(one_match, {"2001-02": ["A", "B"]}, config)
    after = after[after["season"] == "2001-02"].set_index("team")["elo"]

    assert after["A"] > 1500 > after["B"]
    assert after["A"] + after["B"] == pytest.approx(3000)


def test_a_brand_new_club_enters_below_average(matches):
    ratings = preseason_ratings(matches, {"2025-26": ["Neom", "Al-Hilal"]})
    upcoming = ratings[ratings["season"] == "2025-26"].set_index("team")["elo"]
    assert upcoming["Neom"] == EloConfig().newcomer
    assert upcoming["Al-Hilal"] > 1500
