import pytest

from spl_predictor.evaluate import walk_forward
from spl_predictor.forecast import most_likely_winners, predict_week, season_table
from spl_predictor.models import BASELINE, candidates


@pytest.fixture(scope="module")
def backtest(data):
    return walk_forward(data, candidates(), ["2023-24"])


def test_probabilities_sum_to_one(backtest):
    preds, _ = backtest
    assert preds[["p_A", "p_D", "p_H"]].sum(axis=1).round(6).eq(1).all()


def test_baseline_always_picks_home(backtest):
    preds, _ = backtest
    assert (preds.loc[preds["model"] == BASELINE, "pick"] == "H").all()


def test_every_match_of_the_test_season_is_scored(backtest, data):
    preds, _ = backtest
    n = (data["season"] == "2023-24").sum()
    assert (preds.groupby("model").size() == n).all()


def test_week_prediction_uses_only_the_past(data):
    lr = candidates()["Logistic regression"]
    week = predict_week(lr, data, "2025-26", 20)
    assert len(week) == 9
    assert week[["p_H", "p_D", "p_A"]].sum(axis=1).round(6).eq(1).all()
    assert len(most_likely_winners(week)) == 18


def test_season_table_is_a_full_table(matches):
    table = season_table(candidates()["Logistic regression"], matches, "2025-26")
    assert sorted(table["predicted_position"]) == list(range(1, 19))
    # 306 matches hand out between 2 points (a draw) and 3 points (a win) each.
    assert 2 * 306 <= table["expected_points"].sum() <= 3 * 306
