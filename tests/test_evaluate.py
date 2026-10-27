import pytest

from spl_predictor.evaluate import walk_forward
from spl_predictor.models import BASELINE, candidates
from spl_predictor.pipeline import next_week
from spl_predictor.predict import most_likely_winners, predict_week


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
    assert (preds.groupby("model").size() == (data["season"] == "2023-24").sum()).all()


def test_a_week_prediction(data):
    week = predict_week(candidates()["Logistic regression"], data, "2025-26", 20)
    assert len(week) == 9
    assert week[["p_H", "p_D", "p_A"]].sum(axis=1).round(6).eq(1).all()
    assert len(most_likely_winners(week)) == 18


def test_next_week_has_no_results_yet(data):
    season, week = next_week(data)
    fixtures = data[(data["season"] == season) & (data["week"] == week)]
    assert len(fixtures) == 9 and fixtures["result"].isna().all()
