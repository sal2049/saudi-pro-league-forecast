import pandas as pd
import pytest

from spl_predictor.evaluate import random_rank_mae, to_positions, walk_forward
from spl_predictor.forecast import forecast_season
from spl_predictor.models import candidates


@pytest.fixture(scope="module")
def backtest(dataset):
    return walk_forward(dataset, candidates())


def test_tied_scores_share_the_average_position():
    scores = pd.Series([2.0, 1.0, 1.0, 0.5])
    season = pd.Series(["s"] * 4)
    assert to_positions(scores, season).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_models_only_train_on_earlier_seasons(backtest):
    train = backtest[backtest["split"] == "train"]
    test = backtest[backtest["split"] == "test"]
    assert (train["season"] < train["test_season"]).all()
    assert (test["season"] == test["test_season"]).all()
    assert sorted(test["test_season"].unique()) == ["2021-22", "2022-23", "2023-24", "2024-25"]


def test_every_club_in_a_test_season_is_scored(backtest):
    test = backtest[backtest["split"] == "test"]
    sizes = test.groupby(["model", "test_season"]).agg(n=("team", "size"), teams=("n_teams", "first"))
    assert (sizes["n"] == sizes["teams"]).all()


def test_random_rank_mae():
    # Two clubs: a coin flip is either spot on (0) or swapped (1 each).
    assert random_rank_mae(2) == pytest.approx(0.5)


def test_forecast_is_a_valid_table(dataset, backtest):
    elo = next(m for m in candidates() if m.name == "Elo")
    forecast = forecast_season(dataset, elo, backtest, n_simulations=2_000)

    assert forecast["pred_position"].tolist() == list(range(1, 19))
    assert forecast["p_title"].sum() == pytest.approx(1.0)
    assert forecast["p_relegation"].sum() == pytest.approx(3.0)
    assert (forecast["pos_low"] <= forecast["pos_high"]).all()
