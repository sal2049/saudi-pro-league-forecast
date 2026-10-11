"""Forecast a season that has not been played, and score it once it has."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import to_positions
from .models import SEED, Candidate

N_SIMULATIONS = 10_000


def select_model(summary: pd.DataFrame) -> str:
    """The model with the lowest walk-forward position error."""
    return summary["test_mae"].idxmin()


def forecast_season(
    dataset: pd.DataFrame,
    model: Candidate,
    backtest: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
) -> pd.DataFrame:
    """Predict the table for the unlabelled season in ``dataset``.

    The model is refitted on every labelled season. A single predicted table
    hides how unsure the model is, so we also simulate the season many times:
    each run adds noise to every club's predicted points per game, sized by
    how wrong this model was in the backtest, and re-ranks the league. The
    spread of positions across runs gives each club a range and the odds of
    finishing in the places that matter.
    """
    labelled = dataset.dropna(subset=["ppg"])
    upcoming = dataset[dataset["ppg"].isna()]
    if upcoming.empty:
        raise ValueError("dataset has no unlabelled season to forecast")

    fitted = model.fit(labelled)
    pred_ppg = fitted.predict(upcoming[model.features])

    held_out = backtest[(backtest["model"] == model.name) & (backtest["split"] == "test")]
    noise_sd = (held_out["pred_ppg"] - held_out["ppg"]).std()

    rng = np.random.default_rng(SEED)
    n_teams = len(upcoming)
    simulated = pred_ppg + rng.normal(0.0, noise_sd, size=(n_simulations, n_teams))
    # argsort twice turns each simulated season into finishing positions.
    positions = (-simulated).argsort(axis=1).argsort(axis=1) + 1

    forecast = pd.DataFrame(
        {
            "season": upcoming["season"].to_numpy(),
            "team": upcoming["team"].to_numpy(),
            "promoted": upcoming["promoted"].to_numpy(),
            "pred_ppg": pred_ppg,
            "pos_low": np.percentile(positions, 10, axis=0),
            "pos_high": np.percentile(positions, 90, axis=0),
            "p_title": (positions == 1).mean(axis=0),
            "p_top4": (positions <= 4).mean(axis=0),
            "p_relegation": (positions > n_teams - 3).mean(axis=0),
        }
    )
    forecast = forecast.sort_values("pred_ppg", ascending=False).reset_index(drop=True)
    forecast.insert(2, "pred_position", forecast.index + 1)
    return forecast


def score_holdout(
    dataset: pd.DataFrame,
    models: list[Candidate],
    final_table: pd.DataFrame,
) -> pd.DataFrame:
    """Score every model on a season none of them was built or chosen on.

    Returns one row per (model, club) with predicted and actual positions.
    """
    labelled = dataset.dropna(subset=["ppg"])
    upcoming = dataset[dataset["ppg"].isna()]
    actual = final_table.set_index("team")["position"]

    rows = []
    for model in models:
        fitted = model.fit(labelled)
        score = pd.Series(fitted.predict(upcoming[model.features]), index=upcoming.index)
        rows.append(
            pd.DataFrame(
                {
                    "model": model.name,
                    "team": upcoming["team"],
                    "promoted": upcoming["promoted"],
                    "pred_position": to_positions(score, upcoming["season"]),
                    "position": upcoming["team"].map(actual),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    if out["position"].isna().any():
        missing = sorted(out.loc[out["position"].isna(), "team"].unique())
        raise ValueError(f"clubs missing from the final table: {missing}")
    out["abs_error"] = (out["pred_position"] - out["position"]).abs()
    return out
