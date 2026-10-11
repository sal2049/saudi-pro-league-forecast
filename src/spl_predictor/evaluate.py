"""Walk-forward backtest and the metrics used to compare models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import SEED, Candidate

# With fewer training seasons than this a model has under 32 rows to learn from.
MIN_TRAIN_SEASONS = 2


def to_positions(scores: pd.Series, groups: pd.Series) -> pd.Series:
    """Rank clubs within each season, strongest score first.

    Tied clubs share the average of the positions they span, so a model gets
    no lucky (or unlucky) credit from an arbitrary tie-break.
    """
    return scores.groupby(groups).rank(ascending=False, method="average")


def walk_forward(
    dataset: pd.DataFrame,
    models: list[Candidate],
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> pd.DataFrame:
    """Backtest every model the way it would really have been used.

    For each test season, a model is trained on all earlier seasons and asked
    to predict that season's table. Nothing from the test season or later is
    ever visible to it. Predictions on the training seasons are kept as well,
    because the gap between the two is how we measure overfitting.

    Returns one row per (model, test season, split, club).
    """
    labelled = dataset.dropna(subset=["ppg"])
    seasons = sorted(labelled["season"].unique())
    results = []

    for test_season in seasons[min_train_seasons:]:
        train = labelled[labelled["season"] < test_season]
        test = labelled[labelled["season"] == test_season]

        for model in models:
            fitted = model.fit(train)
            for split, rows in (("train", train), ("test", test)):
                score = pd.Series(fitted.predict(rows[model.features]), index=rows.index)
                results.append(
                    pd.DataFrame(
                        {
                            "model": model.name,
                            "test_season": test_season,
                            "split": split,
                            "season": rows["season"],
                            "team": rows["team"],
                            "n_teams": rows["n_teams"],
                            "promoted": rows["promoted"],
                            "position": rows["position"],
                            "pred_position": to_positions(score, rows["season"]),
                            "ppg": rows["ppg"],
                            "pred_ppg": score if model.predicts_ppg else np.nan,
                        }
                    )
                )

    out = pd.concat(results, ignore_index=True)
    out["abs_error"] = (out["pred_position"] - out["position"]).abs()
    return out


def summarise(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per model: test accuracy, spread across seasons, and train-test gap."""
    order = list(dict.fromkeys(predictions["model"]))
    test = predictions[predictions["split"] == "test"]
    train = predictions[predictions["split"] == "train"]

    per_season = test.groupby(["model", "test_season"])["abs_error"].mean().unstack()
    rho = test.groupby(["model", "test_season"])[["position", "pred_position"]].apply(
        lambda g: g["position"].corr(g["pred_position"], method="spearman")
    )

    summary = pd.DataFrame(
        {
            "test_mae": test.groupby("model")["abs_error"].mean(),
            "season_sd": per_season.std(axis=1),
            "train_mae": train.groupby("model")["abs_error"].mean(),
            "within_3": test.groupby("model")["abs_error"].apply(lambda e: (e <= 3).mean()),
            "exact": test.groupby("model")["abs_error"].apply(lambda e: (e == 0).mean()),
            "spearman": rho.groupby("model").mean(),
            "ppg_mae": test.assign(e=(test["pred_ppg"] - test["ppg"]).abs())
            .groupby("model")["e"]
            .mean(),
        }
    )
    summary["gap"] = summary["test_mae"] - summary["train_mae"]
    return summary.join(per_season).loc[order]


def random_rank_mae(n_teams: int) -> float:
    """Expected position error of a uniformly random ordering of ``n_teams``."""
    return (n_teams**2 - 1) / (3 * n_teams)


def paired_bootstrap(
    scored: pd.DataFrame,
    model: str,
    baseline: str,
    n_resamples: int = 10_000,
) -> tuple[float, float, float]:
    """Is ``model`` really better than ``baseline``, or is the gap just noise?

    ``scored`` holds out-of-sample errors with columns ``model, season, team,
    abs_error``. The club-seasons are resampled with replacement and the mean
    difference in position error (model minus baseline, negative = model is
    better) is returned with a 95% interval. An interval that straddles zero
    means the data cannot separate the two.
    """
    errors = scored.pivot(index=["season", "team"], columns="model", values="abs_error")
    diff = (errors[model] - errors[baseline]).to_numpy()

    rng = np.random.default_rng(SEED)
    draws = rng.choice(diff, size=(n_resamples, diff.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
