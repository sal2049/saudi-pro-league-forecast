"""Walk-forward backtest and the metrics used to compare models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss

from .data import RESULTS
from .features import FEATURES
from .models import SEED


def walk_forward(data: pd.DataFrame, models: dict, test_seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test every model the way it would really have been used.

    For each test season, each model is trained on all played matches from
    earlier seasons and predicts every match of the test season. Returns the
    test predictions (one row per model and match) and, per model and season,
    the log loss on the training matches, which is how overfitting is measured.
    """
    played = data.dropna(subset=["result"])
    predictions, train_scores = [], []
    for season in test_seasons:
        train = played[played["season"] < season]
        test = played[played["season"] == season]
        for name, model in models.items():
            fitted = clone(model).fit(train[FEATURES], train["result"])
            probs = pd.DataFrame(fitted.predict_proba(test[FEATURES]), columns=fitted.classes_,
                                 index=test.index)[RESULTS]
            predictions.append(test[["season", "week", "date", "home", "away", "result"]]
                               .assign(model=name, pick=probs.idxmax(axis=1))
                               .join(probs.add_prefix("p_")))
            train_scores.append({"model": name, "season": season,
                                 "train_log_loss": log_loss(train["result"], fitted.predict_proba(train[FEATURES]),
                                                            labels=fitted.classes_)})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(train_scores)


def score(predictions: pd.DataFrame) -> pd.Series:
    """Accuracy and log loss for one set of predictions."""
    probs = predictions[[f"p_{r}" for r in RESULTS]].to_numpy()
    return pd.Series({
        "matches": len(predictions),
        "accuracy": accuracy_score(predictions["result"], predictions["pick"]),
        "log_loss": log_loss(predictions["result"], probs, labels=RESULTS),
        "draws_predicted": (predictions["pick"] == "D").mean(),
    })


def summarise(predictions: pd.DataFrame, train_scores: pd.DataFrame) -> pd.DataFrame:
    """One row per model: test accuracy and log loss, and the train-test gap."""
    order = list(dict.fromkeys(predictions["model"]))
    summary = predictions.groupby("model").apply(score, include_groups=False).loc[order]
    summary["train_log_loss"] = train_scores.groupby("model")["train_log_loss"].mean()
    summary["gap"] = summary["log_loss"] - summary["train_log_loss"]
    per_season = predictions.groupby(["model", "season"]).apply(
        lambda g: accuracy_score(g["result"], g["pick"]), include_groups=False).unstack()
    return summary.join(per_season.add_prefix("acc_"))


def weekly_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    """How many results the picks got right, matchweek by matchweek."""
    correct = predictions["pick"] == predictions["result"]
    return (predictions.assign(correct=correct)
            .groupby(["model", "season", "week"])["correct"].agg(["sum", "size"])
            .rename(columns={"sum": "right", "size": "matches"}).reset_index())


def paired_bootstrap(a: pd.Series, b: pd.Series, n_resamples: int = 10_000) -> tuple[float, float, float]:
    """Mean of (a - b) over matches, with a 95% interval from resampling matches.

    Used for "how many more results does the model get right than the
    baseline": a and b are 1 for a correct pick, 0 otherwise.
    """
    diff = (a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    rng = np.random.default_rng(SEED)
    # In blocks of 1,000 resamples so memory stays small with thousands of matches.
    draws = np.concatenate([rng.choice(diff, size=(1_000, diff.size)).mean(axis=1)
                            for _ in range(n_resamples // 1_000)])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
