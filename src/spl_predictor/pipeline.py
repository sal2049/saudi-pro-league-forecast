"""Run the whole project end to end and write everything under ``reports/``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import plots
from .data import REPO_ROOT, load_matches
from .evaluate import paired_bootstrap, score, summarise, walk_forward, weekly_accuracy
from .features import add_gap
from .models import BASELINE, candidates
from .predict import most_likely_winners, predict_week

REPORTS_DIR = REPO_ROOT / "reports"
BACKTEST = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
            "2021-22", "2022-23", "2023-24", "2024-25"]
VALIDATION = "2025-26"   # scored often while building this; not a clean test
TEST = "2026-27"         # the current season: never looked at while building


def next_week(data: pd.DataFrame) -> tuple[str, int] | None:
    """The earliest matchweek of the current season with no results yet."""
    season = data[data["season"] == TEST]
    unplayed = season.groupby("week")["result"].apply(lambda r: r.isna().all())
    return (TEST, int(unplayed[unplayed].index.min())) if unplayed.any() else None


def window_check(matches: pd.DataFrame) -> pd.DataFrame:
    """Backtest log loss of logistic regression for different form windows.

    This is how the window of 50 matches was picked: on the backtest only.
    """
    lr = {"lr": candidates()["Logistic regression"]}
    rows = {n: score(walk_forward(add_gap(matches, n), lr, BACKTEST)[0]) for n in (10, 20, 34, 50)}
    return pd.DataFrame(rows).T.rename_axis("window")


def run(reports_dir: Path = REPORTS_DIR) -> dict:
    matches = load_matches()
    data = add_gap(matches)
    windows = window_check(matches)
    models = candidates()

    # 1. Backtest: train on earlier seasons, test on the next, nine times.
    backtest, train_scores = walk_forward(data, models, BACKTEST)
    summary = summarise(backtest, train_scores)
    chosen = summary["log_loss"].idxmin()

    # 2. Then the later seasons, with the model already chosen.
    later, later_train = walk_forward(data, models, [VALIDATION, TEST])
    by_season = {s: summarise(later[later["season"] == s], later_train[later_train["season"] == s])
                 for s in (VALIDATION, TEST)}

    # 3. Right or wrong, match by match: is the gain over the baseline real?
    scored = pd.concat([backtest, later])
    right = (scored.assign(right=scored["pick"] == scored["result"])
             .pivot_table(index=["date", "home"], columns="model", values="right"))
    gain = paired_bootstrap(right[chosen], right[BASELINE])

    weeks = weekly_accuracy(later[later["season"] == VALIDATION])
    upcoming = next_week(data)
    week = predict_week(models[chosen], data, *upcoming) if upcoming else None

    reports_dir.mkdir(exist_ok=True)
    summary.to_csv(reports_dir / "backtest.csv", float_format="%.3f")
    windows.to_csv(reports_dir / "window_check.csv", float_format="%.3f")
    for season, table in by_season.items():
        table.to_csv(reports_dir / f"season_{season}.csv", float_format="%.3f")
    scored.to_csv(reports_dir / "match_predictions.csv", index=False, float_format="%.3f")
    weeks.to_csv(reports_dir / f"weekly_{VALIDATION}.csv", index=False)
    if week is not None:
        week.to_csv(reports_dir / "next_week.csv", index=False, float_format="%.3f")

    figures = reports_dir / "figures"
    plots.save(plots.model_comparison(summary, by_season[VALIDATION], by_season[TEST]),
               figures / "model_comparison.png")
    plots.save(plots.overfitting_gap(summary), figures / "overfitting_gap.png")
    plots.save(plots.probability_curve(models[chosen], data[data["season"] < VALIDATION]),
               figures / "probability_curve.png")

    _print_report(summary, by_season, chosen, gain, weeks, upcoming, week)
    print("\nForm window (logistic regression, backtest)\n")
    print(windows[["accuracy", "log_loss"]].round(3).to_string())
    return {"backtest": summary, "seasons": by_season, "week": week}


def _print_report(summary, by_season, chosen, gain, weeks, upcoming, week) -> None:
    cols = ["accuracy", "log_loss", "train_log_loss", "gap", "draws_predicted"]
    print(f"Backtest: {int(summary['matches'].iloc[0])} matches, {BACKTEST[0]} to {BACKTEST[-1]}\n")
    print(summary[cols].round(3).to_string())
    print(f"\nChosen on the backtest (lowest log loss): {chosen}")
    for season, table in by_season.items():
        label = "validation, seen while building" if season == VALIDATION else "test, never seen while building"
        print(f"\n{season} ({label}), {int(table['matches'].iloc[0])} matches\n")
        print(table[["accuracy", "log_loss"]].round(3).to_string())
    print(f"\n{chosen} gets {gain[0]:+.1%} more results right than '{BASELINE}' "
          f"(95% interval {gain[1]:+.1%} to {gain[2]:+.1%}), every test match")

    w = weeks[weeks["model"] == chosen]
    print(f"\n{VALIDATION} week by week: {w['right'].mean():.1f} of {w['matches'].mean():.0f} right on average, "
          f"best {w['right'].max()}, worst {w['right'].min()}")

    if week is not None:
        season, n = upcoming
        print(f"\nNext: {season} matchweek {n}\n")
        print(week[["date", "home", "away", "gap", "p_H", "p_D", "p_A", "pick"]]
              .to_string(index=False, float_format="%.2f"))
        top = most_likely_winners(week).head(3)
        print("\nMost likely to win this week: " + ", ".join(
            f"{r.club} ({r.win_chance:.0%} v {r.opponent})" for r in top.itertuples()))
