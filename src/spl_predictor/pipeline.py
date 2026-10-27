"""Run the whole project end to end and write everything under ``reports/``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import plots
from .data import REPO_ROOT, league_tables, load_matches
from .evaluate import paired_bootstrap, score, summarise, walk_forward, weekly_accuracy
from .features import add_features
from .forecast import most_likely_winners, predict_week, season_table
from .models import BASELINE, candidates

REPORTS_DIR = REPO_ROOT / "reports"
BACKTEST_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
                    "2021-22", "2022-23", "2023-24", "2024-25"]
HOLDOUT_SEASON = "2025-26"


def next_week(data: pd.DataFrame) -> tuple[str, int] | None:
    """The earliest matchweek of the latest season with no results yet."""
    season = data["season"].max()
    open_weeks = data[(data["season"] == season)].groupby("week")["result"].apply(lambda r: r.isna().all())
    return (season, int(open_weeks[open_weeks].index.min())) if open_weeks.any() else None


def last_season_order(matches: pd.DataFrame, season: str) -> pd.Series:
    """Baseline table: last season's points per game, promoted clubs at the bottom-three average."""
    tables = league_tables(matches)
    seasons = sorted(tables["season"].unique())
    last = tables[tables["season"] == seasons[seasons.index(season) - 1]].set_index("team")
    clubs = tables.loc[tables["season"] == season, "team"]
    guess = last.sort_values("position")["ppg"].tail(3).mean()
    return clubs.map(last["ppg"]).fillna(guess).set_axis(clubs).rank(ascending=False, method="average")


def extra_seasons_check(data: pd.DataFrame) -> pd.DataFrame:
    """Did scraping four more seasons (2014-15 to 2017-18) help?

    Logistic regression on 2020-21 to 2025-26, trained once on every earlier
    season and once only from 2019-20 onwards. Added after the model was
    chosen, as a check; it changes nothing in the comparison.
    """
    lr = candidates()["Logistic regression"]
    test = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", HOLDOUT_SEASON]
    rows = []
    for start in ["2015-16", "2019-20"]:
        subset = data[data["season"] >= start]
        preds, _ = walk_forward(subset, {"lr": lr}, test)
        rows.append(score(preds).rename(f"trained from {start}"))
    return pd.DataFrame(rows)


def run(reports_dir: Path = REPORTS_DIR) -> dict[str, pd.DataFrame]:
    matches = load_matches()
    data = add_features(matches)
    models = candidates()

    # 1. Backtest every model on nine seasons it had not seen.
    backtest, train_scores = walk_forward(data, models, BACKTEST_SEASONS)
    summary = summarise(backtest, train_scores)

    # 2. Choose on the backtest alone: lowest log loss.
    chosen = summary["log_loss"].idxmin()

    # 3. Only then score every model on 2025-26, which played no part in any choice.
    holdout, holdout_train = walk_forward(data, models, [HOLDOUT_SEASON])
    holdout_summary = summarise(holdout, holdout_train)

    # 4. Is the chosen model really better than the baseline? Per match, right or wrong.
    scored = pd.concat([backtest, holdout])
    right = scored.assign(right=scored["pick"] == scored["result"]).pivot_table(
        index=["season", "date", "home"], columns="model", values="right")
    gain = paired_bootstrap(right[chosen], right[BASELINE])

    weeks = weekly_accuracy(holdout)

    # 5. A whole-season table from pre-season match predictions, for every test
    # season, against simply keeping last season's order.
    tables = []
    for season in [*BACKTEST_SEASONS, HOLDOUT_SEASON]:
        t = season_table(models[chosen], matches, season)
        t["baseline_position"] = t["team"].map(last_season_order(matches, season))
        tables.append(t.assign(season=season))
    tables = pd.concat(tables, ignore_index=True)
    table = tables[tables["season"] == HOLDOUT_SEASON]
    extra = extra_seasons_check(data)
    upcoming = next_week(data)
    week = predict_week(models[chosen], data, *upcoming) if upcoming else None

    reports_dir.mkdir(exist_ok=True)
    summary.round(3).to_csv(reports_dir / "model_comparison.csv")
    holdout_summary.round(3).to_csv(reports_dir / f"holdout_{HOLDOUT_SEASON}.csv")
    scored.to_csv(reports_dir / "match_predictions.csv", index=False, float_format="%.3f")
    weeks.to_csv(reports_dir / f"weekly_{HOLDOUT_SEASON}.csv", index=False)
    tables.to_csv(reports_dir / "season_tables.csv", index=False, float_format="%.2f")
    extra.to_csv(reports_dir / "extra_seasons_check.csv", float_format="%.3f")
    if week is not None:
        week.to_csv(reports_dir / "next_week.csv", index=False, float_format="%.3f")

    figures = reports_dir / "figures"
    plots.save(plots.model_comparison(summary, holdout_summary), figures / "model_comparison.png")
    plots.save(plots.overfitting_gap(summary), figures / "overfitting_gap.png")
    plots.save(plots.season_table(table, HOLDOUT_SEASON), figures / f"season_table_{HOLDOUT_SEASON}.png")

    _print_report(summary, holdout_summary, chosen, gain, weeks, tables, upcoming, week)
    print("\nExtra seasons check (logistic regression, 2020-21 to 2025-26)\n")
    print(extra[["matches", "accuracy", "log_loss"]].round(3).to_string())
    return {"summary": summary, "holdout": holdout_summary, "predictions": scored,
            "weeks": weeks, "tables": tables, "week": week}


def _print_report(summary, holdout, chosen, gain, weeks, tables, upcoming, week) -> None:
    cols = ["accuracy", "log_loss", "train_log_loss", "gap", "draws_predicted"]
    print(f"Backtest, {int(summary['matches'].iloc[0])} matches over {len(BACKTEST_SEASONS)} seasons\n")
    print(summary[cols].round(3).to_string())
    print(f"\nChosen on the backtest (lowest log loss): {chosen}")
    print(f"\n{HOLDOUT_SEASON}, {int(holdout['matches'].iloc[0])} matches, untouched until now\n")
    print(holdout[["accuracy", "log_loss"]].round(3).to_string())
    print(f"\n{chosen} picks {gain[0]:+.1%} more results correctly than '{BASELINE}' "
          f"(95% interval {gain[1]:+.1%} to {gain[2]:+.1%}), all test matches")

    w = weeks[weeks["model"] == chosen]
    print(f"\n{HOLDOUT_SEASON} week by week: {w['right'].mean():.1f} of {w['matches'].mean():.0f} right on average, "
          f"best {w['right'].max()}, worst {w['right'].min()}")

    errors = tables.assign(model=(tables["predicted_position"] - tables["position"]).abs(),
                           baseline=(tables["baseline_position"] - tables["position"]).abs())
    by_season = errors.groupby("season")[["model", "baseline"]].mean()
    print("\nSeason table from summed pre-season match predictions, places off per club:")
    print(by_season.round(2).T.to_string())
    print(f"Backtest seasons: model {by_season.loc[BACKTEST_SEASONS, 'model'].mean():.2f}, "
          f"last season's order {by_season.loc[BACKTEST_SEASONS, 'baseline'].mean():.2f}")
    table = tables[tables["season"] == HOLDOUT_SEASON]
    print(table[["predicted_position", "team", "expected_points", "position", "points"]]
          .round(1).to_string(index=False))

    if week is not None:
        season, n = upcoming
        print(f"\nNext: {season} matchweek {n}\n")
        print(week[["date", "home", "away", "p_H", "p_D", "p_A", "pick"]].to_string(index=False, float_format="%.2f"))
        top = most_likely_winners(week).head(3)
        print("\nMost likely to win this week: " + ", ".join(
            f"{r.club} ({r.win_chance:.0%} v {r.opponent})" for r in top.itertuples()))
