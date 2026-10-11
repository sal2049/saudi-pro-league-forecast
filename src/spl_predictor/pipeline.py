"""Run the whole project end to end and write everything under ``reports/``."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from . import plots
from .data import REPO_ROOT, load_final_table, load_matches, load_upcoming_teams, next_season
from .elo import EloConfig
from .evaluate import paired_bootstrap, summarise, walk_forward
from .features import build_dataset
from .forecast import forecast_season, score_holdout, select_model
from .models import BASELINE, CORE_FEATURES, SEED, Candidate, candidates

REPORTS_DIR = REPO_ROOT / "reports"


def elo_sensitivity(matches: pd.DataFrame, upcoming_teams: list[str]) -> pd.DataFrame:
    """Backtest error of the Elo model under 72 different Elo settings.

    This is a robustness check, not a tuning step: the settings in
    ``EloConfig`` stay as they are whatever this table says.
    """
    elo_model = [m for m in candidates() if m.name == "Elo"]
    rows = []
    for k, home, carry, newcomer in product(
        [10, 20, 30, 40], [0, 60, 100], [0.5, 0.75, 1.0], [1400, 1450]
    ):
        config = EloConfig(k=k, home_advantage=home, carryover=carry, newcomer=newcomer)
        dataset = build_dataset(matches, upcoming_teams, config)
        mae = summarise(walk_forward(dataset, elo_model)).loc["Elo", "test_mae"]
        rows.append(
            {"k": k, "home_advantage": home, "carryover": carry, "newcomer": newcomer,
             "test_mae": mae}
        )
    return pd.DataFrame(rows)


def tree_capacity_check(dataset: pd.DataFrame) -> pd.DataFrame:
    """Do the tree models still overfit once they are held back?

    The comparison runs the ensembles at scikit-learn defaults, where they
    are free to grow deep trees. This reruns the backtest with shallow trees
    to see whether the train-test gap comes from the algorithm or just from
    giving it too much room on 100 rows. Like the Elo check, it was added
    after the model was chosen and plays no part in choosing it.
    """
    shallow = [
        Candidate(
            "Random forest (depth 2)",
            CORE_FEATURES,
            RandomForestRegressor(
                n_estimators=300, max_depth=2, min_samples_leaf=5, random_state=SEED
            ),
        ),
        Candidate(
            "Gradient boosting (stumps)",
            CORE_FEATURES,
            GradientBoostingRegressor(
                n_estimators=50, max_depth=1, learning_rate=0.05, random_state=SEED
            ),
        ),
    ]
    return summarise(walk_forward(dataset, shallow))[["test_mae", "train_mae", "gap"]]


def run(reports_dir: Path = REPORTS_DIR) -> dict[str, pd.DataFrame]:
    matches = load_matches()
    target_season = next_season(matches["season"].max())
    upcoming_teams = load_upcoming_teams(target_season)
    dataset = build_dataset(matches, upcoming_teams)

    # 1. Backtest every candidate on seasons it has not seen.
    backtest = walk_forward(dataset, candidates())
    summary = summarise(backtest)

    # 2. Pick the model on the backtest alone, then forecast the next season.
    chosen = select_model(summary)
    model = next(m for m in candidates() if m.name == chosen)
    forecast = forecast_season(dataset, model, backtest)

    # 3. Only now open the real final table and score the forecast.
    final_table = load_final_table(target_season)
    holdout = score_holdout(dataset, candidates(), final_table)
    holdout_mae = holdout.groupby("model", sort=False)["abs_error"].mean()
    summary["holdout_mae"] = holdout_mae

    # 4. Pool every out-of-sample season to ask which models beat the baseline.
    scored = pd.concat(
        [backtest[backtest["split"] == "test"], holdout.assign(season=target_season)],
        ignore_index=True,
    )
    summary["all_seasons_mae"] = scored.groupby("model")["abs_error"].mean()
    summary[["vs_baseline", "ci_low", "ci_high"]] = [
        paired_bootstrap(scored, name, BASELINE) if name != BASELINE else (0.0, 0.0, 0.0)
        for name in summary.index
    ]
    forecast["position"] = forecast["team"].map(final_table.set_index("team")["position"])

    sensitivity = elo_sensitivity(matches, upcoming_teams)
    shallow_trees = tree_capacity_check(dataset)

    reports_dir.mkdir(exist_ok=True)
    summary.round(3).to_csv(reports_dir / "model_comparison.csv")
    scored.drop(columns=["split", "test_season"]).round(3).to_csv(
        reports_dir / "out_of_sample_predictions.csv", index=False
    )
    forecast.round(3).to_csv(reports_dir / f"forecast_{target_season}.csv", index=False)
    sensitivity.round(3).to_csv(reports_dir / "elo_sensitivity.csv", index=False)
    shallow_trees.round(3).to_csv(reports_dir / "tree_capacity_check.csv")

    figures = reports_dir / "figures"
    plots.save(plots.model_comparison(summary, holdout_mae), figures / "model_comparison.png")
    plots.save(plots.overfitting_gap(summary), figures / "overfitting_gap.png")
    plots.save(
        plots.forecast_vs_actual(forecast, final_table), figures / "forecast_vs_actual.png"
    )

    _print_report(summary, chosen, scored, forecast, sensitivity, target_season)
    print("\nTree models with shallow trees (check only)\n")
    print(shallow_trees.round(2).to_string())
    return {"summary": summary, "backtest": backtest, "scored": scored,
            "forecast": forecast, "sensitivity": sensitivity,
            "shallow_trees": shallow_trees}


def _print_report(summary, chosen, scored, forecast, sensitivity, target_season) -> None:
    columns = ["test_mae", "train_mae", "gap", "holdout_mae", "all_seasons_mae",
               "vs_baseline", "ci_low", "ci_high"]
    print("Position error by model (places)\n")
    print(summary[columns].round(2).to_string())
    print(f"\nChosen on the backtest: {chosen}")
    by_status = scored[scored["model"] == chosen].groupby("promoted")["abs_error"].mean()
    print(f"{chosen} error, established clubs {by_status[0]:.2f}, promoted clubs {by_status[1]:.2f}")
    print(
        f"Elo settings tried: {len(sensitivity)}, backtest error "
        f"{sensitivity['test_mae'].min():.2f} to {sensitivity['test_mae'].max():.2f}"
    )
    print(f"\n{target_season} forecast against the final table\n")
    table = forecast[["pred_position", "team", "pos_low", "pos_high", "p_title",
                      "p_relegation", "position"]]
    print(table.round(2).to_string(index=False))
    in_range = forecast["position"].between(forecast["pos_low"], forecast["pos_high"]).mean()
    print(f"\nClubs that finished inside their 80% range: {in_range:.0%}")
