# Saudi Pro League table forecast

[![CI](https://github.com/sal2049/saudi-pro-league-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/sal2049/saudi-pro-league-forecast/actions/workflows/ci.yml)

Can you predict next season's league table from past results alone? This
project tries, using seven seasons of Saudi Pro League matches (2018-19 to
2024-25), and then checks its 2025-26 forecast against the table that season
actually produced.

The short answer: a little. The best model is a plain linear regression, about
**2.7 places off per club**, against 3.1 for simply copying last season's table
and 5 to 6 for a random guess. With only 100 club-seasons to learn from, the
more flexible models memorised the training data and did worse than the
baseline.

![Forecast for 2025-26 against the real final table](reports/figures/forecast_vs_actual.png)

The forecast above was built from matches up to May 2025. It put the eventual
champions Al-Nassr third (a 14% title chance, behind Al-Ittihad at 37%), got
three of the top four right, and 17 of 18 clubs finished inside their 80%
range. Its worst miss was Neom, a newly promoted club it had no history for:
forecast last, finished 8th.

## How it works

```mermaid
flowchart LR
    M[("Match results<br/>7 seasons, 1,812 matches")] --> T["League tables<br/>one per season"]
    T --> F["Feature table<br/>one row per club per season"]
    C["Clubs in 2025-26"] --> F
    F --> B["Walk-forward backtest<br/>5 models, 4 test seasons"]
    B --> S{"Lowest<br/>error"}
    S --> P["Refit on every season<br/>and forecast 2025-26"]
    P --> H["Score the forecast"]
    R[("2025-26 final table")] --> H
```

1. **Tables.** Matches are rolled up into a final table per season
   ([data.py](src/spl_predictor/data.py)).
2. **Features.** Each club-season gets three numbers that were known before its
   first match: points per game and goal difference per game last season, and
   a flag for promoted clubs ([features.py](src/spl_predictor/features.py)).
3. **Models.** Five candidates predict next season's points per game; clubs
   are ranked by that prediction ([models.py](src/spl_predictor/models.py)).
4. **Backtest.** Each model is trained on the seasons before a test season and
   scored on that season, for four test seasons in turn
   ([evaluate.py](src/spl_predictor/evaluate.py)).
5. **Forecast.** The backtest winner is refitted on everything and forecasts
   2025-26, with a range for each club from 10,000 simulated seasons
   ([forecast.py](src/spl_predictor/forecast.py)).
6. **Holdout.** Only then is the real 2025-26 table opened and the forecast scored.

[docs/architecture.md](docs/architecture.md) has the design decisions and the
guards against data leakage. [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb)
runs the same steps with the intermediate tables on show.

## Results

Error is the average number of places between a club's predicted and actual
position. Lower is better.

| Model | Backtest<br/>(4 seasons) | On its own<br/>training seasons | 2025-26<br/>holdout | Gain over baseline,<br/>all 5 seasons (95% CI) |
|---|:---:|:---:|:---:|:---:|
| Last season's table (baseline) | 3.10 | 3.24 | 2.89 | |
| **Linear regression** | **2.68** | 2.77 | 2.89 | **0.34** (0.10 to 0.59) |
| v1: KNN classifier on positions | 2.96 | 2.69 | 3.22 | 0.05 (-0.33 to 0.42) |
| Random forest | 3.25 | 1.62 | 3.11 | -0.16 (-0.74 to 0.41) |
| Gradient boosting | 3.44 | 1.45 | 2.89 | -0.27 (-0.88 to 0.34) |

What I take from this:

- **Only the straight line beats the baseline, and only by about a third of a
  place.** It is the one model whose confidence interval stays clear of zero.
  It won three of the five test seasons, lost one narrowly (3.44 against
  3.39 in 2024-25) and tied on 2025-26.
- **What it learned.** Fitted on all 100 rows: points per game next season =
  0.45 + 0.67 &times; last season's points per game + 0.07 &times; last
  season's goal difference per game + 0.11 if promoted. Points and goal
  difference are 97% correlated, so how the weight is split between them
  changes from one training window to the next; read the two together. The
  overall slope is below 1, so clubs drift back toward the average.
- **The gain comes from goal difference.** On points and the promoted flag
  alone, the model ranks clubs exactly like the baseline (3.10). Goal
  difference is a steadier measure of strength than points, so it reorders
  clubs that finished close together.
- **Promoted clubs are the main source of error.** The average miss is 2.2
  places for established clubs and 4.8 for promoted ones. The model knows
  nothing about a promoted side except that it is new.

![Backtest and holdout error for each model](reports/figures/model_comparison.png)

### Overfitting

The clearest sign of overfitting is a model doing much better on seasons it
trained on than on a season it has never seen.

![Error on training seasons against unseen seasons](reports/figures/overfitting_gap.png)

Gradient boosting is off by 1.5 places on its training seasons and 3.4 on new
ones, which is worse than not modelling at all. The random forest shows the
same pattern. Linear regression has no gap, which is what you want from a
model fitted on 100 rows.

## What changed from the first version

This started as a data mining course project, a single notebook. It reported
a KNN classifier with an error of 2.47 places. When I went back to it, the
number did not hold up, for reasons that had little to do with KNN:

| First version | Problem | Now |
|---|---|---|
| Eight models compared on one season, and the winner's score on that same season reported | The reported score is the best of eight tries on 15 data points, so it flatters the model | Walk-forward backtest over four seasons, plus 2025-26 kept aside as a final check |
| No baseline | No way to tell if the model had learned anything | "Same as last season" baseline, with a confidence interval on the gain |
| Position treated as 18 separate classes | Throws away the order (2nd is closer to 1st than 15th is) and leaves about five examples per class | Regression on points per game, then rank |
| Scored on the 15 clubs that stayed up; the forecast table included the three relegated clubs and none of the promoted ones | Not a forecast of the real league | Scored on all 18 clubs in the season being predicted |
| Season totals as features | Seasons have 30 or 34 games, so totals are not comparable; the seven features held only four independent facts | Per-game rates |

Scored the new way, the first version's model comes out at 2.96 in the backtest
and 3.22 on 2025-26. That is no better than the baseline. It is kept in the
comparison as `v1` so the two can be read side by side.

## Limitations

- **Small data.** Six seasons with a previous season to learn from, 100 rows.
  Every number above carries wide uncertainty, and the confidence intervals
  treat clubs in the same season as independent when they are not.
- **Results only.** No squad values, transfers, managers or ownership. The
  league changed sharply in 2023 when four clubs came under PIF ownership and
  the league grew to 18 teams; nothing in the features can see that coming.
- **Promoted clubs** get a generic prior. Al-Qadsiah (4th in 2024-25) and Neom
  (8th in 2025-26) were both forecast near the bottom.
- **The tree models run at scikit-learn defaults.** That is what lets them
  memorise. Limited to shallow trees they stop overfitting and land close to
  the linear model (2.69 for the forest, 2.93 for boosting in the backtest),
  so the lesson is about model capacity on 100 rows, not about trees
  ([tree_capacity_check.csv](reports/tree_capacity_check.csv)). I ran this
  after choosing the model, so it is a check and not part of the comparison.
- **Tie-breaks** use goal difference, not the league's head-to-head rule.

## Run it

```bash
git clone https://github.com/sal2049/saudi-pro-league-forecast.git
cd saudi-pro-league-forecast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m spl_predictor   # backtest, forecast, holdout score; rewrites reports/
pytest                    # 12 tests, including a data-leakage test
```

Needs Python 3.10 or newer. A full run takes a few seconds and reproduces every
number and figure in this README.

## Repository layout

```
data/
  matches/            one CSV of match results per season
  holdout/            2025-26 club list and final table
src/spl_predictor/
  data.py             load matches, build league tables
  features.py         one row per club-season, pre-season information only
  models.py           the five candidates
  evaluate.py         walk-forward backtest, metrics, bootstrap
  forecast.py         forecast, simulated ranges, holdout scoring
  plots.py            figures
  pipeline.py         runs all of the above
tests/                unit tests
notebooks/            step-by-step walkthrough
reports/              generated tables and figures
docs/architecture.md  design notes
```

## Data

Match results were collected from [FBref](https://fbref.com/) with Firecrawl.
The 2025-26 final table is from
[Wikipedia](https://en.wikipedia.org/wiki/2025%E2%80%9326_Saudi_Pro_League),
checked against [RSSSF](https://rsssf.org/tabless/sau2026.html). Details are
in [data/README.md](data/README.md).
