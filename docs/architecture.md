# Architecture

## Data flow

```mermaid
flowchart TB
    subgraph inputs [Inputs]
        M[("data/matches<br/>one row per match")]
        C["data/holdout/teams<br/>clubs in the target season"]
        R[("data/holdout/final_table<br/>how the target season ended")]
    end

    subgraph prep [Preparation]
        T["data.league_tables<br/>points, goals, position per club"]
        F["features.build_dataset<br/>features and targets per club-season"]
    end

    subgraph eval [Evaluation]
        W["evaluate.walk_forward<br/>train on the past, test on the next season"]
        S["evaluate.summarise<br/>error, train-test gap"]
    end

    subgraph out [Forecast]
        P["forecast.forecast_season<br/>table, ranges, probabilities"]
        H["forecast.score_holdout<br/>error on the target season"]
    end

    M --> T --> F
    C --> F
    F --> W --> S --> P
    F --> P --> H
    R --> H
    S --> O["reports/"]
    P --> O
    H --> O
```

`pipeline.run()` calls these in order. Each box is one function that takes
data frames and returns a data frame, so every step can be run and inspected
on its own, which is what the notebook does.

## The modelling table

`build_dataset` returns one row per club per season.

| Column | Known before the season? | Meaning |
|---|:---:|---|
| `prev_ppg`, `prev_gd_pg` | yes | Points and goal difference per game last season |
| `promoted` | yes | 1 if the club was not in last season's table |
| `prev_points`, `prev_wins`, ... | yes | Raw totals, used only by the `v1` model |
| `ppg` | no (target) | Points per game this season |
| `position` | no (target) | Final position this season |

Seven seasons of matches give six seasons with a "previous season", so there
are 100 labelled rows. The 2025-26 rows have features and empty targets.

## Walk-forward backtest

A model is only ever tested on a season that comes after everything it was
trained on, the same way it would be used for real.

| Test season | Training seasons | Training rows | Test rows |
|---|---|:---:|:---:|
| 2021-22 | 2019-20, 2020-21 | 32 | 16 |
| 2022-23 | 2019-20 to 2021-22 | 48 | 16 |
| 2023-24 | 2019-20 to 2022-23 | 64 | 18 |
| 2024-25 | 2019-20 to 2023-24 | 82 | 18 |
| 2025-26 (holdout) | 2019-20 to 2024-25 | 100 | 18 |

The first four rows are the backtest and are used to choose a model. The last
row is the holdout: it plays no part in any choice and is scored once.

A random split of rows would be wrong here. It would let a model train on
2024-25 and be tested on 2021-22, and it would put clubs from the same season
on both sides of the split.

## Design decisions

**Predict points per game, then rank.** League position is the thing we care
about, but it is an awkward target: it is ordinal, it depends on what every
other club does, and its scale changes when the league grows from 16 to 18
clubs. Points per game is continuous and means the same thing in every
season. Ranking the predictions gives the table.

**Three features, all from last season's table.** Points per game, goal
difference per game and a promoted flag. They are easy to explain and known
before a ball is kicked. Points and goal difference are highly correlated
(0.97), but goal difference is less affected by luck in close matches, and it
is where the model's gain over the baseline comes from: on points alone the
linear model ranks clubs exactly as the baseline does.

**Promoted clubs.** A promoted club has no previous top-flight season. Its
`prev_*` features are filled with the average of the previous season's bottom
three, and the `promoted` flag lets a model correct for that.

**Small models first.** With 100 rows, the comparison runs from no parameters
(baseline) to four (linear regression) before anything flexible.
The tree ensembles are there to show what happens when a model has more
capacity than the data can support.

**Uncertainty.** The forecast simulates 10,000 seasons. In each one, every
club's predicted points per game is nudged by random noise with the same
spread as the model's backtest errors, and the league is re-ranked. Position
ranges and probabilities are read off those simulations.

## Guards against leakage

- Features for season *N* are built only from seasons before *N*.
  `tests/test_features.py` rewrites every result from a season onwards and
  checks that the season's features do not change.
- `walk_forward` trains only on seasons before the test season.
  `tests/test_evaluate.py` asserts it.
- Feature scaling sits inside each model pipeline, so it is fitted on the
  training seasons only.
- The model is chosen in `pipeline.run()` before `load_final_table` is called.
  The forecast code path never reads the final table.
