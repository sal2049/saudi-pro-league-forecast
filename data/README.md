# Data

## `matches/`

One CSV per season, 2018-19 to 2024-25, one row per league match (1,812 in
total). Collected from [FBref](https://fbref.com/) with Firecrawl.

| Column | Example | Meaning |
|---|---|---|
| `Date` | `Thu Aug 22 2024` | Match date |
| `Team 1` | `Al-Taawoun` | Home side |
| `FT` | `1-0` | Full-time score, home goals first |
| `Team 2` | `Al-Fayha` | Away side |

Seasons up to 2022-23 have 16 clubs and 240 matches. From 2023-24 the league
has 18 clubs and 306 matches.

## `holdout/`

Files for 2025-26, the season the project forecasts. They are kept apart from
the match data because they play a different role.

- `teams_2025-26.csv` lists the 18 clubs in the league. This is known before
  the season starts and is the only 2025-26 input the forecast uses.
- `final_table_2025-26.csv` is the table as it finished in May 2026, taken from
  [Wikipedia](https://en.wikipedia.org/wiki/2025%E2%80%9326_Saudi_Pro_League)
  and checked row by row against [RSSSF](https://rsssf.org/tabless/sau2026.html).
  It is read once, after the forecast is written, to score it.
