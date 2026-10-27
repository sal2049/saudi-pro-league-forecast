# Data

## `matches/`

One CSV per season, 2014-15 to 2026-27, scraped with Firecrawl from FBref's
"Scores and Fixtures" pages, for example
<https://fbref.com/en/comps/70/2025-2026/schedule/2025-2026-Saudi-Pro-League-Scores-and-Fixtures>.
The current season's file was scraped on 27 October 2026 and includes
fixtures that have not been played yet.

| Column | Example | Meaning |
|---|---|---|
| `season` | `2024-25` | Season |
| `week` | `12` | Matchweek (round) |
| `date` | `2024-08-22` | Match date |
| `home` | `Al-Taawoun` | Home side |
| `score` | `1–0` | Full-time score, home goals first. Blank if not played yet |
| `away` | `Al-Fayha` | Away side |
| `notes` | | FBref's note, if any |

| Seasons | Clubs | Matches per season |
|---|:---:|:---:|
| 2014-15 to 2017-18 | 14 | 182 |
| 2018-19 to 2022-23 | 16 | 240 |
| 2023-24 onwards | 18 | 306 |

2,940 played matches in total up to the scrape date.

**Checks.** For the seven seasons 2018-19 to 2024-25, every score and date
matches the files this project used before (1,812 matches). Every finished
season is a complete double round-robin, and the computed champions match
the real ones (tested in `tests/test_data.py`).

**Known quirks.**
- Al-Orobah v Al-Nassr on 28 February 2025 is recorded as 0&ndash;3, "match
  awarded to Al-Nassr". It is kept as the official result.
- In 2026-27, five matches from weeks 8 to 10 were postponed to later dates.
  The "next matchweek" is the earliest week with no results at all.

To refresh the current season, scrape its fixtures page again and replace
`2026-27.csv`, keeping the same columns.
