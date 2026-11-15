Run python scripts/refresh_today.py during the season. It does not run on a timer.

Completed scores are team rows from stats.nba.com leaguegamefinder. 2025-26 Regular Season failed (ConnectionResetError) 2025-26 Playoffs failed (TimeoutError) 2026-27 Regular Season failed (TimeoutError) stats.nba.com failed. The saved team game log was loaded. ESPN added 56 finals dated after the league log. ESPN schedules are used for venues and for games that have not been played. ESPN marks some postponed games as post with a 0-0 score. Those games are not stored. ESPN finals matched 1503, mismatched 0, missing from stats.nba.com 1. The ESPN final dated 2025-12-16, SAS at NYK, 113-124, is not in the league log, so it is not stored. Player leader rows are the ESPN 2026-27 regular-season table. Form uses only earlier games: win rate, points scored, points allowed, rest days, and the last four games when that check improves the holdout. A game does not train on its own score. The published home_win_prob on a completed game is in-sample. The held-out accuracy below is a separate fit. The win model is a ridge classifier. Points use linear regression, plus a half-weight of prior points per game only when that blend lowers the holdout total error.

There is no recorded public closing line in this table. spread_line and total_line are blank. The line column says this is not a bet. This does not price a bet.

ESPN abbreviations GS, NO, NY, SA, UTAH, and WSH are stored as GSW, NOP, NYK, SAS, UTA, and WAS.

Completed rows: 1503
2026-27 regular-season games: 188
2025-26 playoff games: 85
Zero-score rows stored: 0
Training rows: 1487
Feature count: 12
Minimum date: 2025-10-21
Maximum date: 2026-11-14
Next games through 2026-11-21: 53
Next games with home_win_prob: 53
Held-out games from 2026-10-20: 188 of 1299 training rows, accuracy 0.6755
In-sample accuracy: 0.6907 on 1487 scored rows
High-confidence next games: 21
Baseline holdout: accuracy 0.6702, total MAE 21.650, prior-points blend MAE 17.535, 188 test games after 1299 earlier games
Recent-form holdout: accuracy 0.6755, total MAE 21.536, prior-points blend MAE 17.462, 188 test games after 1299 earlier games
Recent form adopted: yes
Points blend adopted: yes
Current-season team rows: 30. The team file is wins and points from the completed 2026 regular-season games. Box-score counting stats were left out because stats.nba.com did not return a new game log.
Player rows: 250
