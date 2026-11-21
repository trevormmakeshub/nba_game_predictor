Run python scripts/refresh_today.py during the season. It does not run on a timer.

Completed scores are team rows from stats.nba.com leaguegamefinder. 2025-26 Regular Season: 1230 games 2025-26 Playoffs: 85 games 2026-27 Regular Season: 232 games ESPN schedules are used for venues and for games that have not been played. ESPN marks some postponed games as post with a 0-0 score. Those games are not stored. ESPN finals matched 1541, mismatched 0, missing from stats.nba.com 1. The ESPN final dated 2025-12-16, SAS at NYK, 113-124, is not in the league log, so it is not stored. Player totals are the 2026-27 per-game base table from stats.nba.com. Form uses only earlier games: win rate, points scored, points allowed, rest days, and the last four games when that check improves the holdout. A game does not train on its own score. The published home_win_prob on a completed game is in-sample. The held-out accuracy below is a separate fit. The win model is a ridge classifier. Points use linear regression. A half-weight of prior points per game, or a residual on top of prior points per game, is used only when that choice lowers the holdout total error. Opponent win rate is the other side's prior win rate.

There is no recorded public closing line in this table. spread_line and total_line are blank. The line column says this is not a bet. This does not price a bet.

ESPN abbreviations GS, NO, NY, SA, UTAH, and WSH are stored as GSW, NOP, NYK, SAS, UTA, and WAS.

Completed rows: 1547
2026-27 regular-season games: 232
2025-26 playoff games: 85
Zero-score rows stored: 0
Training rows: 1531
Feature count: 12
Minimum date: 2025-10-21
Maximum date: 2026-11-20
Next games through 2026-11-27: 49
Next games with home_win_prob: 49
Held-out games from 2026-10-20: 232 of 1299 training rows, accuracy 0.6552
In-sample accuracy: 0.6884 on 1531 scored rows
High-confidence next games: 20
Baseline holdout: accuracy 0.6379, total MAE 23.976, prior-points blend MAE 20.809, residual MAE 23.976, 232 test games after 1299 earlier games
Recent-form holdout: accuracy 0.6552, total MAE 23.943, prior-points blend MAE 20.767, residual MAE 23.943, 232 test games after 1299 earlier games
Recent form adopted: yes
Points blend adopted: yes
Points method: blend
Current-season team rows: 30. Counting stats sum the saved team log and 0 later ESPN box scores. Each box score's points match the completed file. oreb, dreb, reb, ast, stl, blk, tov, and pf are per game. Shot makes are season totals.
Player rows: 473
