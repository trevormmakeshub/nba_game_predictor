Run python scripts/refresh_today.py during the season. It does not run on a timer.

Completed scores are team rows from stats.nba.com leaguegamefinder. 2025-26 Regular Season: 1230 games 2025-26 Playoffs: 85 games 2026-27 Regular Season: 132 games ESPN schedules are used for venues and for games that have not been played. ESPN marks some postponed games as post with a 0-0 score. Those games are not stored. ESPN finals matched 1447, mismatched 0, missing from stats.nba.com 1. The ESPN final dated 2025-12-16, SAS at NYK, 113-124, is not in the league log, so it is not stored. Player totals are the 2026-27 per-game base table from stats.nba.com. Form uses only earlier games: win rate, points scored, points allowed, and rest days. A game does not train on its own score. The published home_win_prob on a completed game is in-sample. The held-out accuracy below is a separate fit. The win model is a ridge classifier. Points use linear regression.

There is no recorded public closing line in this table. spread_line and total_line are blank. The line column says this is not a bet. This does not price a bet.

ESPN abbreviations GS, NO, NY, SA, UTAH, and WSH are stored as GSW, NOP, NYK, SAS, UTA, and WAS.

Completed rows: 1447
2026-27 regular-season games: 132
2025-26 playoff games: 85
Zero-score rows stored: 0
Training rows: 1431
Feature count: 8
Minimum date: 2025-10-21
Maximum date: 2026-11-06
Next games through 2026-11-14: 56
Next games with home_win_prob: 56
Held-out games from 2026-10-20: 132 of 1299 training rows, accuracy 0.6515
Current-season team rows: 30
Player rows: 457
