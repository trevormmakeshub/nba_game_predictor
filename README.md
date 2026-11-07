# NBA Game Predictor 2026

Completed games run from 2025-10-21 through 2026-11-06. stats.nba.com leaguegamefinder is the score source for the 2025-26 regular season, the 2025-26 playoffs, and the 2026-27 regular season. ESPN team schedules supply the upcoming schedule and venues. Team abbreviations follow stats.nba.com. A sportsbook is not called. Refresh is manual. This does not price a bet.

Run `python scripts/refresh_today.py` during the season. The script keeps results through its as-of date, 2026-11-07.

Output files use the same seeds columns as nfl_game_predictor: date, away, home, home_win_prob, predicted_points, spread_line, total_line, lean, line. spread_line and total_line are blank. The line column says this is not a bet.
