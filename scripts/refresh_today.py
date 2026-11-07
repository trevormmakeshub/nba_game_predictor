"""Pull NBA results and retrain the prior-form table through 2026-11-07.

Completed scores come from stats.nba.com leaguegamefinder. ESPN team
schedules supply the upcoming schedule and venues. A postponed ESPN game
is not stored as a final. A sportsbook is not called. A game's own score
is not a feature. This does not price a bet.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from scipy.special import expit
    from sklearn.linear_model import LinearRegression, RidgeClassifier
except ImportError:
    def expit(value):
        array = np.asarray(value, dtype=float)
        return 1.0 / (1.0 + np.exp(-array))

    class RidgeClassifier:
        def __init__(self, alpha: float = 1.0) -> None:
            self.alpha = alpha
            self.classes_ = np.array([0, 1])
            self.coef_ = None
            self.intercept_ = 0.0

        def fit(self, samples, target):
            samples = np.asarray(samples, dtype=float)
            target = np.asarray(target, dtype=float).reshape(-1)
            self.classes_ = np.unique(target.astype(int))
            center = samples.mean(axis=0)
            gram = (samples - center).T @ (samples - center)
            gram = gram + (self.alpha * np.eye(samples.shape[1]))
            self.coef_ = np.linalg.solve(gram, (samples - center).T @ (target - target.mean()))
            self.intercept_ = float(target.mean() - center @ self.coef_)
            return self

        def decision_function(self, samples):
            samples = np.asarray(samples, dtype=float)
            return samples @ self.coef_ + self.intercept_

    class LinearRegression:
        def fit(self, samples, target):
            samples = np.asarray(samples, dtype=float)
            target = np.asarray(target, dtype=float).reshape(-1)
            ones = np.column_stack([np.ones(len(samples)), samples])
            coef, *_ = np.linalg.lstsq(ones, target, rcond=None)
            self.intercept_ = float(coef[0])
            self.coef_ = coef[1:]
            return self

        def predict(self, samples):
            samples = np.asarray(samples, dtype=float)
            return samples @ self.coef_ + self.intercept_


AS_OF = date(2026, 11, 7)
NEXT_THROUGH = date(2026, 11, 14)
HOLDOUT_START = date(2026, 10, 20)
EASTERN = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = ROOT / "data" / "schedules" / "games.csv"
LOG_PATH = ROOT / "data" / "stats_team" / "team_game_log.csv"
TEAM_STATS_PATH = ROOT / "data" / "stats_team" / "stats_team_reg_2026.csv"
PLAYER_STATS_PATH = ROOT / "data" / "stats_player" / "stats_player_reg_2026.csv"
COMPLETED_PATH = ROOT / "artifacts" / "nba_completed.csv"
NEXT_PATH = ROOT / "artifacts" / "nba_next_games.csv"
SEEDS_PATH = ROOT / "artifacts" / "seeds.csv"
MAX_DATE_PATH = ROOT / "artifacts" / "last_max_date.txt"
README_PATH = ROOT / "README.md"
NOTES_PATH = ROOT / "NOTES.md"
FINDER_URL = "https://stats.nba.com/stats/leaguegamefinder"
PLAYER_NBA_URL = "https://stats.nba.com/stats/leaguedashplayerstats"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=50"
SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule?season={season}"
PLAYER_ESPN_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/statistics/byathlete?limit=250&season=2027&seasontype=2"
NBA_FROM_ESPN = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
SLICES = (
    ("2025-26", "Regular Season"),
    ("2025-26", "Playoffs"),
    ("2026-27", "Regular Season"),
)
FEATURES = [
    "home_win_rate",
    "away_win_rate",
    "home_ppg",
    "away_ppg",
    "home_papg",
    "away_papg",
    "home_rest_days",
    "away_rest_days",
]
COUNT_COLUMNS = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
LINE = "not a bet"
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def fetch_json(url: str, timeout: int = 60, headers: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        headers=headers or {"User-Agent": "nba-game-predictor/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def official_abbrev(code: str) -> str:
    return NBA_FROM_ESPN.get(code, code)


def league_games(season: str, season_type: str) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {"PlayerOrTeam": "T", "Season": season, "SeasonType": season_type, "LeagueID": "00"}
    )
    payload = fetch_json(f"{FINDER_URL}?{query}", headers=NBA_HEADERS)
    block = payload["resultSets"][0]
    frame = pd.DataFrame(block["rowSet"], columns=block["headers"])
    if frame.empty:
        raise ValueError(f"no rows for {season} {season_type}")
    frame["season"] = int(season[:4])
    frame["game_type"] = "POST" if season_type == "Playoffs" else "REG"
    frame["GAME_DATE"] = frame["GAME_DATE"].astype(str).str.slice(0, 10)
    return frame


def load_logs() -> tuple[pd.DataFrame, list[str]]:
    frames = []
    notes = []
    for season, season_type in SLICES:
        try:
            frame = league_games(season, season_type)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            notes.append(f"{season} {season_type} failed ({type(exc).__name__})")
            print(notes[-1])
            continue
        notes.append(f"{season} {season_type}: {frame['GAME_ID'].nunique()} games")
        print(notes[-1])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), notes
    return pd.concat(frames, ignore_index=True), notes


def paired_sides(group: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Return away and home rows. A few games repeat the away matchup on both rows."""
    if len(group) != 2:
        return None
    matchup = group["MATCHUP"].astype(str)
    home = group[matchup.str.contains("vs.", regex=False)]
    away = group[matchup.str.contains(" @ ", regex=False)]
    if len(home) == 1 and len(away) == 1:
        return away.iloc[0], home.iloc[0]
    text = matchup.iloc[0]
    if " @ " in text:
        away_code, home_code = [part.strip() for part in text.split(" @ ", 1)]
    elif "vs." in text:
        home_code, away_code = [part.strip() for part in text.split("vs.", 1)]
    else:
        return None
    away = group[group["TEAM_ABBREVIATION"] == away_code]
    home = group[group["TEAM_ABBREVIATION"] == home_code]
    if len(away) != 1 or len(home) != 1:
        return None
    return away.iloc[0], home.iloc[0]


def games_from_log(log: pd.DataFrame) -> list[dict]:
    rows = []
    skipped = 0
    for game_id, group in log.groupby("GAME_ID", sort=False):
        sides = paired_sides(group)
        if sides is None:
            skipped += 1
            continue
        away_row, home_row = sides
        game_date = str(home_row["GAME_DATE"])
        if game_date > AS_OF.isoformat():
            continue
        rows.append(
            {
                "event_id": str(game_id),
                "date": game_date,
                "season": int(home_row["season"]),
                "game_type": home_row["game_type"],
                "away": away_row["TEAM_ABBREVIATION"],
                "home": home_row["TEAM_ABBREVIATION"],
                "away_name": away_row["TEAM_NAME"],
                "home_name": home_row["TEAM_NAME"],
                "away_score": int(away_row["PTS"]),
                "home_score": int(home_row["PTS"]),
                "played": True,
                "venue": "",
            }
        )
    if skipped:
        print(f"unpaired game ids {skipped}")
    rows.sort(key=lambda row: (row["date"], row["event_id"]))
    return rows


def team_ids() -> list[dict]:
    payload = fetch_json(TEAMS_URL)
    teams = payload["sports"][0]["leagues"][0]["teams"]
    return [{"id": str(item["team"]["id"]), "name": item["team"]["displayName"]} for item in teams]


def collect_events(teams: list[dict]) -> list[dict]:
    urls = [
        SCHEDULE_URL.format(team_id=team["id"], season=season)
        for season in (2026, 2027)
        for team in teams
    ]
    urls.extend(
        SCHEDULE_URL.format(team_id=team["id"], season=2026) + "&seasontype=3"
        for team in teams
    )
    events: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_json, url): url for url in urls}
        for future in as_completed(futures):
            payload = future.result()
            for event in payload.get("events", []):
                events[str(event["id"])] = event
    return list(events.values())


def score_value(score: object) -> int | None:
    if isinstance(score, dict):
        score = score.get("displayValue", score.get("value"))
    if score in (None, ""):
        return None
    return int(float(score))


def espn_games(events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        competition = event["competitions"][0]
        competitors = competition.get("competitors", [])
        home = next((team for team in competitors if team.get("homeAway") == "home"), None)
        away = next((team for team in competitors if team.get("homeAway") == "away"), None)
        if home is None or away is None:
            continue
        stamp = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(EASTERN)
        status = competition.get("status", {}).get("type", {})
        rows.append(
            {
                "date": stamp.date().isoformat(),
                "away": official_abbrev(away["team"].get("abbreviation") or ""),
                "home": official_abbrev(home["team"].get("abbreviation") or ""),
                "away_name": away["team"].get("displayName") or "",
                "home_name": home["team"].get("displayName") or "",
                "away_score": score_value(away.get("score")),
                "home_score": score_value(home.get("score")),
                "status": status.get("name") or "",
                "venue": (competition.get("venue") or {}).get("fullName") or "",
            }
        )
    return rows


def compare_finals(api_rows: list[dict], espn_rows: list[dict]) -> str:
    api = {(row["date"], row["away"], row["home"]): (row["away_score"], row["home_score"]) for row in api_rows}
    matched = 0
    mismatched = 0
    missing = 0
    for row in espn_rows:
        if row["status"] != "STATUS_FINAL" or row["date"] > AS_OF.isoformat():
            continue
        if row["away_score"] == 0 and row["home_score"] == 0:
            continue
        key = (row["date"], row["away"], row["home"])
        found = api.get(key)
        if found is None:
            missing += 1
            if missing <= 8:
                print(f"missing from stats.nba.com {key} espn {row['away_score']}-{row['home_score']}")
            continue
        if found != (row["away_score"], row["home_score"]):
            mismatched += 1
            if mismatched <= 5:
                print(f"score mismatch {key} espn {row['away_score']}-{row['home_score']} api {found[0]}-{found[1]}")
            continue
        matched += 1
    text = f"ESPN finals matched {matched}, mismatched {mismatched}, missing from stats.nba.com {missing}."
    print(text)
    return text


def attach_venues(rows: list[dict], espn_rows: list[dict]) -> None:
    venues = {
        (row["date"], row["away"], row["home"]): row["venue"]
        for row in espn_rows
        if row["venue"]
    }
    for row in rows:
        row["venue"] = venues.get((row["date"], row["away"], row["home"]), "")


def upcoming_games(espn_rows: list[dict], completed: list[dict]) -> list[dict]:
    done = {(row["date"], row["away"], row["home"]) for row in completed}
    rows = []
    seen = set()
    for row in espn_rows:
        key = (row["date"], row["away"], row["home"])
        if key in seen or key in done or row["status"] == "STATUS_POSTPONED":
            continue
        game_date = date.fromisoformat(row["date"])
        if row["status"] == "STATUS_FINAL" or not (AS_OF <= game_date <= NEXT_THROUGH):
            continue
        seen.add(key)
        rows.append(
            {
                "date": row["date"],
                "season": 2026,
                "game_type": "REG",
                "away": row["away"],
                "home": row["home"],
                "away_name": row["away_name"],
                "home_name": row["home_name"],
                "away_score": None,
                "home_score": None,
                "played": False,
                "venue": row["venue"],
            }
        )
    rows.sort(key=lambda row: (row["date"], row["away"], row["home"]))
    return rows


def add_prior_form(rows: list[dict]) -> None:
    history: dict[str, list[tuple[date, int, int, int]]] = {}
    for row in rows:
        for side in ("home", "away"):
            games = history.get(row[side], [])
            if games:
                row[f"{side}_win_rate"] = sum(item[3] for item in games) / len(games)
                row[f"{side}_ppg"] = sum(item[1] for item in games) / len(games)
                row[f"{side}_papg"] = sum(item[2] for item in games) / len(games)
                row[f"{side}_rest_days"] = (date.fromisoformat(row["date"]) - games[-1][0]).days
            else:
                row[f"{side}_win_rate"] = None
                row[f"{side}_ppg"] = None
                row[f"{side}_papg"] = None
                row[f"{side}_rest_days"] = None
        if not row["played"]:
            continue
        home_win = int(row["home_score"] > row["away_score"])
        row["home_win"] = home_win
        history.setdefault(row["home"], []).append(
            (date.fromisoformat(row["date"]), row["home_score"], row["away_score"], home_win)
        )
        history.setdefault(row["away"], []).append(
            (date.fromisoformat(row["date"]), row["away_score"], row["home_score"], 1 - home_win)
        )


def win_probability(row: dict, center, scale, ridge, positive_is_home: bool) -> float | None:
    if any(row.get(feature) is None for feature in FEATURES):
        return None
    sample = (np.array([row[feature] for feature in FEATURES], dtype=float) - center) / scale
    decision = float(ridge.decision_function(sample.reshape(1, -1))[0])
    if not positive_is_home:
        decision = -decision
    return float(expit(decision))


def fit_model(rows: list[dict]):
    frame = pd.DataFrame(rows)
    matrix = frame[FEATURES].to_numpy(dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (matrix - center) / scale
    target = frame["home_win"].to_numpy(dtype=int)
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(scaled, target)
    home_points = LinearRegression().fit(scaled, frame["home_score"].to_numpy(dtype=float))
    away_points = LinearRegression().fit(scaled, frame["away_score"].to_numpy(dtype=float))
    positive_is_home = int(ridge.classes_[1]) == 1
    return center, scale, ridge, home_points, away_points, positive_is_home


def holdout_accuracy(completed: list[dict]) -> tuple[int, int, float | None]:
    train_rows = [
        row for row in completed
        if date.fromisoformat(row["date"]) < HOLDOUT_START and all(row.get(feature) is not None for feature in FEATURES)
    ]
    test_rows = [
        row for row in completed
        if date.fromisoformat(row["date"]) >= HOLDOUT_START and all(row.get(feature) is not None for feature in FEATURES)
    ]
    if len(train_rows) < 2 or len(test_rows) == 0 or len({row["home_win"] for row in train_rows}) < 2:
        return len(train_rows), len(test_rows), None
    center, scale, ridge, _, _, positive_is_home = fit_model(train_rows)
    correct = 0
    for row in test_rows:
        probability = win_probability(row, center, scale, ridge, positive_is_home)
        correct += int(probability is not None and int(probability >= 0.5) == row["home_win"])
    return len(train_rows), len(test_rows), correct / len(test_rows)


def score_rows(completed: list[dict], upcoming: list[dict]) -> int:
    trainable = [
        row for row in completed
        if row.get("home_win") is not None and all(row.get(feature) is not None for feature in FEATURES)
    ]
    if len(trainable) < 2 or len({row["home_win"] for row in trainable}) < 2:
        return 0
    center, scale, ridge, home_points, away_points, positive_is_home = fit_model(trainable)
    for row in completed:
        row["home_win_prob"] = win_probability(row, center, scale, ridge, positive_is_home)
    for row in upcoming:
        probability = win_probability(row, center, scale, ridge, positive_is_home)
        row["home_win_prob"] = probability
        if probability is None:
            continue
        sample = ((np.array([row[feature] for feature in FEATURES], dtype=float) - center) / scale).reshape(1, -1)
        row["predicted_away_points"] = float(away_points.predict(sample)[0])
        row["predicted_home_points"] = float(home_points.predict(sample)[0])
        row["predicted_points"] = row["predicted_away_points"] + row["predicted_home_points"]
    return len(trainable)


def season_team_stats(log: pd.DataFrame) -> pd.DataFrame:
    current = log[(log["season"] == 2026) & (log["game_type"] == "REG")].copy()
    if current.empty:
        return pd.DataFrame()
    opponent = current[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "opponent", "PTS": "points_against"}
    )
    paired = current.merge(opponent, on="GAME_ID")
    paired = paired[paired["TEAM_ABBREVIATION"] != paired["opponent"]]
    grouped = paired.groupby("TEAM_ABBREVIATION", as_index=False).agg(
        games=("GAME_ID", "nunique"),
        wins=("WL", lambda values: int((values == "W").sum())),
        points_for=("PTS", "sum"),
        points_against=("points_against", "sum"),
        **{column.lower(): (column, "sum") for column in COUNT_COLUMNS if column != "PTS"},
    )
    grouped["losses"] = grouped["games"] - grouped["wins"]
    grouped["ppg"] = grouped["points_for"] / grouped["games"]
    grouped["papg"] = grouped["points_against"] / grouped["games"]
    grouped["fg_pct"] = grouped["fgm"] / grouped["fga"]
    grouped["fg3_pct"] = grouped["fg3m"] / grouped["fg3a"]
    grouped["ft_pct"] = grouped["ftm"] / grouped["fta"]
    for column in ("oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"):
        grouped[column] = grouped[column] / grouped["games"]
    grouped = grouped.rename(columns={"TEAM_ABBREVIATION": "team"})
    order = [
        "team", "games", "wins", "losses", "points_for", "points_against", "ppg", "papg",
        "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct",
        "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
    ]
    return grouped[order].sort_values("team")


def pull_player_stats() -> tuple[int, str]:
    try:
        query = urllib.parse.urlencode(
            {
                "College": "",
                "Conference": "",
                "Country": "",
                "DateFrom": "",
                "DateTo": "",
                "Division": "",
                "DraftPick": "",
                "DraftYear": "",
                "GameScope": "",
                "GameSegment": "",
                "Height": "",
                "ISTRound": "",
                "LastNGames": "0",
                "LeagueID": "00",
                "Location": "",
                "MeasureType": "Base",
                "Month": "0",
                "OpponentTeamID": "0",
                "Outcome": "",
                "PORound": "",
                "PaceAdjust": "N",
                "PerMode": "PerGame",
                "Period": "0",
                "PlayerExperience": "",
                "PlayerPosition": "",
                "PlusMinus": "N",
                "Rank": "N",
                "Season": "2026-27",
                "SeasonSegment": "",
                "SeasonType": "Regular Season",
                "ShotClockRange": "",
                "StarterBench": "",
                "TeamID": "0",
                "TwoWay": "",
                "VsConference": "",
                "VsDivision": "",
                "Weight": "",
            }
        )
        payload = fetch_json(f"{PLAYER_NBA_URL}?{query}", headers=NBA_HEADERS)
        block = payload["resultSets"][0]
        frame = pd.DataFrame(block["rowSet"], columns=block["headers"])
        if frame.empty:
            raise ValueError("no player rows")
        PLAYER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(PLAYER_STATS_PATH, index=False)
        print(f"player rows {len(frame)} from stats.nba.com")
        return len(frame), "Player totals are the 2026-27 per-game base table from stats.nba.com."
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"stats.nba.com player table failed: {type(exc).__name__}")
    try:
        payload = fetch_json(PLAYER_ESPN_URL)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"player leaders failed: {type(exc).__name__}")
        return 0, "Player totals were not available."
    catalogs = {
        category["name"]: category.get("names") or []
        for category in payload.get("categories") or []
        if category.get("name")
    }
    rows = []
    for athlete in payload.get("athletes") or []:
        person = athlete.get("athlete") or {}
        teams = person.get("teams") or []
        record = {
            "player": person.get("displayName") or "",
            "team": (teams[0].get("abbreviation") if teams else "") or person.get("teamShortName") or "",
        }
        by_name = {category.get("name"): category for category in athlete.get("categories") or []}
        for category_name, names in catalogs.items():
            values = (by_name.get(category_name) or {}).get("values") or []
            for name, value in zip(names, values):
                record[f"{category_name}_{name}"] = value
        if record["player"]:
            rows.append(record)
    if not rows:
        return 0, "Player totals were not available."
    PLAYER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PLAYER_STATS_PATH, index=False)
    print(f"player rows {len(rows)} from ESPN")
    return len(rows), "Player leader rows are the ESPN 2026-27 regular-season table."


def write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame[columns].to_csv(path, index=False)


def main() -> None:
    log, log_notes = load_logs()
    if log.empty or not ((log["season"] == 2026) & (log["game_type"] == "REG")).any():
        raise SystemExit("stats.nba.com did not return the 2026-27 regular season")
    log = log[log["GAME_DATE"] <= AS_OF.isoformat()].copy()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)
    completed = games_from_log(log)
    teams = team_ids()
    print(f"teams {len(teams)}")
    espn_rows = espn_games(collect_events(teams))
    comparison = compare_finals(completed, espn_rows)
    attach_venues(completed, espn_rows)
    upcoming = upcoming_games(espn_rows, completed)
    ordered = sorted(completed + upcoming, key=lambda row: (row["date"], row.get("event_id", ""), row["away"], row["home"]))
    add_prior_form(ordered)
    train_rows = score_rows(completed, upcoming)
    hold_train, hold_test, accuracy = holdout_accuracy(completed)
    player_rows, player_note = pull_player_stats()
    team_stats = season_team_stats(log)
    team_stats.to_csv(TEAM_STATS_PATH, index=False)

    for row in completed + upcoming:
        row["game_id"] = f"{row['date']}_{row['away']}_{row['home']}"

    schedule_rows = [
        row for row in ordered
        if row["played"] or date.fromisoformat(row["date"]) >= AS_OF
    ]
    future_schedule = []
    seen = {(row["date"], row["away"], row["home"]) for row in schedule_rows}
    for row in espn_rows:
        key = (row["date"], row["away"], row["home"])
        if key in seen or row["status"] == "STATUS_POSTPONED" or row["date"] < AS_OF.isoformat():
            continue
        seen.add(key)
        future_schedule.append(
            {
                "date": row["date"],
                "season": 2026,
                "game_type": "REG",
                "away": row["away"],
                "home": row["home"],
                "away_name": row["away_name"],
                "home_name": row["home_name"],
                "away_score": None,
                "home_score": None,
                "venue": row["venue"],
            }
        )
    schedule_columns = [
        "date", "season", "game_type", "away", "home", "away_name", "home_name",
        "away_score", "home_score", "venue",
    ]
    write_table(SCHEDULE_PATH, schedule_rows + future_schedule, schedule_columns)
    completed_columns = [
        "game_id", "date", "season", "game_type", "away", "home", "away_score", "home_score", "home_win",
        *FEATURES, "home_win_prob",
    ]
    next_columns = completed_columns + ["predicted_away_points", "predicted_home_points", "predicted_points"]
    write_table(COMPLETED_PATH, completed, completed_columns)
    write_table(NEXT_PATH, upcoming, next_columns)
    seeds = [
        {
            "date": row["date"],
            "away": row["away"],
            "home": row["home"],
            "home_win_prob": row.get("home_win_prob"),
            "predicted_points": row.get("predicted_points"),
            "spread_line": "",
            "total_line": "",
            "lean": "",
            "line": LINE,
        }
        for row in upcoming
    ]
    write_table(
        SEEDS_PATH,
        seeds,
        ["date", "away", "home", "home_win_prob", "predicted_points", "spread_line", "total_line", "lean", "line"],
    )
    max_date = max(row["date"] for row in completed)
    min_date = min(row["date"] for row in completed)
    MAX_DATE_PATH.write_text(max_date + "\n", encoding="utf-8", newline="\n")
    postseason = sum(row["game_type"] == "POST" for row in completed)
    current_rows = sum(row["season"] == 2026 and row["game_type"] == "REG" for row in completed)
    scored_next = sum(row.get("home_win_prob") is not None for row in upcoming)
    zero_scores = sum(row["away_score"] == 0 and row["home_score"] == 0 for row in completed)
    README_PATH.write_text(
        "# NBA Game Predictor 2026\n\n"
        f"Completed games run from {min_date} through {max_date}. "
        "stats.nba.com leaguegamefinder is the score source for the 2025-26 regular season, "
        "the 2025-26 playoffs, and the 2026-27 regular season. ESPN team schedules supply "
        "the upcoming schedule and venues. Team abbreviations follow stats.nba.com. "
        "A sportsbook is not called. Refresh is manual. This does not price a bet.\n\n"
        "Run `python scripts/refresh_today.py` during the season. "
        "The script keeps results through its as-of date, 2026-11-07.\n\n"
        "Output files use the same seeds columns as nfl_game_predictor: "
        "date, away, home, home_win_prob, predicted_points, spread_line, total_line, lean, line. "
        "spread_line and total_line are blank. The line column says this is not a bet.\n",
        encoding="utf-8",
        newline="\n",
    )
    accuracy_text = "not computed" if accuracy is None else f"{accuracy:.4f}"
    NOTES_PATH.write_text(
        "Run python scripts/refresh_today.py during the season. It does not run on a timer.\n\n"
        "Completed scores are team rows from stats.nba.com leaguegamefinder. "
        + " ".join(log_notes)
        + " ESPN schedules are used for venues and for games that have not been played. "
        "ESPN marks some postponed games as post with a 0-0 score. Those games are not stored. "
        f"{comparison} The ESPN final dated 2025-12-16, SAS at NYK, 113-124, is not in the league log, so it is not stored. "
        f"{player_note} "
        "Form uses only earlier games: win rate, points scored, points allowed, and rest days. "
        "A game does not train on its own score. The published home_win_prob on a completed game is in-sample. "
        "The held-out accuracy below is a separate fit. The win model is a ridge classifier. "
        "Points use linear regression.\n\n"
        "There is no recorded public closing line in this table. spread_line and total_line are blank. "
        "The line column says this is not a bet. This does not price a bet.\n\n"
        "ESPN abbreviations GS, NO, NY, SA, UTAH, and WSH are stored as GSW, NOP, NYK, SAS, UTA, and WAS.\n\n"
        f"Completed rows: {len(completed)}\n"
        f"2026-27 regular-season games: {current_rows}\n"
        f"2025-26 playoff games: {postseason}\n"
        f"Zero-score rows stored: {zero_scores}\n"
        f"Training rows: {train_rows}\n"
        f"Feature count: {len(FEATURES)}\n"
        f"Minimum date: {min_date}\n"
        f"Maximum date: {max_date}\n"
        f"Next games through {NEXT_THROUGH.isoformat()}: {len(upcoming)}\n"
        f"Next games with home_win_prob: {scored_next}\n"
        f"Held-out games from {HOLDOUT_START.isoformat()}: {hold_test} of {hold_train} training rows, accuracy {accuracy_text}\n"
        f"Current-season team rows: {len(team_stats)}\n"
        f"Player rows: {player_rows}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"completed {len(completed)} {min_date} {max_date} postseason {postseason}")
    print(f"next {len(upcoming)} scored {scored_next}")
    print(f"train_rows {train_rows}")
    print(f"holdout {hold_train} {hold_test} {accuracy_text}")
    print(f"teams in snapshot {len(team_stats)}")
    print(f"zero scores {zero_scores}")


if __name__ == "__main__":
    main()
