"""Check the NBA artifacts after a refresh. This does not train a model."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REQUIRED_NEXT = (
    "game_id",
    "date",
    "home",
    "away",
    "home_win_prob",
    "predicted_home_points",
    "predicted_away_points",
    "predicted_total",
    "confidence_tier",
    "pick_side",
)
HIGH_FLOOR = 0.60
LOW_FLOOR = 0.40


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fail(message: str) -> None:
    print(f"fail {message}")


def number(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    return float(text)


def main() -> int:
    problems = 0
    completed_path = ARTIFACTS / "nba_completed.csv"
    next_path = ARTIFACTS / "nba_next_games.csv"
    high_path = ARTIFACTS / "high_confidence.csv"
    calibration_path = ARTIFACTS / "calibration.csv"
    max_path = ARTIFACTS / "last_max_date.txt"
    for path in (completed_path, next_path, high_path, calibration_path, max_path):
        if not path.exists():
            fail(f"missing {path.name}")
            problems += 1
    if problems:
        print(f"validation fail {problems}")
        return 1

    completed = read_rows(completed_path)
    upcoming = read_rows(next_path)
    high_rows = read_rows(high_path)
    calibration = read_rows(calibration_path)
    recorded_max = max_path.read_text(encoding="utf-8").strip()
    today = date.today().isoformat()

    if not completed:
        fail("completed file is empty")
        problems += 1
    ids = [row["game_id"] for row in completed]
    if len(ids) != len(set(ids)):
        fail("duplicate completed game_id")
        problems += 1
    dates = [row["date"] for row in completed if row.get("date")]
    if dates and max(dates) != recorded_max:
        fail(f"last_max_date {recorded_max} does not match {max(dates)}")
        problems += 1
    if dates and max(dates) > today:
        fail(f"completed date {max(dates)} is after {today}")
        problems += 1
    missing_scores = sum(1 for row in completed if row.get("away_score") in (None, "") or row.get("home_score") in (None, ""))
    if missing_scores:
        fail(f"{missing_scores} completed rows lack scores")
        problems += 1

    next_ids = [row["game_id"] for row in upcoming]
    if len(next_ids) != len(set(next_ids)):
        fail("duplicate next game_id")
        problems += 1
    header = upcoming[0].keys() if upcoming else []
    for column in REQUIRED_NEXT:
        if column not in header:
            fail(f"next games missing {column}")
            problems += 1
    for row in upcoming:
        if dates and row["date"] < max(dates):
            fail(f"next game {row['game_id']} is before the last completed date")
            problems += 1
            break
        probability = number(row.get("home_win_prob", ""))
        if probability is None:
            continue
        if not 0.0 < probability < 1.0:
            fail(f"{row['game_id']} probability {probability} is outside (0, 1)")
            problems += 1
        expected = "home" if probability >= 0.5 else "away"
        if row.get("pick_side") != expected:
            fail(f"{row['game_id']} pick_side does not match the probability")
            problems += 1
        home_points = number(row.get("predicted_home_points", ""))
        away_points = number(row.get("predicted_away_points", ""))
        total = number(row.get("predicted_total", ""))
        if home_points is None or away_points is None or total is None:
            fail(f"{row['game_id']} is missing a score projection")
            problems += 1
        elif abs((home_points + away_points) - total) > 0.02:
            fail(f"{row['game_id']} predicted_total does not match the two scores")
            problems += 1
        tier = row.get("confidence_tier")
        high = probability >= HIGH_FLOOR or probability <= LOW_FLOOR
        if high and tier != "high":
            fail(f"{row['game_id']} should be high confidence")
            problems += 1
        if not high and tier == "high":
            fail(f"{row['game_id']} is marked high without clearing 0.60 or 0.40")
            problems += 1

    high_ids = {row["game_id"] for row in high_rows}
    expected_ids = {row["game_id"] for row in upcoming if row.get("confidence_tier") == "high"}
    if high_ids != expected_ids:
        fail("high_confidence.csv does not match the next-game tiers")
        problems += 1

    floor_row = next(
        (row for row in calibration if row.get("split") == "in_sample" and row.get("bin") == "50-55"),
        None,
    )
    if floor_row is None:
        fail("calibration is missing the in-sample 50-55 bin")
        problems += 1
    else:
        rate = number(floor_row.get("home_win_rate", ""))
        if rate is not None and rate < 0.50 and floor_row.get("calibrated") == "yes":
            fail("50-55 bin is marked yes while its rate is below the floor")
            problems += 1

    print(f"completed {len(completed)}")
    print(f"next {len(upcoming)}")
    print(f"high {len(high_rows)}")
    print(f"max_date {recorded_max}")
    if problems:
        print(f"validation fail {problems}")
        return 1
    print("validation pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
