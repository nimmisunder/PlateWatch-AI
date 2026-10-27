# ai_tools.py

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import Counter


# Use the same database as main.py
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "vehicle_entries.db"


def get_connection():
    """Open the existing PlateWatch database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. "
            "Make sure main.py has created the database."
        )

    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_plate(plate_number):
    """Normalize a registration number for comparison."""
    return re.sub(r"[^A-Z0-9]", "", plate_number.upper())


def count_entries_today():
    """Count vehicle entries recorded today."""
    today = datetime.now().astimezone().date().isoformat()

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM vehicle_entries
            WHERE substr(entry_time, 1, 10) = ?
            """,
            (today,),
        ).fetchone()

    return {
        "date": today,
        "total_entries": row["total"],
    }


def find_repeat_vehicles():
    """Find registration numbers recorded more than once."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT plate_number, COUNT(*) AS entry_count
            FROM vehicle_entries
            GROUP BY UPPER(
                REPLACE(
                    REPLACE(
                        REPLACE(plate_number, ' ', ''),
                        '-', ''
                    ),
                    '.', ''
                )
            )
            HAVING COUNT(*) > 1
            ORDER BY entry_count DESC
            """
        ).fetchall()

    # Merge variants such as KL 01 AB 1234 and KL01AB1234.
    counts = Counter()
    display_names = {}

    for row in rows:
        key = normalize_plate(row["plate_number"])
        counts[key] += row["entry_count"]
        display_names.setdefault(key, row["plate_number"])

    vehicles = [
        {
            "plate_number": display_names[key],
            "entry_count": count,
        }
        for key, count in counts.items()
        if count > 1
    ]

    vehicles.sort(key=lambda item: item["entry_count"], reverse=True)

    return {
        "repeat_vehicle_count": len(vehicles),
        "vehicles": vehicles,
    }


def summarize_today():
    """Summarize today's recorded vehicle activity."""
    today = datetime.now().astimezone().date().isoformat()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT plate_number, entry_time
            FROM vehicle_entries
            WHERE substr(entry_time, 1, 10) = ?
            ORDER BY entry_time
            """,
            (today,),
        ).fetchall()

    plates = [normalize_plate(row["plate_number"]) for row in rows]
    plate_counts = Counter(plates)

    repeated_vehicles = [
        {
            "plate_number": plate,
            "entry_count": count,
        }
        for plate, count in plate_counts.items()
        if count > 1
    ]

    return {
        "date": today,
        "total_entries": len(rows),
        "unique_vehicles": len(plate_counts),
        "vehicles_entered_multiple_times": repeated_vehicles,
        "first_entry": rows[0]["entry_time"] if rows else None,
        "last_entry": rows[-1]["entry_time"] if rows else None,
    }


def count_entries_for_plate(plate_number):
    """Count entries for a specific registration number."""
    target = normalize_plate(plate_number)

    if not target:
        return {
            "plate_number": plate_number,
            "entry_count": 0,
            "message": "No valid registration number was provided.",
        }

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT plate_number, entry_time, status
            FROM vehicle_entries
            ORDER BY entry_time
            """
        ).fetchall()

    matches = [
        {
            "plate_number": row["plate_number"],
            "entry_time": row["entry_time"],
            "status": row["status"],
        }
        for row in rows
        if normalize_plate(row["plate_number"]) == target
    ]

    return {
        "plate_number": plate_number,
        "entry_count": len(matches),
        "entries": matches,
    }


def find_busiest_hour_today():
    """Find the hour with the most recorded entries today."""
    today = datetime.now().astimezone().date().isoformat()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT substr(entry_time, 12, 2) AS hour,
                   COUNT(*) AS entry_count
            FROM vehicle_entries
            WHERE substr(entry_time, 1, 10) = ?
            GROUP BY hour
            ORDER BY entry_count DESC, hour ASC
            """,
            (today,),
        ).fetchall()

    if not rows:
        return {
            "date": today,
            "busiest_hours": [],
            "message": "No vehicle entries were recorded today.",
        }

    highest_count = rows[0]["entry_count"]

    busiest_hours = [
        {
            "hour": f'{row["hour"]}:00–{row["hour"]}:59',
            "entry_count": row["entry_count"],
        }
        for row in rows
        if row["entry_count"] == highest_count
    ]

    return {
        "date": today,
        "busiest_hours": busiest_hours,
        "entry_count": highest_count,
    }