"""Build one feature row per account, using only events before the label."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import numpy as np

SQL_PATH = Path(__file__).parent / "sql" / "features.sql"

SCHEMAS = {
    "accounts": {
        "account_id": "VARCHAR",
        "plan": "VARCHAR",
        "country": "VARCHAR",
        "signup_date": "DATE",
        "observation_date": "DATE",
    },
    "usage": {
        "account_id": "VARCHAR",
        "event_date": "DATE",
        "events": "INTEGER",
    },
    "payments": {
        "account_id": "VARCHAR",
        "event_date": "DATE",
        "status": "VARCHAR",
        "amount": "DOUBLE",
    },
    "tickets": {
        "account_id": "VARCHAR",
        "opened_date": "DATE",
    },
    "cancellations": {
        "account_id": "VARCHAR",
        "cancel_date": "DATE",
    },
}

FEATURE_CATALOG = [
    {
        "name": "tenure_days",
        "description": "Days from signup to the observation date.",
    },
    {
        "name": "usage_events_7d",
        "description": "Sum of usage events in [observation_date - 7 days, observation_date).",
    },
    {
        "name": "usage_events_30d",
        "description": "Sum of usage events in [observation_date - 30 days, observation_date).",
    },
    {
        "name": "failed_payments_30d",
        "description": "Failed payments in the 30 days before the observation date.",
    },
    {
        "name": "tickets_30d",
        "description": "Support tickets opened in the 30 days before the observation date.",
    },
    {
        "name": "days_since_last_usage",
        "description": "Days since the last usage before observation. 365 when there is none.",
    },
    {
        "name": "prior_cancellations",
        "description": "Cancellations on or before the observation date. These accounts are not scored.",
    },
    {
        "name": "churned",
        "description": "1 when a cancellation falls in (observation_date, observation_date + 30 days].",
    },
]

MODEL_FEATURES = [
    "tenure_days",
    "usage_events_7d",
    "usage_events_30d",
    "failed_payments_30d",
    "tickets_30d",
    "days_since_last_usage",
    "plan_plus",
    "plan_pro",
]


def _columns(schema: dict[str, str]) -> str:
    return ", ".join(f"'{name}': '{dtype}'" for name, dtype in schema.items())


def compute_features(raw_dir: Path) -> list[dict]:
    con = duckdb.connect()
    try:
        for table, schema in SCHEMAS.items():
            path = (raw_dir / f"{table}.csv").resolve().as_posix().replace("'", "''")
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM "
                f"read_csv('{path}', header=true, columns={{{_columns(schema)}}})"
            )
        con.execute(SQL_PATH.read_text())
        cursor = con.execute("SELECT * FROM features ORDER BY account_id")
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        con.close()


def scoring_population(rows: list[dict]) -> list[dict]:
    """Accounts that had not already cancelled by the observation date."""
    return [row for row in rows if int(row["prior_cancellations"]) == 0]


def design_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in rows:
        features.append(
            [
                float(row["tenure_days"]),
                float(row["usage_events_7d"]),
                float(row["usage_events_30d"]),
                float(row["failed_payments_30d"]),
                float(row["tickets_30d"]),
                float(row["days_since_last_usage"]),
                1.0 if row["plan"] == "plus" else 0.0,
                1.0 if row["plan"] == "pro" else 0.0,
            ]
        )
        labels.append(int(row["churned"]))
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
