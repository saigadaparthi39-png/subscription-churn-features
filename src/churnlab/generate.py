"""Synthetic subscriptions whose churn label follows usage and billing stress."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

from churnlab.features import write_rows

OBSERVATION_DATE = date(2026, 1, 1)


def generate(raw_dir: Path, n_accounts: int = 2000, seed: int = 7) -> None:
    rng = random.Random(seed)
    accounts = []
    usage = []
    payments = []
    tickets = []
    cancellations = []

    for account_id in range(1, n_accounts + 1):
        # One latent risk drives sparse usage, failed payments, tickets, and the label.
        # The model only sees the aggregates, so a high AUC means those aggregates kept the signal.
        risk = rng.random()
        signup = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 900))
        plan = rng.choices(["basic", "plus", "pro"], weights=[5, 3, 2])[0]
        amount = {"basic": "40.00", "plus": "70.00", "pro": "110.00"}[plan]
        accounts.append(
            {
                "account_id": str(account_id),
                "plan": plan,
                "country": rng.choice(["US", "CA", "UK"]),
                "signup_date": signup.isoformat(),
                "observation_date": OBSERVATION_DATE.isoformat(),
            }
        )
        daily_probability = 0.12 + 0.8 * (1 - risk)
        usage_30 = 0
        for days_back in range(1, 91):
            if rng.random() < daily_probability:
                events = max(1, 1 + int(rng.random() * (1 + 6 * (1 - risk))))
                if days_back <= 30:
                    usage_30 += events
                usage.append(
                    {
                        "account_id": str(account_id),
                        "event_date": (OBSERVATION_DATE - timedelta(days=days_back)).isoformat(),
                        "events": str(events),
                    }
                )
        # 6, 16, and 28 fall inside the 30-day feature window. 64 does not.
        failed_in_window = 0
        for days_back in (6, 16, 28, 64):
            failed = rng.random() < (0.05 + 0.85 * risk)
            if failed and days_back <= 30:
                failed_in_window += 1
            payments.append(
                {
                    "account_id": str(account_id),
                    "event_date": (OBSERVATION_DATE - timedelta(days=days_back)).isoformat(),
                    "status": "failed" if failed else "ok",
                    "amount": amount,
                }
            )
        ticket_count = rng.choices(
            [0, 1, 2, 3],
            weights=[
                0.82 - 0.55 * risk,
                0.12,
                0.04 + 0.25 * risk,
                0.02 + 0.3 * risk,
            ],
        )[0]
        for _ in range(ticket_count):
            tickets.append(
                {
                    "account_id": str(account_id),
                    "opened_date": (
                        OBSERVATION_DATE - timedelta(days=rng.randrange(1, 31))
                    ).isoformat(),
                }
            )
        # The label is a noisy function of the same 30-day aggregates the feature query builds.
        logit = -2.4 + 1.45 * failed_in_window + 0.45 * ticket_count - 0.05 * usage_30
        churn_probability = 1 / (1 + math.exp(-max(-8.0, min(8.0, logit))))
        if rng.random() < churn_probability:
            cancellations.append(
                {
                    "account_id": str(account_id),
                    "cancel_date": (
                        OBSERVATION_DATE + timedelta(days=rng.randrange(1, 31))
                    ).isoformat(),
                }
            )

    write_rows(
        raw_dir / "accounts.csv",
        accounts,
        ["account_id", "plan", "country", "signup_date", "observation_date"],
    )
    write_rows(raw_dir / "usage.csv", usage, ["account_id", "event_date", "events"])
    write_rows(
        raw_dir / "payments.csv",
        payments,
        ["account_id", "event_date", "status", "amount"],
    )
    write_rows(raw_dir / "tickets.csv", tickets, ["account_id", "opened_date"])
    write_rows(
        raw_dir / "cancellations.csv",
        cancellations,
        ["account_id", "cancel_date"],
    )
