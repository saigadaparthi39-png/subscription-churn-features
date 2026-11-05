from datetime import date, timedelta

from churnlab.features import compute_features, write_rows

OBSERVATION = date(2026, 1, 1)


def _fixture(raw):
    write_rows(
        raw / "accounts.csv",
        [
            {
                "account_id": "1",
                "plan": "basic",
                "country": "US",
                "signup_date": "2025-01-01",
                "observation_date": OBSERVATION.isoformat(),
            },
            {
                "account_id": "2",
                "plan": "plus",
                "country": "US",
                "signup_date": "2025-06-01",
                "observation_date": OBSERVATION.isoformat(),
            },
            {
                "account_id": "3",
                "plan": "pro",
                "country": "CA",
                "signup_date": "2024-01-01",
                "observation_date": OBSERVATION.isoformat(),
            },
        ],
        ["account_id", "plan", "country", "signup_date", "observation_date"],
    )
    write_rows(
        raw / "usage.csv",
        [
            {"account_id": "1", "event_date": "2025-12-20", "events": "3"},
            {"account_id": "1", "event_date": "2025-12-31", "events": "2"},
            {"account_id": "1", "event_date": "2026-01-01", "events": "9"},
            {"account_id": "3", "event_date": "2026-01-01", "events": "5"},
        ],
        ["account_id", "event_date", "events"],
    )
    write_rows(
        raw / "payments.csv",
        [
            {"account_id": "1", "event_date": "2025-12-15", "status": "failed", "amount": "40"},
            {"account_id": "1", "event_date": "2026-01-05", "status": "failed", "amount": "40"},
            {"account_id": "2", "event_date": "2025-12-01", "status": "ok", "amount": "70"},
        ],
        ["account_id", "event_date", "status", "amount"],
    )
    write_rows(
        raw / "tickets.csv",
        [
            {"account_id": "1", "opened_date": "2025-12-30"},
            {"account_id": "1", "opened_date": "2026-01-02"},
        ],
        ["account_id", "opened_date"],
    )
    write_rows(
        raw / "cancellations.csv",
        [
            {"account_id": "1", "cancel_date": "2026-01-10"},
            {"account_id": "2", "cancel_date": "2026-01-01"},
        ],
        ["account_id", "cancel_date"],
    )


def _by_id(rows):
    return {row["account_id"]: row for row in rows}


def test_windows_stop_at_the_observation_date(tmp_path):
    raw = tmp_path / "raw"
    _fixture(raw)
    rows = _by_id(compute_features(raw))

    first = rows["1"]
    assert int(first["tenure_days"]) == (OBSERVATION - date(2025, 1, 1)).days
    assert int(first["usage_events_30d"]) == 5
    assert int(first["usage_events_7d"]) == 2
    assert int(first["failed_payments_30d"]) == 1
    assert int(first["tickets_30d"]) == 1
    assert int(first["days_since_last_usage"]) == 1
    assert int(first["prior_cancellations"]) == 0
    assert int(first["churned"]) == 1

    already_gone = rows["2"]
    assert int(already_gone["prior_cancellations"]) == 1
    assert int(already_gone["churned"]) == 0
    assert int(already_gone["days_since_last_usage"]) == 365

    future_only = rows["3"]
    assert int(future_only["usage_events_30d"]) == 0
    assert int(future_only["days_since_last_usage"]) == 365
    assert int(future_only["churned"]) == 0


def test_events_after_observation_do_not_change_features(tmp_path):
    raw = tmp_path / "raw"
    _fixture(raw)
    before = _by_id(compute_features(raw))["1"]

    with (raw / "usage.csv").open("a") as handle:
        handle.write("1,2026-01-15,50\n")
    with (raw / "payments.csv").open("a") as handle:
        handle.write("1,2026-01-20,failed,40\n")
    with (raw / "tickets.csv").open("a") as handle:
        handle.write("1,2026-01-18\n")
    after = _by_id(compute_features(raw))["1"]
    assert after == before

    with (raw / "usage.csv").open("a") as handle:
        handle.write("1,2025-12-30,4\n")
    changed = _by_id(compute_features(raw))["1"]
    assert int(changed["usage_events_30d"]) == 9
    assert int(changed["usage_events_7d"]) == 6
    assert changed["observation_date"] == before["observation_date"]
    assert OBSERVATION - timedelta(days=7) <= date(2025, 12, 30)
