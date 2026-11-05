import csv

from churnlab.features import compute_features
from churnlab.generate import generate
from churnlab.train import score, train


def test_held_out_auc_and_scores_skip_prior_cancels(tmp_path):
    raw = tmp_path / "raw"
    generate(raw, n_accounts=1600, seed=7)
    metrics = train(raw, tmp_path / "model.joblib", tmp_path / "metrics.json")
    assert metrics["auc"] >= 0.80
    assert metrics["train_rows"] + metrics["test_rows"] == 1600
    assert "usage_events_30d" in {item["name"] for item in metrics["features"]}

    written = score(raw, tmp_path / "model.joblib", tmp_path / "scores.csv")
    with (tmp_path / "scores.csv").open() as handle:
        scored = {row["account_id"] for row in csv.DictReader(handle)}
    assert written == len(scored) == 1600

    # A cancellation on the observation date removes the account from scoring.
    with (raw / "cancellations.csv").open("a") as handle:
        handle.write("1,2026-01-01\n")
    features = {row["account_id"]: row for row in compute_features(raw)}
    assert int(features["1"]["prior_cancellations"]) >= 1
    score(raw, tmp_path / "model.joblib", tmp_path / "scores.csv")
    with (tmp_path / "scores.csv").open() as handle:
        scored_again = {row["account_id"] for row in csv.DictReader(handle)}
    assert "1" not in scored_again
    assert len(scored_again) == 1599
