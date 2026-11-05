# Subscription churn features

Sample project on the line between data engineering and modeling. Subscription history is turned into one row per account, the label looks only at the next 30 days, and a logistic regression is a baseline rather than the point of the repo.

## What this shows

- Point-in-time features: every aggregate stops at the observation date
- A label window that does not overlap the features
- Accounts that already cancelled are removed from scoring
- A scaler fit on the training fold only
- A held-out AUC check so the synthetic signal is real

## Time cut

```text
history ......................| T observation | label window
events with date < T          |               | (T, T+30 days]
features are computed here    | not scored if | churned = 1 if a
                              | cancelled     | cancellation falls
                              | on or before T| inside the window
```

`days_since_last_usage` is 365 when the account has no usage before T. That sentinel is in the feature catalog written next to the metrics file.

Country is on the account file and is not a model feature. The sample is too small for a high-cardinality country effect, and the column is there to show that not every field belongs in the matrix.

Plans are `basic`, `plus`, and `pro`. `basic` is the reference level. `plan_plus` and `plan_pro` are the two indicators.

## Run

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m churnlab generate
PYTHONPATH=src python -m churnlab train
PYTHONPATH=src python -m churnlab score
pytest -q
```

`data/metrics.json` has the held-out AUC. `data/scores.csv` has one probability per account that was still active on the observation date.

The generator writes 2,000 accounts, seed 7, observation date 2026-01-01. The label is a noisy function of failed payments, tickets, and usage in the 30 days before that date. On that draw the held-out ROC AUC is 0.9388, with a positive rate of about 0.29, 1,500 training rows, and 500 test rows. Tests also use a smaller draw plus a hand-built fixture that pins the 7-day and 30-day windows.

## Portfolio

- [medallion-retail-pipeline](https://github.com/saigadaparthi39-png/medallion-retail-pipeline) publishes the warehouse this style of feature would read from
- [rag-knowledge-assistant](https://github.com/saigadaparthi39-png/rag-knowledge-assistant) answers from the note that defines this label window

## What I would add next

Multiple observation dates per account, with features rebuilt as-of each date, and a check that training rows from the same account do not land on both sides of the split. One snapshot per account is enough to show the time cut. It is not enough for a production training set.
