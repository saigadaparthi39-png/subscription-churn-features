CREATE OR REPLACE TABLE features AS
WITH accounts_t AS (
    SELECT
        CAST(account_id AS VARCHAR) AS account_id,
        lower(trim(CAST(plan AS VARCHAR))) AS plan,
        CAST(country AS VARCHAR) AS country,
        CAST(signup_date AS DATE) AS signup_date,
        CAST(observation_date AS DATE) AS observation_date
    FROM accounts
),
usage_t AS (
    SELECT
        CAST(account_id AS VARCHAR) AS account_id,
        CAST(event_date AS DATE) AS event_date,
        CAST(events AS INTEGER) AS events
    FROM usage
),
payments_t AS (
    SELECT
        CAST(account_id AS VARCHAR) AS account_id,
        CAST(event_date AS DATE) AS event_date,
        lower(CAST(status AS VARCHAR)) AS status
    FROM payments
),
tickets_t AS (
    SELECT
        CAST(account_id AS VARCHAR) AS account_id,
        CAST(opened_date AS DATE) AS opened_date
    FROM tickets
),
cancellations_t AS (
    SELECT
        CAST(account_id AS VARCHAR) AS account_id,
        CAST(cancel_date AS DATE) AS cancel_date
    FROM cancellations
),
usage_30 AS (
    SELECT u.account_id, sum(u.events) AS usage_events_30d
    FROM usage_t u
    JOIN accounts_t a ON a.account_id = u.account_id
    WHERE u.event_date >= a.observation_date - INTERVAL 30 DAY
      AND u.event_date < a.observation_date
    GROUP BY 1
),
usage_7 AS (
    SELECT u.account_id, sum(u.events) AS usage_events_7d
    FROM usage_t u
    JOIN accounts_t a ON a.account_id = u.account_id
    WHERE u.event_date >= a.observation_date - INTERVAL 7 DAY
      AND u.event_date < a.observation_date
    GROUP BY 1
),
last_usage AS (
    SELECT u.account_id, max(u.event_date) AS last_usage_date
    FROM usage_t u
    JOIN accounts_t a ON a.account_id = u.account_id
    WHERE u.event_date < a.observation_date
    GROUP BY 1
),
failed AS (
    SELECT p.account_id, count(*) AS failed_payments_30d
    FROM payments_t p
    JOIN accounts_t a ON a.account_id = p.account_id
    WHERE p.status = 'failed'
      AND p.event_date >= a.observation_date - INTERVAL 30 DAY
      AND p.event_date < a.observation_date
    GROUP BY 1
),
ticks AS (
    SELECT t.account_id, count(*) AS tickets_30d
    FROM tickets_t t
    JOIN accounts_t a ON a.account_id = t.account_id
    WHERE t.opened_date >= a.observation_date - INTERVAL 30 DAY
      AND t.opened_date < a.observation_date
    GROUP BY 1
),
prior_cancel AS (
    SELECT c.account_id, count(*) AS prior_cancellations
    FROM cancellations_t c
    JOIN accounts_t a ON a.account_id = c.account_id
    WHERE c.cancel_date <= a.observation_date
    GROUP BY 1
),
labels AS (
    SELECT c.account_id, 1 AS churned
    FROM cancellations_t c
    JOIN accounts_t a ON a.account_id = c.account_id
    WHERE c.cancel_date > a.observation_date
      AND c.cancel_date <= a.observation_date + INTERVAL 30 DAY
    GROUP BY 1
)
SELECT
    a.account_id,
    a.plan,
    a.country,
    a.signup_date,
    a.observation_date,
    date_diff('day', a.signup_date, a.observation_date) AS tenure_days,
    coalesce(u30.usage_events_30d, 0) AS usage_events_30d,
    coalesce(u7.usage_events_7d, 0) AS usage_events_7d,
    coalesce(f.failed_payments_30d, 0) AS failed_payments_30d,
    coalesce(tk.tickets_30d, 0) AS tickets_30d,
    coalesce(date_diff('day', lu.last_usage_date, a.observation_date), 365) AS days_since_last_usage,
    coalesce(pc.prior_cancellations, 0) AS prior_cancellations,
    coalesce(lb.churned, 0) AS churned
FROM accounts_t a
LEFT JOIN usage_30 u30 ON u30.account_id = a.account_id
LEFT JOIN usage_7 u7 ON u7.account_id = a.account_id
LEFT JOIN last_usage lu ON lu.account_id = a.account_id
LEFT JOIN failed f ON f.account_id = a.account_id
LEFT JOIN ticks tk ON tk.account_id = a.account_id
LEFT JOIN prior_cancel pc ON pc.account_id = a.account_id
LEFT JOIN labels lb ON lb.account_id = a.account_id;
