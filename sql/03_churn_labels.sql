WITH observation_customers AS (
    SELECT DISTINCT CustomerID
    FROM transactions
    WHERE InvoiceDate < TIMESTAMP '2011-07-01'
      AND CustomerID IS NOT NULL
),
future_customers AS (
    SELECT DISTINCT CustomerID
    FROM transactions
    WHERE InvoiceDate >= TIMESTAMP '2011-07-01'
      AND CustomerID IS NOT NULL
)
SELECT
    o.CustomerID,
    CASE WHEN f.CustomerID IS NULL THEN 1 ELSE 0 END AS churned
FROM observation_customers o
LEFT JOIN future_customers f USING (CustomerID);
