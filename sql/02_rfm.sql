WITH customer_rfm AS (
    SELECT
        CustomerID,
        DATEDIFF('day', MAX(InvoiceDate), MAX(MAX(InvoiceDate)) OVER () + INTERVAL 1 DAY) AS recency,
        COUNT(DISTINCT InvoiceNo) AS frequency,
        SUM(Quantity * UnitPrice) AS monetary
    FROM transactions
    WHERE Quantity > 0 AND UnitPrice > 0
      AND InvoiceNo NOT LIKE 'C%'
      AND CustomerID IS NOT NULL
    GROUP BY CustomerID
)
SELECT *,
       NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
       NTILE(5) OVER (ORDER BY frequency) AS f_score,
       NTILE(5) OVER (ORDER BY monetary) AS m_score
FROM customer_rfm;
