SELECT
    CustomerID,
    DATEDIFF('day', MAX(InvoiceDate), DATE '2011-07-01') AS recency_days,
    COUNT(DISTINCT InvoiceNo) AS frequency,
    SUM(Quantity * UnitPrice) AS monetary,
    COUNT(DISTINCT CAST(InvoiceDate AS DATE)) AS unique_purchase_days,
    COUNT(DISTINCT StockCode) AS unique_products
FROM transactions
WHERE InvoiceDate < TIMESTAMP '2011-07-01'
  AND Quantity > 0
  AND UnitPrice > 0
  AND InvoiceNo NOT LIKE 'C%'
  AND CustomerID IS NOT NULL
GROUP BY CustomerID;
