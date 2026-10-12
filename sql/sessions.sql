SELECT 
session_id,
COUNT(*) AS n_quotes,
MIN(ts) AS start_ts,
MAX(ts) AS end_ts
FROM read_parquet('data/clean/**/*.parquet', hive_partitioning = true, union_by_name = true)
WHERE symbol = 'EURUSD'
GROUP BY session_id
ORDER BY session_id

