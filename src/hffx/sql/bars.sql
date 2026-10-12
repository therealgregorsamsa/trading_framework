SELECT
    symbol,
    session_id,
    date_trunc('second', ts) + INTERVAL 1 SECOND AS bar_ts,
    COUNT(*) AS n_quotes,
    arg_min(mid, ts) AS open,
    arg_max(mid, ts) AS close,
    MAX(mid) AS high,
    MIN(mid) AS low,
    AVG(mid) AS mean,
    arg_max(ask - bid, ts) AS spread
FROM(
    SELECT*, (bid + ask) / 2 AS mid
    FROM read_parquet('data/clean/**/*.parquet', hive_partitioning = true, union_by_name = true)
)
GROUP BY symbol, session_id, date_trunc('second', ts)
ORDER BY bar_ts
