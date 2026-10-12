import duckdb

query = """
SELECT
    COUNT(*) AS n,
    COUNT(*) - COUNT(ask_5m) AS null_ask_5m,
    COUNT(*) - COUNT(bid_5m) AS null_bid_5m,
    COUNT(*) - COUNT(ask_3m) AS null_ask_3m,
    AVG((ask_3m = ask)::INT) AS share_ask_3m_equals_top,
    AVG((ask_5m = ask)::INT) AS share_ask_5m_equals_top,
    AVG((bid_5m = bid)::INT) AS share_bid_5m_equals_top,
    quantile_cont((ask_5m - ask) * 1e4, [0.05, 0.25, 0.5, 0.75, 0.95]) AS ask_slip_5m_pips,
    quantile_cont((bid - bid_5m) * 1e4, [0.05, 0.25, 0.5, 0.75, 0.95]) AS bid_slip_5m_pips
FROM read_parquet('data/clean/**/*.parquet', hive_partitioning=true, union_by_name=true)
WHERE symbol = 'EURUSD'
"""
print(duckdb.sql(query).df().T)