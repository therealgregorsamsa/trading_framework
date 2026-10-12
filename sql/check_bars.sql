SELECT
    symbol,
    COUNT(*) AS n_bars,
    SUM(n_quotes) AS total_quotes
FROM read_parquet('data/bars/bars.parquet')
GROUP BY symbol
ORDER BY symbol