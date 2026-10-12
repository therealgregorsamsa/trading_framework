SELECT *
FROM read_parquet('data/clean/**/*.parquet', hive_partitioning=true,union_by_name=true)
WHERE symbol = 'EURUSD'
LIMIT 5