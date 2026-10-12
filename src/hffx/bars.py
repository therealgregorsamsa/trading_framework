"""Build 1-second bars from the clean layer and save them as Parquet."""

from pathlib import Path

import duckdb

BARS_SQL = Path(__file__).parent / "sql" / "bars.sql"
BARS_PATH = Path("data/bars/bars.parquet")


def build_bars(sql_path: Path = BARS_SQL, out_path: Path = BARS_PATH) -> None:
    """Aggregate clean quotes into 1-second bars and write them as one Parquet file.

    The aggregation is defined in a SQL file and executed by DuckDB, which
    reads the partitioned clean layer directly.

    Args:
        sql_path: SQL file producing the bars.
        out_path: destination Parquet file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("SET TimeZone = 'UTC'")
    query = sql_path.read_text()
    con.sql(f"COPY ({query}) TO '{out_path}' (FORMAT parquet)")
    print(f"Bars written to {out_path}")


if __name__ == "__main__":
    build_bars()
