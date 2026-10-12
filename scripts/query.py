"""Run a SQL file with DuckDB and print the result.

Usage: python scripts/query.py sql/look.sql
"""

import sys
from pathlib import Path

import duckdb

con = duckdb.connect()
con.sql("SET TimeZone = 'UTC'")
con.sql(Path(sys.argv[1]).read_text()).show(max_width=250)
