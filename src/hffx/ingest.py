"""Ingestion of raw tick files: raw layer -> clean layer (partitioned Parquet)."""

import time
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
COLUMN_MAPPING = {
    "Gmt time": "ts",
    "Ask": "ask",
    "Bid": "bid",
    "AskVolume": "ask_volume",
    "BidVolume": "bid_volume",
}
SESSION_GAP = pd.Timedelta(minutes=5)
MAX_SPREAD_RATIO = 10
SPREAD_WINDOW = 1000
SPREAD_MIN_PERIODS = 100
KNOWN_SYMBOLS = {"XAUUSD", "EURUSD"}
LINES_PER_QUOTE = 10
SIDE_NAMES = {
    "B": "bid",
    "S": "ask",
}
SIZE_NAMES = {
    100_000.0: "100k",
    300_000.0: "300k",
    1_000_000.0: "1m",
    3_000_000.0: "3m",
    5_000_000.0: "5m",
}
KNOWN_SOURCES = {"Ticks", "L2"}


def parse_source(csv_path: Path) -> str:
    """Extract the data source from the name of the file's folder.

    Args:
        csv_path: path of a raw file, stored as data/raw/<SYMBOL>_<SOURCE>/.

    Returns:
        The source, e.g. 'Ticks' or 'L2'.

    Raises:
        ValueError: if the source is not in KNOWN_SOURCES.
    """
    source = csv_path.parent.name.split("_")[1]
    if source not in KNOWN_SOURCES:
        raise ValueError(f"Unknown source '{source}' in folder {csv_path.parent}")
    return source


def load_source(csv_paths: list[Path], source: str) -> pd.DataFrame:
    """Read raw files of one source and bring them to the common schema.

    Each source has its own reader; the output always has 'ts', 'bid'
    and 'ask', so that the rest of the pipeline is shared.

    Args:
        csv_paths: raw files, all from the same source.
        source: 'Ticks' (Dukascopy) or 'L2' (EURUSD level 2).

    Returns:
        One DataFrame with all files, in the common schema.
    """
    if source == "Ticks":
        df = pd.concat([load_raw_ticks(p) for p in csv_paths], ignore_index=True)
        return standardize_columns(df)
    if source == "L2":
        frames = [prepare_eurusd_file(p) for p in csv_paths]
        return pd.concat(frames, ignore_index=True)
    raise ValueError(f"No reader for source '{source}'")


def load_raw_ticks(csv_path: Path) -> pd.DataFrame:
    """Read a raw tick CSV file without modifying it.

    Args:
        csv_path: path of the raw CSV file.

    Returns:
        DataFrame with the ticks exactly as stored in the file.
    """
    return pd.read_csv(csv_path)


def load_raw_eurusd(csv_path: Path) -> pd.DataFrame:
    """Read a raw EURUSD level 2 file, keeping only the useful columns.

    The file has no header and uses ';' as separator. Each quote is spread
    over 10 lines (5 sizes x bid/ask), linked by the first part of quote_id.

    Args:
        csv_path: path of the raw EURUSD file.

    Returns:
        DataFrame with columns quote_id, ts, size, price and side.
    """
    return pd.read_csv(csv_path, sep=";", header=None,
                       usecols=[1, 4, 5, 8, 9],
                       names=["quote_id", "ts", "size", "price", "side"])


def standardize_eurusd(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the common part of quote_id and parse timestamps (epoch ms) as UTC."""
    return df.assign(quote_id=df["quote_id"].str.split("-").str[0],
                     ts=pd.to_datetime(df["ts"], unit="ms", utc=True))


def drop_incomplete_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop quotes that do not have all their LINES_PER_QUOTE lines."""
    mask = df.groupby("quote_id")["price"].transform("size") == LINES_PER_QUOTE
    print(f"Removed lines: {(~mask).sum()}")
    return df[mask].reset_index(drop=True)


def pivot_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the 10 lines of each quote into one row, with one column per side and size.

    The timestamp of a quote is the latest timestamp of its lines. The 100k
    prices become the top of book (bid, ask).
    """
    df = df.assign(level=df["side"].map(SIDE_NAMES) + "_" + df["size"].map(SIZE_NAMES),
                   ts=df.groupby("quote_id")["ts"].transform("max"))
    df = df.pivot(index=["quote_id", "ts"], columns="level", values="price")
    df = df.rename(columns={"bid_100k": "bid", "ask_100k": "ask"})
    df.columns.name = None
    return df.reset_index()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Dukascopy columns, parse timestamps as UTC and cast prices and volumes to float."""
    df = df.rename(columns=COLUMN_MAPPING)
    df["ts"] = pd.to_datetime(df["ts"], format="%d.%m.%Y %H:%M:%S.%f", utc=True)
    return df.astype({"ask": "float64", "bid": "float64",
                      "ask_volume": "float64", "bid_volume": "float64"})


def sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Sort ticks in chronological order and drop duplicates."""
    n_before = len(df)
    df = df.sort_values("ts", kind="stable")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    print(f"Deleted duplicates: {n_before - len(df)}")
    return df


def filter_price_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible or suspicious quotes.

    Removes non-positive prices, crossed quotes (bid >= ask) and spreads larger
    than MAX_SPREAD_RATIO times the rolling median of past spreads (shifted by
    one row, so a quote is never compared with a window that contains itself).
    """
    mask = (df["bid"] > 0) & (df["ask"] > 0)
    print(f"Negative or null prices: {(~mask).sum()}")
    df = df[mask]

    mask = df["bid"] < df["ask"]
    df = df[mask]

    spread = df["ask"] - df["bid"]
    median = spread.rolling(SPREAD_WINDOW, min_periods=SPREAD_MIN_PERIODS).median().shift(1)
    mask = (spread <= MAX_SPREAD_RATIO * median) | median.isna()
    print(f"Spread > {MAX_SPREAD_RATIO}x median: {(~mask).sum()}")
    df = df[mask]

    return df.reset_index(drop=True)


def add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a session number to each tick.

    A new session starts whenever the time gap between two consecutive
    ticks is greater than SESSION_GAP (5 minutes). This threshold is far
    above normal gaps (under one minute) and far below market breaks
    (the 1-hour daily pause, weekends), so both breaks and feed outages
    are detected.

    Args:
        df: ticks sorted by 'ts'.

    Returns:
        The same ticks with a 'session_id' column, starting at 0.
    """
    gaps = df["ts"].diff()
    is_new_session = gaps > SESSION_GAP
    df = df.assign(session_id=is_new_session.cumsum())
    print(f"number of sessions: {df['session_id'].nunique()}")
    return df


def parse_symbol(csv_path: Path) -> str:
    """Extract the symbol from the name of the file's folder (data/raw/<SYMBOL>_<SOURCE>/).

    Raises:
        ValueError: if the symbol is not in KNOWN_SYMBOLS.
    """
    symbol = csv_path.parent.name.split("_")[0].upper()
    if symbol not in KNOWN_SYMBOLS:
        raise ValueError(f"Unknown symbol '{symbol}' in file name {csv_path.name}")
    return symbol


def write_clean_parquet(df: pd.DataFrame, symbol: str) -> None:
    """Write the clean ticks as Parquet partitioned by symbol and date.

    Existing partitions are replaced, so re-running the pipeline is idempotent.
    """
    df = df.assign(symbol=symbol, date=df["ts"].dt.date)
    print(f"Writing {len(df)} ticks to {CLEAN_DIR}...")
    start = time.perf_counter()
    df.to_parquet(CLEAN_DIR, partition_cols=["symbol", "date"],
                  existing_data_behavior="delete_matching")
    print(f"Done in {time.perf_counter() - start:.1f} s")


def validate_raw_file(csv_path: Path, separator: str) -> None:
    """Fail fast with a clear message if a raw file is empty or corrupted.

    Only the first kilobyte is read, so the check is instant.

    Args:
        csv_path: path of the raw file.
        separator: the column separator expected for this source.

    Raises:
        ValueError: if the file is empty, made of null bytes, or has no separator.
    """
    with open(csv_path, "rb") as f:
        head = f.read(1024)
    if not head.strip(b"\0"):
        raise ValueError(f"{csv_path.name} is empty or only contains null bytes")
    if separator.encode() not in head:
        raise ValueError(f"{csv_path.name} has no '{separator}' separator in its first line")


def prepare_eurusd_file(csv_path: Path) -> pd.DataFrame:
    """Read one EURUSD level 2 file and return one row per complete quote.

    Files are pivoted one at a time, before being concatenated, so that
    memory never holds more than one raw file (10x larger than pivoted).

    Args:
        csv_path: path of one raw EURUSD file.

    Returns:
        The pivoted quotes of this file, in the common schema.
    """
    validate_raw_file(csv_path, ";")
    print(f"Preparing {csv_path.name}...")
    df = load_raw_eurusd(csv_path)
    df = standardize_eurusd(df)
    df = drop_incomplete_quotes(df)
    return pivot_quotes(df)


def main(csv_paths: list[Path]) -> None:
    """Run the raw -> clean pipeline on the files of one raw folder.

    Args:
        csv_paths: raw files of a single symbol and source.

    Raises:
        ValueError: if the files mix several symbols or sources.
    """
    symbols = {parse_symbol(p) for p in csv_paths}
    sources = {parse_source(p) for p in csv_paths}
    if len(symbols) != 1 or len(sources) != 1:
        raise ValueError(f"Mixed inputs: symbols={symbols}, sources={sources}")
    symbol = symbols.pop()
    source = sources.pop()

    df = load_source(csv_paths, source)
    df = sort_and_deduplicate(df)
    df = filter_price_anomalies(df)
    df = add_session_id(df)
    write_clean_parquet(df, symbol)


if __name__ == "__main__":
    for folder in sorted(RAW_DIR.iterdir()):
        if folder.is_dir():
            csv_paths = sorted(folder.glob("*.csv"))
            print(f"\n=== {folder.name}: {len(csv_paths)} files ===")
            main(csv_paths)
