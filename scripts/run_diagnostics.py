"""Information coefficient of every feature, on the development sessions only."""

from hffx.config import FEATURE_COLUMNS
from hffx.dataset import load_dataset
from hffx.diagnostics import session_ic


def main() -> None:
    df = load_dataset("EURUSD")
    sessions = sorted(df["session_id"].unique())
    dev = df[df["session_id"].isin(sessions[:8])]  # IC computed on the first 8 sessions only

    print(session_ic(dev, FEATURE_COLUMNS))
    dev = dev.assign(abs_future_return=dev["future_return_60s"].abs())
    print(session_ic(dev, FEATURE_COLUMNS, target="abs_future_return"))
    print("std of 60s return (bps):", round(dev["future_return_60s"].std() * 1e4, 2))


if __name__ == "__main__":
    main()
