# hffx: a research framework for short-horizon signals on tick data

`hffx` takes raw FX quotes to a cost-aware, leakage-safe backtest. It is an
**offline research framework**, not a live trading system (no execution layer,
no real-time feed, no risk limits).

## Background

The project started as a course study of one question: *can a machine
learning model predict the next-minute move of EURUSD from tick data well
enough to trade it?* I built the whole chain for that study and evaluated it
carefully: walk-forward validation by session, spread costs, and a held-out
period left untouched.

The study answered the question: at a typical top-of-book spread of about
1.4 bp, a 60-second directional signal does not cover its trading costs.
The part worth keeping is the infrastructure. I refactored the exploration
scripts into a tested package so that any new signal idea can be evaluated the
same honest way: same data layers, same validation, same cost model.

```
raw files ──► clean ──► 1 s bars ──► features + labels ──► models ──► backtest
 (CSV)      (Parquet)   (Parquet)       (Parquet)       walk-forward   with costs
```

## 1. Data Pipeline

`src/hffx/ingest.py`, `bars.py`, `features.py`, `dataset.py`

This module is the base for everything else: it turns raw files into a
model-ready dataset.

Main tasks

* Validate raw files: fail fast with a clear message on empty or corrupted files.
* Unify sources: tick CSVs and level 2 files are brought to one schema (`ts`, `bid`, `ask`).
* Clean: sort, drop duplicates, remove non-positive prices, crossed quotes and abnormal spreads.
* Segment sessions: a gap above 5 minutes between two quotes starts a new session.
* Store: Parquet partitioned by symbol and date, with idempotent writes.
* Build bars: 1-second bars aggregated with DuckDB SQL (close = mid at the last quote of the second).
* Fill gaps: forward-fill only, inside each session, never interpolation.
* Generate features: log returns (1, 10, 60, 300 s), volatility (60, 300 s), 5-minute z-score, quote activity, spread in bps, hour of day.
* Build labels: 60-second forward log return against a threshold, giving 1, -1 or 0. The last seconds of a session have no known future and stay NaN.

## 2. Feature Evaluation

`src/hffx/diagnostics.py`, `scripts/run_diagnostics.py`

This module tells whether a feature carries information, before any model is trained.

Main tasks

* Information coefficient: Spearman correlation between each feature and the target, computed per session.
* Significance across sessions: mean IC, t-statistic and share of sessions with a positive IC.
* Direction versus magnitude: the same analysis on the sign of the move and on its absolute size.
* Permutation importance: drop in log loss when a feature is shuffled, measured out of sample in each walk-forward round.
* Leakage demonstration: accuracy under a shuffled split versus walk-forward.

There is deliberately no automatic feature selection (RFE, sequential
selection, importance thresholds). Neighbouring seconds are highly
correlated, so such procedures fit the noise and leak information. Features
are judged one by one by their per-session IC.

## 3. Model Validation

`src/hffx/validation.py`, `models.py`, `scripts/run_baselines.py`

This module compares models under a validation scheme that cannot leak.

Main tasks

* Walk-forward by session: expanding window, whole sessions, at least 3 training sessions; the model never sees the session it is tested on.
* Baseline: a classifier that always predicts the most frequent class, the number to beat when most labels are neutral.
* Models: standardised logistic regression and histogram gradient boosting, with fixed hyperparameters (tuning a model before the features contain a signal would only fit noise).
* Metrics: accuracy per round, confusion matrix, precision and recall per class.

## 4. Backtest

`src/hffx/backtest.py`

This module judges a model by what it would have earned, net of cost.

Main tasks

* Positions: the model's prediction, or only the most confident predictions (cutoff computed on the training set).
* Costs: one full top-of-book spread per round trip, charged whenever a position is taken.
* PnL: gross, cost and net, in basis points.
* Performance: per-session Sharpe ratio, maximum drawdown, number of trades and hit rate.
* Regimes: average edge per trade by volatility tercile.

## Getting Started

```bash
pip install -e ".[dev]"
```

Place raw files as `data/raw/<SYMBOL>_<SOURCE>/*.csv`, for example
`data/raw/EURUSD_L2/` or `data/raw/XAUUSD_Ticks/`. Data files are not part of
this repository.

Run the pipeline step by step:

```bash
python -m hffx.ingest
python -m hffx.bars
python -m hffx.features
python scripts/run_baselines.py
python scripts/run_diagnostics.py
```

Run the tests:

```bash
pytest
```

## Evaluating your own feature

1. Add a function in `src/hffx/features.py` that takes the bars and returns
   them with a new column, computed from past data only, inside each symbol
   and session.
2. Call it in the pipeline at the bottom of `features.py` and add the column
   name to `FEATURE_COLUMNS` in `src/hffx/config.py`.
3. Check it against look-ahead with `tests/test_features.py`: the test builds
   features on 700 seconds and on the first 500, and requires the first 500
   rows to be identical.
4. Evaluate it with `session_ic`, or pass `feature_columns` to the evaluation
   and backtest functions.

## Tests

Unit tests cover the walk-forward splits, the X/y construction, the absence
of look-ahead in the features, forward-fill, label construction, and
hand-computed PnL, costs and performance metrics.

## Findings

On EURUSD, a 60-second directional signal built from price, volatility and
activity features does not cover trading costs at the observed spread, while
volatility is predictable. Detailed numbers are available on request.

## Limitations

* Offline and batch only: no execution, no real-time data, no risk limits.
* Quotes only: order-book volume columns are empty in the data I had, so volume and order-flow features are not possible.
* One instrument modelled (EURUSD). XAUUSD ticks are ingested but not modelled.
* About 144 hours of data: enough to reject a signal at 60 seconds, too little to estimate effects at horizons of an hour or more.
* Costs are the top-of-book spread with small size; no market impact.
* Adding a feature means editing `features.py` and `config.py`; there is no plug-in registry yet.

## Possible extensions

* Volatility-based strategies, since volatility is the predictable part.
* Longer horizons, with more data.
* Lower-cost execution assumptions (passive orders, a fill model).
* Range-based volatility estimators and a feature registry.
