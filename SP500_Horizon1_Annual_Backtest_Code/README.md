# S&P 500 Horizon-1 Annual Backtest

This script automatically downloads **all files** tracked under `data/results 1 horizon 1/` from the repository `Christophenolans/s-p500_-prediction`, branch `guide`, before running the backtest. This includes every available window folder and its files, not only one selected CSV. It needs internet access when run.

## Run
Place `backtest_horizon1.py` in the repository root (or run it from any folder) and execute:

```bash
python backtest_horizon1.py
```

By default, downloaded files are stored in `data/results 1 horizon 1/` and refreshed from GitHub on each run. To use existing local files without downloading:

```bash
python backtest_horizon1.py --no-download
```

To choose a different local data folder:

```bash
python backtest_horizon1.py --data-dir "data/results 1 horizon 1"
```

Outputs under `result/horizon1_trade_backtest/`:
- `daily_backtest.csv`: daily scores, training-bin expected returns, signals, and equity.
- `annual_summary.csv`: one row per train/test year with strategy and buy-and-hold results.
- `skipped_windows.csv`: missing input windows.

## Method
A seven-year window (e.g. 2000–2006) calibrates mean return per 0.1-wide Score bin. The next year (2007) is tested using the 2007 rows from the adjacent 2001–2007 file. Bins need at least 20 training examples. Positive historical mean return means long; otherwise cash. The simulation is long-only and deducts 5 basis points per position change by default.

Returns are recomputed as `(Future_Open / Open - 1) * 100`, rather than trusting the field called `Return_14_Trading_Days_Percent`. Verify `Future_Date` represents the next trading session.

**Important limitation:** the precomputed score in the next overlapping window may have been calculated using weights fitted on a window that includes the test year. This can cause look-ahead bias even though the return calibration comes from the previous training window. A strict out-of-sample test requires recomputing test-year scores with only the prior training window's weights. Historical educational simulation only, not investment advice.
