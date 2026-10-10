# OOD-REsult — PCA, Matrix Regression and Walk-forward Backtest using `spy_data.csv`

## Price data source
The project uses **`data/spy_data.csv`** from:
https://github.com/Frikzxyz/OOD-REsult/blob/main/data/spy_data.csv

If the file is not present locally, `main.py` attempts to download it from the repository's raw CSV URL and save it into `data/spy_data.csv`. If that fails, download the file manually and place it in `data/`.

The loader handles the file's multi-row metadata header (`Price,Close,...`, followed by ticker and date metadata rows) as well as a normalized CSV with `date` and `close` columns. SPY is an ETF that tracks the S&P 500, so this run uses SPY prices rather than the index's own quoted level.

## Run
```bash
python -m pip install -r requirements.txt
python main.py
```

## Backtest design
- Training window: **7 × 252 ≈ 1,764 trading-day rows**.
- Test window: **252 trading-day rows** (approximately one year).
- Walk-forward rounds: after each test block, the training window moves forward by one test block and the model is refit.
- Target: next-day close-to-close return `(Close[t+1] - Close[t]) / Close[t]`.
- The scaler and PCA basis are fitted on each training block only, then applied unchanged to its test block.
- Methods compared: Gaussian Elimination, Gauss-Jordan, LU, Adjugate Inverse, Least Squares and SVD.
- Metrics: MSE, MAE, directional accuracy and correlation.

## Macro indicators and no-look-ahead caution
Other CSVs in `data/` are read as indicator series and aligned to SPY trading dates. The code delays low-frequency data by one observation as a conservative approximation. This is **not** a full point-in-time release-calendar implementation; for research-grade results, use actual release dates/vintages. Existing `result/rolling_corr_result_*.csv` files are preserved as correlation-analysis outputs and are not used as raw training observations.

## Outputs
Written to `result/backtest/`: `predictions.csv`, `metrics_by_round.csv`, `method_comparison.csv`, `summary.txt`, and `prediction_*.png`.

A meaningful 7-year train + 1-year test run requires enough complete overlapping dates after macro-feature alignment. If too few vectors remain, check missing values and date coverage rather than shortening the window silently.

## Matrix-method caveat
Gaussian Elimination, Gauss-Jordan, LU and Adjugate are implemented directly in the script. NumPy is used for PCA eigen-decomposition and for the Least Squares/SVD solver paths. Replace those routines with manually implemented algorithms if the course requires all core matrix calculations to be coded from scratch.
