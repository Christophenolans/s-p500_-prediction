# Walk-forward Backtest: PCA + Matrix Regression

## What this adds

This module continues the existing correlation/data project with an out-of-sample backtest:

1. Read daily SPY/S&P 500 OHLCV data.
2. Read available macro indicator CSVs from `data/` and align them by date.
3. Approximate macro publication availability with explicit fixed lags.
4. Build features and the next-session return target.
5. Fit standardization and PCA on training data only.
6. Compare Gaussian Elimination, Gauss-Jordan, LU, Adjugate inverse, Least Squares, and an SVD/pseudoinverse-equivalent solver.
7. Evaluate out-of-sample MSE, MAE, Direction Accuracy, and correlation.
8. Save CSV outputs under `result/backtest/`.

## Input data

Preferred input is `data/spy_data.csv`. The loader also accepts `data/SPY.csv`, `data/spy.csv`, and the current archive's `data/SPX500_1d.csv`. If none exists, it attempts to download the configured GitHub raw CSV URL. Network access may be unavailable; in that case manually put `spy_data.csv` in `data/`.

Macro CSVs recognized in `data/`: `CPI.csv`, `DGS10.csv`, `DTWEXBGS.csv`, `FEDFUNDS.csv`, `GOLD.csv`, `NFP.csv`, `OIL.csv`, `PCEPI.csv`, `PPI.csv`, `REALGDP.csv`, `UNEMPLOYRATE.csv`, and `VIX.csv`. Missing files are skipped. The two credit-spread files are intentionally excluded because they start much later and would unnecessarily shorten the common sample.

## Backtest design

- When enough history is available, use seven calendar years for training and the next calendar year for testing, then roll forward one year.
- When history is too short, use a clearly labeled fallback of 252 training rows and 63 testing rows so the pipeline can be checked. Do not report fallback results as a 7-year/1-year backtest.
- The training labels are filtered so a target return crossing into the test period is not used to train the model.
- PCA and standardization are refit inside each round using training rows only.

## Matrix methods

All six methods solve the same regularized normal-equation system for fair comparison. The `SVD` path uses an eigensystem of `X^T X` to compute a truncated pseudoinverse-equivalent solution using the standard library. This is educational and is less numerically robust than a full bidiagonal SVD implementation. `Least Squares` uses the normal equations and Gaussian elimination; it is not a distinct numerical algorithm from Gaussian in this implementation.

The core matrix algorithms use Python lists and the standard library only. No NumPy, pandas, SciPy, or scikit-learn is required.

## Outputs

- `result/backtest/predictions.csv`
- `result/backtest/metrics_by_round.csv`
- `result/backtest/method_comparison.csv`
- `result/backtest/pca_summary.csv`
- `result/backtest/errors.csv`
- `result/backtest/README_RESULTS.txt`

## Run

```bash
python main.py
```

This is historical educational analysis, not financial advice. Backtest performance does not guarantee future results. Fixed macro release lags are approximations; formal research should use actual release dates and data vintages.
