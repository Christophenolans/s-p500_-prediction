# S&P 500 Matrix Decision Model — Backtest Extension

This project contains the original data/correlation files and an added walk-forward backtest pipeline.

## Pipeline

`Market + macro CSVs → feature engineering → train-only standardization → PCA → matrix regression methods → out-of-sample prediction → evaluation`

Methods compared:
- Gaussian Elimination
- Gauss-Jordan
- LU Decomposition
- Adjugate Matrix Inverse
- Least Squares
- SVD/pseudoinverse-equivalent solver

Metrics:
- MSE
- MAE
- Direction Accuracy
- Prediction correlation

## Run

```bash
python main.py
```

The script prefers `data/spy_data.csv`. It also accepts `data/SPY.csv`, `data/spy.csv`, or `data/SPX500_1d.csv`. If none is present, it attempts to download the configured GitHub raw CSV. If internet access is unavailable, put the requested CSV at `data/spy_data.csv` manually.

The provided archive does **not** currently include `data/spy_data.csv`; the included local `data/SPX500_1d.csv` covers 2024-05-06 through 2026-09-11. Therefore, the checked-in output currently demonstrates the clearly labeled short-data fallback (252 training rows / 63 testing rows), not the requested 7-year-training/1-year-testing study. To run the full rolling annual backtest, provide sufficiently long historical data.

## Results

Files are written to `result/backtest/`:
- `predictions.csv`
- `metrics_by_round.csv`
- `method_comparison.csv`
- `pca_summary.csv`
- `errors.csv`
- `README_RESULTS.txt`

## Methodological notes

- The target is next-session close-to-close return; each feature row is dated at the close used to form its features.
- Training rows whose target date crosses into the test period are excluded from that round's training set.
- Standardization and PCA are fitted on each training set only.
- Macro data is aligned using fixed publication-lag approximations; rigorous work should use actual release calendars and vintage data to avoid revision/look-ahead leakage.
- The matrix methods use hand-written Python-list operations, without NumPy/pandas/SciPy/scikit-learn. Least Squares is implemented via normal equations solved by Gaussian elimination, so its numerical path is not independent of Gaussian. The SVD path uses the eigensystem of `XᵀX` as a truncated pseudoinverse-equivalent educational implementation, not a full bidiagonal SVD.
- Results are historical educational evaluation only, not investment advice or a guarantee of future performance.

See [README_BACKTEST.md](README_BACKTEST.md) for more detail.
