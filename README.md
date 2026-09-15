# S&P 500 Matrix Decision Model

Educational matrix-mathematics project using historical S&P 500 OHLCV data.

## Important
The included `data/sp500_sample.csv` is a small runnable sample so the project works immediately. For the actual report, replace it with the full historical CSV from a public source such as the GitHub dataset cited in the project documentation.

## Core mathematics implemented manually
- Gaussian Elimination
- Gauss-Jordan Matrix Inverse
- Least Squares
- Covariance Matrix
- Power Iteration for Eigenvalue/Eigenvector
- Eigenvalue Deflation
- PCA-related principal directions
- Standardization

No NumPy, SciPy, pandas, scikit-learn, or ML library is used for the core calculations.

## Run
```bash
python main.py
```

Outputs:
- `result/predictions.csv`
- `result/metrics.txt`

## Feature matrix
1. Daily return
2. 5-day momentum
3. Intraday return
4. Daily range percentage
5. Log volume

Target:
Next-day return.

Training/testing is chronological (70/30), not random, to avoid using future observations to train the model.

This is an educational backtest/simulation, not a recommendation to trade real money.
