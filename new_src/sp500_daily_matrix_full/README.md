# S&P 500 Daily Matrix Analysis - Standalone

Run:
`py -3.13 -m pip install -r requirements.txt`
`py -3.13 main.py`

Pipeline: S&P 500 Daily OHLCV + FRED PPI -> cleaning/alignment -> 20 daily indicators -> correlation -> standardization -> PCA -> SVD -> Gaussian/Gauss-Jordan/LU/Adjugate regression -> 7-year rolling adaptive backtest -> next-day return prediction -> reports and plots.

The project is standalone and does not depend on the previous project.

This is an educational historical modeling project, not financial advice.
