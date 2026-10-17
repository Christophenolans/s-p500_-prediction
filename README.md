# S&P 500 Horizon-1 Prediction and Portfolio Backtest

This project generates next-session return predictions from the supplied Horizon-1 macro-indicator score data, then runs a historical portfolio simulation from **2007 through 2025**.

## Source organization

```text
.
├── main.py                       # entry point only
├── src/
│   ├── horizon1_data.py           # CSV loading, dates, weights, scores, realized returns
│   ├── horizon1_model.py         # training calibration and prediction generation
│   ├── portfolio.py              # capital, shares, cash, costs, equity and drawdown simulation
│   ├── horizon1_backtest.py      # walk-forward orchestration and output writing
│   ├── matrix_backtest.py        # optional older matrix-regression research pipeline
│   ├── matrix.py / eigen.py / features.py / stats.py / pipeline.py
│   └── legacy/                   # archived preprocessing scripts, not called by default
├── data/                         # CSV data and Horizon-1 source windows
└── result/horizon1_backtest/     # generated prediction and backtest outputs
```

## Run

Python 3.10+; current Horizon-1 workflow uses only the Python standard library.

```bash
python main.py --start-year 2007 --end-year 2025 --capital 10000 --cost-bps 5
```

Options:

- `--start-year`: first test year (default 2007)
- `--end-year`: last test year (default 2025; the provided data ends in August 2025)
- `--capital`: simulated starting capital in USD (default 10,000)
- `--cost-bps`: transaction cost per entry/exit, in basis points (default 5)
- `--min-bin-count`: minimum training observations for a score-bin estimate (default 10)
- `--threshold-pct`: minimum expected return required to enter/hold the asset (default 0)

`run_backtest.bat` runs the same default period on Windows.

## Outputs

All outputs are written to `result/horizon1_backtest/`:

- `predictions.csv`: one prediction per test date, expected return, probability-up estimate, method, signal, and actual next-session return
- `daily_backtest.csv`: simulated portfolio equity, cash, shares, fees, trade events, and buy-and-hold comparison
- `annual_summary.csv`: annual return, direction accuracy, and trade counts
- `overall_summary.csv`: whole-period return, CAGR, maximum drawdown, Sharpe estimate, and assumptions
- `skipped_years.csv`: years skipped because required data was missing
- `README_RESULTS.txt`: run summary and limitations

## Walk-forward design

For test year Y, the model uses the preceding seven-year window Y-7 through Y-1, and tests on year Y. Training labels whose `Future_Date` reaches the test year are excluded. Training-window weights are used to calculate test scores; weights from the test window are not used.

The investment simulation is long-only and all-in/all-out, allows fractional shares, assumes cash earns 0%, and charges a configurable fee on each entry/exit. The benchmark is buy-and-hold with estimated entry and exit fees. It excludes taxes, dividends, market impact, and any additional slippage.

## Important research limitation

The provided normalized indicator columns may have been normalized independently within each seven-year output window, which can use information from the test year when scaling features. The code does not refit that normalization from raw indicators using training-only parameters. Therefore, these results are a **provisional historical simulation, not a fully leakage-free out-of-sample result**. Rebuild normalization using only the training window before relying on the result for research. The code does not connect to a broker and does not place actual trades.

## Generated output files

This source package intentionally does **not** include pre-generated `result/` files. When you run `main.py`, the code creates the output directory and writes fresh results to `result/horizon1_backtest/`; it also writes the main historical prediction file to `result/predictions.csv`.

Generated files include:

- `predictions.csv`
- `daily_backtest.csv`
- `annual_summary.csv`
- `overall_summary.csv`
- `skipped_years.csv`
- `README_RESULTS.txt`

These files are created by the program at runtime and are not bundled in this ZIP.

**Data-folder check:** inspect `data/` before rerunning. This archive may contain `results 1 horizon 14` folders rather than the required `results 1 horizon 1` folders. Do not rename Horizon-14 data to Horizon-1; the correct Horizon-1 CSVs must be supplied for a valid Horizon-1 rerun.
