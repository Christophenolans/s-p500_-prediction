# Horizon-1 prediction and capital-based backtest

Run `python main.py --start-year 2007 --end-year 2025 --capital 10000 --cost-bps 5`.

The active workflow is implemented under `src/`. It walks forward one calendar year at a time, fits score-to-return calibration using the previous seven-year window, produces next-session predictions, and simulates a long-only portfolio with cash, fractional shares, transaction costs, and a buy-and-hold benchmark. Results are written to `result/horizon1_backtest/`.

The simulation is educational research only. It is not connected to a broker and does not place real orders. Current source indicator values are pre-normalized in the supplied data; if their scaling used the test window, the results may contain look-ahead bias. See `README.md` and `result/horizon1_backtest/README_RESULTS.txt` for assumptions and limitations.
