# Legacy preprocessing scripts

These are archived preprocessing experiments moved out of `data/` so source code is kept under `src/`.
They are **not called by the current Horizon-1 prediction/backtest pipeline**. The original files contain
absolute Windows paths from the development machine and must be adapted to this project's relative paths
before running them. The current backtest consumes the supplied CSV datasets and `data/weighted_correlation_lag1.csv`.
