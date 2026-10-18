
from __future__ import annotations
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_col(df: pd.DataFrame, *names: str) -> str | None:
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for name in names:
        key = name.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


def _num(df: pd.DataFrame, *names: str, default=np.nan) -> pd.Series:
    col = _find_col(df, *names)
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_pct_return(equity: pd.Series, initial: float) -> pd.Series:
    prev = equity.shift(1)
    if len(prev):
        prev.iloc[0] = initial
    out = (equity / prev - 1.0) * 100.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _cagr(equity: pd.Series, years: float, initial: float) -> float:
    if equity.empty or years <= 0 or initial <= 0 or not np.isfinite(equity.iloc[-1]) or equity.iloc[-1] < 0:
        return np.nan
    if equity.iloc[-1] == 0:
        return -100.0
    return ((equity.iloc[-1] / initial) ** (1.0 / years) - 1.0) * 100.0


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax().replace(0, np.nan)
    return ((equity / peak - 1.0).min()) * 100.0


def _sharpe(return_pct: pd.Series) -> float:
    r = pd.to_numeric(return_pct, errors="coerce").dropna() / 100.0
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def _score_barriers(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-based score bands for descriptive analysis, not signal generation."""
    score_col = _find_col(df, "score", "Score", "barrier_score")
    if score_col is None:
        return pd.DataFrame(columns=["Barrier", "Score Min", "Score Max", "Rows", "Directional Accuracy (%)", "Mean Predicted Return (%)", "Mean Actual Return (%)", "Positive Actual Return (%)", "Strategy Return (%)"])
    scores = pd.to_numeric(df[score_col], errors="coerce")
    valid = scores.notna()
    if valid.sum() == 0:
        return pd.DataFrame()
    # Rank-based qcut handles ties better than fixed-width bins for score ranges
    # that differ between rolling training windows. Duplicated edges are dropped.
    try:
        bands = pd.qcut(scores[valid].rank(method="first"), q=min(5, int(valid.sum())), labels=False, duplicates="drop")
    except ValueError:
        bands = pd.Series(0, index=scores[valid].index)
    tmp = df.loc[valid].copy()
    tmp["_score"] = scores[valid]
    tmp["_band"] = bands.astype(int)
    actual = _num(tmp, "actual_return_pct", "Actual_Return_Pct")
    pred = _num(tmp, "predicted_return_pct", "Predicted_Return_Pct")
    correct_col = _find_col(tmp, "direction_correct", "Direction_Correct")
    position = _num(tmp, "position", default=0).fillna(0)
    rows = []
    n_bands = int(tmp["_band"].nunique())
    for b, g in tmp.groupby("_band", sort=True):
        idx = g.index
        a = actual.loc[idx]
        p = pred.loc[idx]
        if correct_col:
            acc = pd.to_numeric(g[correct_col], errors="coerce").mean() * 100
        else:
            acc = (((p >= 0) == (a >= 0)) & p.notna() & a.notna()).mean() * 100
        gross = (position.loc[idx] * a / 100.0).fillna(0.0)
        strat_ret = ((1 + gross).prod() - 1) * 100
        rows.append({
            "Barrier": f"B{int(b)+1}",
            "Score Min": float(g["_score"].min()),
            "Score Max": float(g["_score"].max()),
            "Rows": int(len(g)),
            "Directional Accuracy (%)": round(float(acc), 4) if pd.notna(acc) else np.nan,
            "Mean Predicted Return (%)": round(float(p.mean()), 6) if p.notna().any() else np.nan,
            "Mean Actual Return (%)": round(float(a.mean()), 6) if a.notna().any() else np.nan,
            "Positive Actual Return (%)": round(float((a > 0).mean() * 100), 4) if a.notna().any() else np.nan,
            "Strategy Return (%)": round(float(strat_ret), 4),
        })
    return pd.DataFrame(rows)


def evaluate_and_save(file_path: str, initial_capital: float | None = None, cost_bps: float = 5.0):
    file_path = os.path.abspath(file_path.strip().strip('"').strip("'"))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์: {file_path}")
    out_dir = os.path.dirname(file_path)
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("ไฟล์ CSV ไม่มีข้อมูล")

    date_col = _find_col(df, "date", "Date", "timestamp")
    if date_col is None:
        raise ValueError("ไม่พบคอลัมน์ date/Date")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].sort_values(date_col).reset_index(drop=True)
    if df.empty:
        raise ValueError("ไม่มีแถวที่มีวันที่ถูกต้อง")

    position = _num(df, "position", "Position", default=0).fillna(0).clip(0, 1)
    actual = _num(df, "actual_return_pct", "Actual_Return_Pct")
    predicted = _num(df, "predicted_return_pct", "Predicted_Return_Pct")
    score = _num(df, "score", "Score", "barrier_score")
    correct_col = _find_col(df, "direction_correct", "Direction_Correct")
    if correct_col:
        direction_correct = pd.to_numeric(df[correct_col], errors="coerce")
    else:
        direction_correct = ((predicted >= 0) == (actual >= 0)).where(predicted.notna() & actual.notna())

    initial_col = _num(df, "initial_capital_usd", "initial_capital", default=np.nan)
    if initial_capital is None:
        initial_capital = float(initial_col.dropna().iloc[0]) if initial_col.notna().any() else 10000.0
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial capital ต้องเป็นค่ามากกว่า 0")

    equity_col = _find_col(df, "portfolio_equity_usd", "equity", "portfolio_equity")
    benchmark_equity_col = _find_col(df, "buy_hold_equity_usd", "benchmark_equity_usd")
    daily_ret_col = _find_col(df, "daily_portfolio_return_pct")
    if equity_col:
        strategy_equity = pd.to_numeric(df[equity_col], errors="coerce")
        # If simulator equity is present, use it: includes cash, fractional shares and fees.
        strategy_equity = strategy_equity.ffill()
        strategy_returns = _safe_pct_return(strategy_equity, initial_capital)
        if daily_ret_col:
            simulator_returns = _num(df, daily_ret_col)
            strategy_returns = simulator_returns.where(simulator_returns.notna(), strategy_returns)
    else:
        # Legacy prediction files have no capital simulation; this is gross return,
        # without transaction costs. State this in the report.
        strategy_returns = (position * actual).fillna(0.0)
        strategy_equity = initial_capital * (1.0 + strategy_returns / 100.0).cumprod()

    if benchmark_equity_col:
        benchmark_equity = pd.to_numeric(df[benchmark_equity_col], errors="coerce").ffill()
        benchmark_returns = _safe_pct_return(benchmark_equity, initial_capital)
    else:
        benchmark_returns = actual.fillna(0.0)
        benchmark_equity = initial_capital * (1.0 + benchmark_returns / 100.0).cumprod()

    days = len(df)
    start_date, end_date = df[date_col].iloc[0], df[date_col].iloc[-1]
    years = max((end_date - start_date).days / 365.25, days / 252.0, 1 / 252.0)
    strategy_equity_clean = strategy_equity.replace([np.inf, -np.inf], np.nan).dropna()
    benchmark_equity_clean = benchmark_equity.replace([np.inf, -np.inf], np.nan).dropna()
    final_strategy = float(strategy_equity_clean.iloc[-1]) if len(strategy_equity_clean) else initial_capital
    final_benchmark = float(benchmark_equity_clean.iloc[-1]) if len(benchmark_equity_clean) else initial_capital
    strat_total = (final_strategy / initial_capital - 1) * 100
    # Apply estimated liquidation fee to Buy & Hold at the end, matching the
    # simulator README convention. The daily benchmark curve remains pre-liquidation.
    exit_fee_rate = max(0.0, float(cost_bps)) / 10000.0
    benchmark_final_after_exit = final_benchmark * (1.0 - exit_fee_rate)
    bench_total = (benchmark_final_after_exit / initial_capital - 1) * 100
    active = position > 0
    active_actual = actual[active]
    direction_accuracy = direction_correct.mean() * 100 if direction_correct.notna().any() else np.nan
    mae = (predicted - actual).abs().mean()
    rmse = np.sqrt(((predicted - actual) ** 2).mean())
    exposure = active.mean() * 100
    # A win means an active holding period's realized return is positive.
    win_rate = (active_actual > 0).mean() * 100 if active_actual.notna().any() else np.nan
    summary = pd.DataFrame([
        {"Metric": "Total Trading Rows", "Strategy": days, "Buy & Hold (Benchmark)": days},
        {"Metric": "Period (Years)", "Strategy": round(years, 4), "Buy & Hold (Benchmark)": round(years, 4)},
        {"Metric": "Directional Accuracy (%)", "Strategy": round(direction_accuracy, 4) if pd.notna(direction_accuracy) else np.nan, "Buy & Hold (Benchmark)": "-"},
        {"Metric": "MAE (percentage points)", "Strategy": round(float(mae), 6) if pd.notna(mae) else np.nan, "Buy & Hold (Benchmark)": "-"},
        {"Metric": "RMSE (percentage points)", "Strategy": round(float(rmse), 6) if pd.notna(rmse) else np.nan, "Buy & Hold (Benchmark)": "-"},
        {"Metric": "Market Exposure (%)", "Strategy": round(float(exposure), 4), "Buy & Hold (Benchmark)": 100.0},
        {"Metric": "Total Return (%)", "Strategy": round(strat_total, 4), "Buy & Hold (Benchmark)": round(bench_total, 4)},
        {"Metric": "CAGR (%)", "Strategy": round(_cagr(strategy_equity_clean, years, initial_capital), 4), "Buy & Hold (Benchmark)": round(_cagr(pd.Series([benchmark_final_after_exit]), years, initial_capital), 4)},
        {"Metric": "Annualized Volatility (%)", "Strategy": round(float(strategy_returns.std(ddof=1) * np.sqrt(252)), 4), "Buy & Hold (Benchmark)": round(float(benchmark_returns.std(ddof=1) * np.sqrt(252)), 4)},
        {"Metric": "Sharpe Ratio", "Strategy": round(_sharpe(strategy_returns), 4), "Buy & Hold (Benchmark)": round(_sharpe(benchmark_returns), 4)},
        {"Metric": "Max Drawdown (%)", "Strategy": round(_max_drawdown(strategy_equity_clean), 4), "Buy & Hold (Benchmark)": round(_max_drawdown(benchmark_equity_clean), 4)},
        {"Metric": "Win Rate (Active Holding Periods, %)", "Strategy": round(float(win_rate), 4) if pd.notna(win_rate) else np.nan, "Buy & Hold (Benchmark)": round(float((actual > 0).mean() * 100), 4)},
    ])
    summary_path = os.path.join(out_dir, "evaluation_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Yearly results: calculate compound returns from daily portfolio returns, preserving costs.
    year_values = df[date_col].dt.year
    yearly_rows = []
    for year, indexes in df.groupby(year_values).groups.items():
        ix = list(indexes)
        sr = strategy_returns.loc[ix].dropna()
        br = benchmark_returns.loc[ix].dropna()
        a = actual.loc[ix]
        c = direction_correct.loc[ix]
        yearly_rows.append({
            "Year": int(year), "Rows": len(ix),
            "Directional Accuracy (%)": round(float(c.mean() * 100), 4) if c.notna().any() else np.nan,
            "Strategy Return (%)": round(float(((1 + sr / 100).prod() - 1) * 100), 4),
            "Benchmark Return (%)": round(float(((1 + br / 100).prod() - 1) * 100), 4),
            "Excess Return (percentage points)": round(float(((1 + sr / 100).prod() - (1 + br / 100).prod()) * 100), 4),
            "Active Rows": int((position.loc[ix] > 0).sum()),
            "Mean Score": round(float(score.loc[ix].mean()), 6) if score.notna().any() else np.nan,
            "Mean Predicted Return (%)": round(float(predicted.loc[ix].mean()), 6) if predicted.loc[ix].notna().any() else np.nan,
            "Mean Actual Return (%)": round(float(a.mean()), 6) if a.notna().any() else np.nan,
        })
    yearly = pd.DataFrame(yearly_rows)
    yearly_path = os.path.join(out_dir, "evaluation_yearly.csv")
    yearly.to_csv(yearly_path, index=False)

    barriers = _score_barriers(df)
    barrier_path = os.path.join(out_dir, "evaluation_barrier_score.csv")
    barriers.to_csv(barrier_path, index=False)

    # Equity/drawdown chart; values are in USD and use simulator outputs where available.
    strat_dd = strategy_equity / strategy_equity.cummax().replace(0, np.nan) - 1
    bench_dd = benchmark_equity / benchmark_equity.cummax().replace(0, np.nan) - 1
    chart_path = os.path.join(out_dir, "evaluation_chart.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(df[date_col], strategy_equity, label="Strategy Equity", linewidth=1.4)
    ax1.plot(df[date_col], benchmark_equity, label="Buy & Hold Benchmark", linestyle="--", alpha=0.8)
    ax1.set_title("Backtest Evaluation (using simulated equity when available)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax2.plot(df[date_col], strat_dd * 100, label="Strategy Drawdown", linewidth=1)
    ax2.plot(df[date_col], bench_dd * 100, label="Benchmark Drawdown", linestyle=":", alpha=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)

    print("\n================ OVERALL EVALUATION ================")
    print(summary.to_string(index=False))
    print("\n================ YEARLY BREAKDOWN ================")
    print(yearly.to_string(index=False))
    print("\n================ SCORE BARRIERS (DESCRIPTIVE) ================")
    print(barriers.to_string(index=False) if not barriers.empty else "ไม่มีคอลัมน์ score สำหรับวิเคราะห์ barrier")
    print("\nSaved:")
    for p in [summary_path, yearly_path, barrier_path, chart_path]:
        print(f"  {p}")
    if not equity_col:
        warnings.warn("Input has no portfolio_equity_usd: strategy results are gross estimates and do not include trading costs.")
    return summary, yearly, barriers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a walk-forward backtest CSV and calculate score barriers")
    parser.add_argument("file", nargs="?", help="Path to daily_backtest.csv or predictions.csv")
    parser.add_argument("--capital", type=float, default=None, help="Initial capital in USD (default: from CSV or 10000)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Estimated Buy & Hold liquidation fee in basis points (default: 5)")
    args = parser.parse_args()
    input_file = args.file or input("กรุณาใส่ Path ของไฟล์ daily_backtest.csv หรือ predictions.csv: ").strip()
    try:
        evaluate_and_save(input_file, args.capital, args.cost_bps)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
