"""Long-only, all-in/all-out portfolio simulation with explicit cash, shares and costs."""
from __future__ import annotations
import math
from datetime import date


def max_drawdown(equity_values):
    peak = None
    worst = 0.0
    for value in equity_values:
        peak = value if peak is None else max(peak, value)
        if peak and peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst * 100.0


def sharpe_ratio(daily_returns_pct):
    if len(daily_returns_pct) < 2:
        return 0.0
    mean = sum(daily_returns_pct) / len(daily_returns_pct)
    var = sum((x-mean)**2 for x in daily_returns_pct) / (len(daily_returns_pct)-1)
    sd = math.sqrt(var)
    return (mean / sd) * math.sqrt(252) if sd > 1e-12 else 0.0


def simulate_portfolio(predictions, initial_capital=10000.0, cost_bps=5.0):
    """Trade a simulated SPY-like asset using fractional shares and transaction costs.

    Signals are acted on at each row's Open. The position is marked at Future_Open.
    Fees are charged on gross traded value at each entry/exit. Cash earns 0%.
    This is not connected to a broker and places no actual orders.
    """
    ordered = sorted(predictions, key=lambda r: (r["date"], r["train_window"]))
    cash = float(initial_capital)
    shares = 0.0
    fee_rate = max(0.0, float(cost_bps)) / 10000.0
    benchmark_shares = 0.0
    benchmark_initialized = False
    daily = []
    daily_returns = []
    trades = 0
    for p in ordered:
        open_price = float(p["open_price"])
        future_open = float(p["future_open_price"])
        if open_price <= 0 or future_open <= 0:
            continue
        desired = int(p["position"])
        equity_before = cash + shares * open_price
        fee = 0.0
        if desired == 1 and shares <= 1e-12:
            fee = cash * fee_rate
            investable = max(0.0, cash - fee)
            shares = investable / open_price
            cash = 0.0
            trade_event = "BUY"
            trades += 1
        elif desired == 0 and shares > 1e-12:
            gross = shares * open_price
            fee = gross * fee_rate
            cash = max(0.0, gross - fee)
            shares = 0.0
            trade_event = "SELL_TO_CASH"
            trades += 1
        elif desired == 1:
            trade_event = "HOLD"
        else:
            trade_event = "CASH"

        if not benchmark_initialized:
            benchmark_fee = initial_capital * fee_rate
            benchmark_shares = max(0.0, initial_capital - benchmark_fee) / open_price
            benchmark_initialized = True
        equity_after = cash + shares * future_open
        benchmark_equity = benchmark_shares * future_open
        daily_return = (equity_after / equity_before - 1.0) * 100.0 if equity_before > 0 else 0.0
        daily_returns.append(daily_return)
        daily.append({
            **p,
            "initial_capital_usd": initial_capital,
            "equity_before_trade_usd": equity_before,
            "trade_cost_usd": fee,
            "cash_usd_after_trade_and_mark": cash,
            "shares_held_after_trade": shares,
            "portfolio_equity_usd": equity_after,
            "buy_hold_equity_usd": benchmark_equity,
            "daily_portfolio_return_pct": daily_return,
            "trade_event": trade_event,
        })
        # At the next row, its Open should equal this row's Future_Open; shares/cash carry forward.
    if daily:
        start = date.fromisoformat(daily[0]["date"])
        end = date.fromisoformat(daily[-1]["target_date"] or daily[-1]["date"])
        years = max(1, (end - start).days) / 365.25
        final_equity = float(daily[-1]["portfolio_equity_usd"])
        benchmark_mark = float(daily[-1]["buy_hold_equity_usd"])
        benchmark_final = benchmark_mark * (1.0 - fee_rate)  # hypothetical liquidation fee
        strategy_return = (final_equity / initial_capital - 1.0) * 100.0
        benchmark_return = (benchmark_final / initial_capital - 1.0) * 100.0
        summary = {
            "start_date": daily[0]["date"], "end_date": daily[-1]["target_date"] or daily[-1]["date"],
            "trading_rows": len(daily), "initial_capital_usd": initial_capital,
            "final_equity_usd": final_equity, "strategy_total_return_pct": strategy_return,
            "buy_hold_final_equity_usd_after_estimated_exit_fee": benchmark_final,
            "buy_hold_total_return_pct": benchmark_return,
            "strategy_CAGR_pct": ((final_equity/initial_capital)**(1/years)-1)*100 if final_equity > 0 else -100.0,
            "buy_hold_CAGR_pct": ((benchmark_final/initial_capital)**(1/years)-1)*100 if benchmark_final > 0 else -100.0,
            "strategy_max_drawdown_pct": max_drawdown([float(r["portfolio_equity_usd"]) for r in daily]),
            "buy_hold_max_drawdown_pct": max_drawdown([float(r["buy_hold_equity_usd"]) for r in daily]),
            "strategy_annualized_sharpe": sharpe_ratio(daily_returns),
            "position_changes": trades,
            "profitable_days_pct": 100.0 * sum(float(r["daily_portfolio_return_pct"]) > 0 for r in daily) / len(daily),
            "assumptions": f"Long-only, cash return 0%, fractional shares, {cost_bps:g} bps per entry/exit; benchmark includes estimated entry/exit fees; excludes taxes, dividends, and extra slippage. Simulation only.",
        }
    else:
        summary = {"start_date":"", "end_date":"", "trading_rows":0, "initial_capital_usd":initial_capital, "final_equity_usd":initial_capital, "strategy_total_return_pct":0, "buy_hold_final_equity_usd_after_estimated_exit_fee":initial_capital, "buy_hold_total_return_pct":0, "strategy_CAGR_pct":0, "buy_hold_CAGR_pct":0, "strategy_max_drawdown_pct":0, "buy_hold_max_drawdown_pct":0, "strategy_annualized_sharpe":0, "position_changes":0, "profitable_days_pct":0, "assumptions":"No predictions available."}
    return daily, summary


def annual_summary(daily, initial_capital):
    years = sorted({int(row["test_year"]) for row in daily})
    result = []
    for year in years:
        rows = [r for r in daily if int(r["test_year"]) == year]
        if not rows:
            continue
        start = float(rows[0]["equity_before_trade_usd"])
        end = float(rows[-1]["portfolio_equity_usd"])
        bh_start = float(rows[0]["buy_hold_equity_usd"])
        bh_end = float(rows[-1]["buy_hold_equity_usd"])
        result.append({
            "year": year, "rows": len(rows), "starting_equity_usd": start,
            "ending_equity_usd": end, "strategy_return_pct": (end/start-1)*100 if start else 0,
            "buy_hold_return_pct": (bh_end/bh_start-1)*100 if bh_start else 0,
            "position_changes": sum(1 for r in rows if r["trade_event"] in {"BUY", "SELL_TO_CASH"}),
            "direction_accuracy_pct": 100.0*sum(int(r["direction_correct"]) for r in rows)/len(rows),
            "mean_predicted_return_pct": sum(float(r["predicted_return_pct"]) for r in rows)/len(rows),
            "mean_actual_return_pct": sum(float(r["actual_return_pct"]) for r in rows)/len(rows),
        })
    return result
