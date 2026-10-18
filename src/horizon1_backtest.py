"""Walk-forward prediction and capital-based backtest using Horizon-1 CSVs."""
from __future__ import annotations
import argparse, os
from datetime import date
from .horizon1_data import read_csv, load_weights, parse_date, write_csv
from .horizon1_model import build_labeled_rows, fit_calibration, predict_rows
from .portfolio import simulate_portfolio, annual_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "results 3 horizon 14")
DEFAULT_WEIGHT_FILE = os.path.join(ROOT, "data", "weighted_correlation_lag3.csv")
DEFAULT_OUT_DIR = os.path.join(ROOT, "result", "horizon314_backtest")


def run_backtest(data_dir=DEFAULT_DATA_DIR, weight_file=DEFAULT_WEIGHT_FILE, out_dir=DEFAULT_OUT_DIR,
                 start_test_year=2007, end_test_year=2025, initial_capital=10000.0,
                 cost_bps=5.0, min_bin_count=10, threshold_pct=0.0):
    if not os.path.isfile(weight_file):
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    weights_by_window = load_weights(weight_file)
    all_predictions = []
    skipped = []
    for year in range(start_test_year, end_test_year + 1):
        train_start, train_end = year - 7, year - 1
        train_window = f"{train_start}-{train_end}"
        test_window = f"{train_start+1}-{train_end+1}"
        train_path = os.path.join(data_dir, train_window, "spy_daily_score.csv")
        test_path = os.path.join(data_dir, test_window, "spy_daily_score.csv")
        train_weights = weights_by_window.get(train_window)
        if not os.path.isfile(train_path):
            skipped.append({"test_year": year, "train_window": train_window, "reason": f"Missing training file: {train_path}"})
            print(f"SKIP {year}: training file missing")
            continue
        if not os.path.isfile(test_path):
            skipped.append({"test_year": year, "train_window": train_window, "reason": f"Missing test file: {test_path}"})
            print(f"SKIP {year}: test file missing")
            continue
        if not train_weights or len(train_weights) < 5:
            skipped.append({"test_year": year, "train_window": train_window, "reason": "Training-window weights missing/incomplete"})
            print(f"SKIP {year}: weights missing")
            continue
        train_raw = read_csv(train_path)
        test_raw = read_csv(test_path)
        train_labeled = build_labeled_rows(train_raw, train_weights, target_before=date(year, 1, 1))
        test_raw = [r for r in test_raw if (parse_date(r.get("Date")) and parse_date(r.get("Date")).year == year)]
        if len(train_labeled) < 30:
            skipped.append({"test_year": year, "train_window": train_window, "reason": f"Only {len(train_labeled)} valid training rows"})
            print(f"SKIP {year}: only {len(train_labeled)} valid training rows")
            continue
        model = fit_calibration(train_labeled, min_bin_count=max(2, min_bin_count))
        preds = predict_rows(test_raw, train_weights, model, train_window, year, threshold_pct)
        if not preds:
            skipped.append({"test_year": year, "train_window": train_window, "reason": "No valid test rows after feature/price checks"})
            print(f"SKIP {year}: no valid test rows")
            continue
        all_predictions.extend(preds)
        print(f"Train {train_window} -> Test {year}: train={len(train_labeled)} predictions={len(preds)}")

    daily, overall = simulate_portfolio(all_predictions, initial_capital, cost_bps)
    annual = annual_summary(daily, initial_capital)
    os.makedirs(out_dir, exist_ok=True)
    prediction_fields = ["train_window", "test_year", "date", "target_date", "open_price", "future_open_price", "score", "predicted_return_pct", "predicted_probability_up_pct", "calibration_samples", "prediction_method", "signal", "position", "actual_return_pct", "direction_correct"]
    write_csv(os.path.join(out_dir, "predictions.csv"), all_predictions, prediction_fields)
    # Keep the conventional project-level prediction path in sync with the walk-forward run.
    project_result_dir = os.path.join(ROOT, "result")
    os.makedirs(project_result_dir, exist_ok=True)
    write_csv(os.path.join(project_result_dir, "predictions.csv"), all_predictions, prediction_fields)
    daily_fields = prediction_fields + ["initial_capital_usd", "equity_before_trade_usd", "trade_cost_usd", "portfolio_equity_usd", "buy_hold_equity_usd", "fractional_shares_after_return_estimate", "daily_portfolio_return_pct", "drawdown_pct", "trade_event"]
    write_csv(os.path.join(out_dir, "daily_backtest.csv"), daily, daily_fields)
    write_csv(os.path.join(out_dir, "annual_summary.csv"), annual, ["year", "rows", "starting_equity_usd", "ending_equity_usd", "strategy_return_pct", "buy_hold_return_pct", "position_changes", "direction_accuracy_pct", "mean_predicted_return_pct", "mean_actual_return_pct"])
    write_csv(os.path.join(out_dir, "skipped_years.csv"), skipped, ["test_year", "train_window", "reason"])
    write_csv(os.path.join(out_dir, "overall_summary.csv"), [overall], list(overall.keys()))
    with open(os.path.join(out_dir, "README_RESULTS.txt"), "w", encoding="utf-8") as f:
        f.write("S&P 500 Horizon-1 Walk-Forward Prediction + Capital Backtest\n")
        f.write("===========================================================\n")
        f.write(f"Test years requested: {start_test_year}-{end_test_year}\n")
        f.write(f"Prediction rows: {len(all_predictions)}\n")
        f.write(f"Initial capital: ${initial_capital:,.2f}\nTransaction cost: {cost_bps:g} bps per position change\n")
        f.write("Training weights are loaded from the matching prior 7-year window; test-year weights are not used.\n")
        f.write("The normalized indicator columns in each provided score CSV may themselves have been normalized over a window containing the test year; treat results as provisional until normalization is rebuilt using training-only parameters.\n")
        f.write("This is a historical simulation, not a live broker order. No actual transactions were placed.\n")
        f.write("Portfolio assumptions: long-only, fractional shares, cash return 0%, explicit transaction cost; excludes taxes, dividends, and additional slippage.\n\n")
        for k,v in overall.items(): f.write(f"{k}: {v}\n")
        if skipped:
            f.write(f"\nSkipped years: {len(skipped)}; see skipped_years.csv\n")
    print("\nBacktest complete")
    print(f"Prediction rows: {len(all_predictions)}")
    print(f"Final equity: ${overall.get('final_equity_usd', initial_capital):,.2f}")
    print(f"Strategy total return: {overall.get('strategy_total_return_pct', 0):.2f}%")
    print(f"Buy-and-hold total return: {overall.get('buy_hold_total_return_pct', 0):.2f}%")
    print(f"Outputs: {out_dir}")
    return overall, annual, all_predictions


def main():
    parser = argparse.ArgumentParser(description="Horizon-1 S&P 500 walk-forward prediction and capital simulation")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--weight-file", default=DEFAULT_WEIGHT_FILE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-year", type=int, default=2007)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--min-bin-count", type=int, default=10)
    parser.add_argument("--threshold-pct", type=float, default=0.0)
    args = parser.parse_args()
    run_backtest(args.data_dir, args.weight_file, args.out_dir, args.start_year, args.end_year,
                 args.capital, args.cost_bps, args.min_bin_count, args.threshold_pct)

if __name__ == "__main__":
    main()
