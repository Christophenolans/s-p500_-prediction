import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_and_save(file_path: str):
    # 1. ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
    file_path = os.path.abspath(file_path.strip('"').strip("'"))
    if not os.path.exists(file_path):
        print(f"❌ Error: ไม่พบไฟล์ตาม path ที่ระบุ: {file_path}")
        return

    # ดึงโฟลเดอร์ปลายทางจาก path ของไฟล์ input
    output_dir = os.path.dirname(file_path)
    print(f"\n📂 กำลังประเมินผลไฟล์: {file_path}")
    print(f"📁 บันทึกไฟล์ผลลัพธ์ไว้ที่: {output_dir}\n")

    # 2. อ่านข้อมูลและเตรียมข้อมูล
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # คำนวณ Strategy Return (position * actual_return_pct)
    df['strategy_return_pct'] = df['position'] * df['actual_return_pct']

    r_strat = df['strategy_return_pct'] / 100.0
    r_bench = df['actual_return_pct'] / 100.0

    cum_strat = (1 + r_strat).cumprod()
    cum_bench = (1 + r_bench).cumprod()

    trading_days = len(df)
    years = trading_days / 252.0

    # --- 3. คำนวณภาพรวม (Overall Metrics) ---
    total_strat_ret = (cum_strat.iloc[-1] - 1) * 100
    total_bench_ret = (cum_bench.iloc[-1] - 1) * 100

    cagr_strat = ((cum_strat.iloc[-1]) ** (1 / years) - 1) * 100
    cagr_bench = ((cum_bench.iloc[-1]) ** (1 / years) - 1) * 100

    vol_strat = r_strat.std() * np.sqrt(252) * 100
    vol_bench = r_bench.std() * np.sqrt(252) * 100

    sharpe_strat = (r_strat.mean() * 252) / (r_strat.std() * np.sqrt(252)) if r_strat.std() != 0 else 0
    sharpe_bench = (r_bench.mean() * 252) / (r_bench.std() * np.sqrt(252)) if r_bench.std() != 0 else 0

    peak_strat = cum_strat.cummax()
    dd_strat = (cum_strat - peak_strat) / peak_strat
    max_dd_strat = dd_strat.min() * 100

    peak_bench = cum_bench.cummax()
    dd_bench = (cum_bench - peak_bench) / peak_bench
    max_dd_bench = dd_bench.min() * 100

    dir_acc = df['direction_correct'].mean() * 100
    mae = np.mean(np.abs(df['predicted_return_pct'] - df['actual_return_pct']))
    rmse = np.sqrt(np.mean((df['predicted_return_pct'] - df['actual_return_pct']) ** 2))

    active_trades = df[df['position'] != 0]
    win_rate = (active_trades['actual_return_pct'] > 0).mean() * 100 if len(active_trades) > 0 else 0
    exposure = (len(active_trades) / trading_days) * 100

    summary_df = pd.DataFrame({
        "Metric": [
            "Total Trading Days",
            "Period (Years)",
            "Directional Accuracy (%)",
            "MAE (%)",
            "RMSE (%)",
            "Market Exposure (%)",
            "Total Return (%)",
            "CAGR (%)",
            "Annualized Volatility (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Win Rate (Active Trades %)"
        ],
        "Strategy": [
            trading_days,
            f"{years:.2f}",
            f"{dir_acc:.2f}%",
            f"{mae:.4f}",
            f"{rmse:.4f}",
            f"{exposure:.2f}%",
            f"{total_strat_ret:.2f}%",
            f"{cagr_strat:.2f}%",
            f"{vol_strat:.2f}%",
            f"{sharpe_strat:.2f}",
            f"{max_dd_strat:.2f}%",
            f"{win_rate:.2f}%"
        ],
        "Buy & Hold (Benchmark)": [
            trading_days,
            f"{years:.2f}",
            "-",
            "-",
            "-",
            "100.00%",
            f"{total_bench_ret:.2f}%",
            f"{cagr_bench:.2f}%",
            f"{vol_bench:.2f}%",
            f"{sharpe_bench:.2f}",
            f"{max_dd_bench:.2f}%",
            f"{(r_bench > 0).mean() * 100:.2f}%"
        ]
    })

    # --- 4. คำนวณผลการดำเนินงานรายปี (Yearly Breakdown) ---
    yearly = []
    for year, group in df.groupby('test_year'):
        g_strat = group['strategy_return_pct'] / 100.0
        g_bench = group['actual_return_pct'] / 100.0

        ret_s = ((1 + g_strat).prod() - 1) * 100
        ret_b = ((1 + g_bench).prod() - 1) * 100
        acc = group['direction_correct'].mean() * 100
        trades = (group['position'] != 0).sum()

        yearly.append({
            "Year": year,
            "Directional Acc (%)": round(acc, 2),
            "Strategy Return (%)": round(ret_s, 2),
            "Benchmark Return (%)": round(ret_b, 2),
            "Excess Return (%)": round(ret_s - ret_b, 2),
            "Active Days": trades
        })
    yearly_df = pd.DataFrame(yearly)

    # --- 5. บันทึกไฟล์ตารางไปยังโฟลเดอร์เดียวกัน ---
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    yearly_path = os.path.join(output_dir, "evaluation_yearly.csv")
    chart_path = os.path.join(output_dir, "evaluation_chart.png")

    summary_df.to_csv(summary_path, index=False)
    yearly_df.to_csv(yearly_path, index=False)

    # --- 6. สร้างและบันทึกภาพกราฟ ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(df['date'], cum_strat, label='Strategy (Linear Model)', color='blue', linewidth=1.5)
    ax1.plot(df['date'], cum_bench, label='Buy & Hold (S&P 500)', color='gray', linestyle='--', alpha=0.7)
    ax1.set_title('Backtest Performance Evaluation', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Growth of $1')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(df['date'], dd_strat * 100, label='Strategy Drawdown', color='red', linewidth=1)
    ax2.plot(df['date'], dd_bench * 100, label='Benchmark Drawdown', color='orange', linestyle=':', alpha=0.6)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # --- แสดงผลหน้าจอ ---
    print("================ OVERALL EVALUATION ================")
    print(summary_df.to_string(index=False))
    print("\n================ YEARLY BREAKDOWN ================")
    print(yearly_df.to_string(index=False))

    print("\n✅ บันทึกไฟล์ผลลัพธ์ลงโฟลเดอร์เรียบร้อยแล้ว:")
    print(f"  1. {summary_path}")
    print(f"  2. {yearly_path}")
    print(f"  3. {chart_path}")

# --- วิธีการเรียกใช้งาน ---
if __name__ == "__main__":
    # สามารถใส่ path ไฟล์ตรงนี้ หรือ รับจาก Command Line Argument ได้
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("กรุณาใส่ Path ของไฟล์ predictions.csv: ").strip()

    evaluate_and_save(input_file)