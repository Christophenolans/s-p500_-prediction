import os
import pandas as pd
from src.features import prepare_features_and_target
from src.regression import train_ridge_regression, get_model_coefficients

BEST_LAGS = {
    'CPI': 1,
    'OIL': 1,
    'DGS10': 14,
    'FEDFUNDS': 14,
    'REALGDP': 1,
    'NFP': 1,
    'PCEPI': 1,
    'PPI': 1
}

if __name__ == "__main__":
    SPY_PATH = "result/standardization_data/spy_data.csv"
    DATA_DIR = "result/standardization_data"
    OUTPUT_PATH = "result/rolling_window_coefficients.csv"

    os.makedirs("result", exist_ok=True)
    all_window_coefs = []

    print("=== เริ่มต้นประมวลผล Rolling Window (7 ปี) ===")
    for start_yr in range(2000, 2020):
        end_yr = start_yr + 6
        
        try:
            window_data = prepare_features_and_target(
                spy_path=SPY_PATH,
                data_dir=DATA_DIR,
                best_lags=BEST_LAGS,
                start_year=start_yr,
                end_year=end_yr,
                test_size=0.2
            )
            
            window_model = train_ridge_regression(window_data['X_train'], window_data['y_train'], alpha=1.0)
            coef_df = get_model_coefficients(window_model, window_data['feature_names'])
            coef_df['Window'] = f"{start_yr}-{end_yr}"
            all_window_coefs.append(coef_df)
            
            print(f" ประมวลผล Window {start_yr}-{end_yr} สำเร็จ")
        except Exception as e:
            print(f" Skipping window {start_yr}-{end_yr}: {e}")

    if all_window_coefs:
        final_coefs_df = pd.concat(all_window_coefs, ignore_index=True)
        final_coefs_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n บันทึกผลลัพธ์ Rolling Window เรียบร้อยที่: {OUTPUT_PATH}")