import os
from src.correlation import correlation
from src.features import prepare_features_and_target
from src.features import prepare_features_and_target
from src.regression import train_ridge_regression, get_model_coefficients
from src.stats import evaluate_model

BEST_LAGS = {
    'CPI': 1,          # ไฟล์ CPI.csv (คอลัมน์ CPIAUCSL)
    'OIL': 1,          # ไฟล์ OIL.csv (คอลัมน์ DCOILWTICO)
    'DGS10': 14,       # ไฟล์ DGS10.csv
    'FEDFUNDS': 14,    # ไฟล์ FEDFUNDS.csv
    'REALGDP': 1,      # ไฟล์ REALGDP.csv (คอลัมน์ GDPC1)
    'NFP': 1,          # ไฟล์ NFP.csv (คอลัมน์ PAYEMS)
    'PCEPI': 1,        # ไฟล์ PCEPI.csv
    'PPI': 1           # ไฟล์ PPI.csv (คอลัมน์ PPIACO)
}

if __name__ == "__main__":
    #SPY_DATA_PATH = "result/standardization_data/spy_data.csv"
    #INDICATOR_DATA_PATH = "result/standardization_data/REALGDP.csv"
    
    #OUTPUT_RESULT_PATH = "result/correlation_std_spy_ind/correlation_spy_REALGDP.csv"

    #os.makedirs(os.path.dirname(OUTPUT_RESULT_PATH), exist_ok=True)

    #results_df = correlation(
    #    spy_path=SPY_DATA_PATH,
    #    indicator_path=INDICATOR_DATA_PATH,
    #    window_years=7,
    #    max_lag=14
    #)

    #results_df.to_csv(OUTPUT_RESULT_PATH, index=False)
    #print(f" บันทึกผลลัพธ์เรียบร้อยที่: {OUTPUT_RESULT_PATH}")

    SPY_PATH = "result/standardization_data/spy_data.csv"
    DATA_DIR = "result/standardization_data"
    
    # 2. ส่ง BEST_LAGS เข้าฟังก์ชัน
    data = prepare_features_and_target(
        spy_path=SPY_PATH,
        data_dir=DATA_DIR,
        best_lags=BEST_LAGS,
        test_size=0.2
    )
    
    #print("X_train shape:", data['X_train'].shape)
    #print("X_test shape:", data['X_test'].shape)

    model = train_ridge_regression(data['X_train'], data['y_train'], alpha=1.0)

    # 3. พยากรณ์ค่าบน Train Set และ Test Set
    y_train_pred = model.predict(data['X_train'])
    y_test_pred = model.predict(data['X_test'])

    # 4. ประเมินผลประสิทธิภาพ
    train_metrics = evaluate_model(data['y_train'], y_train_pred)
    test_metrics = evaluate_model(data['y_test'], y_test_pred)

    print("\n=== Model Performance ===")
    print(f"Train R^2: {train_metrics['R2']:.4f} | RMSE: {train_metrics['RMSE']:.4f}")
    print(f"Test  R^2: {test_metrics['R2']:.4f} | RMSE: {test_metrics['RMSE']:.4f}")

    # 5. แสดงค่าน้ำหนัก Coefficients ของแต่ละ Feature
    coef_df = get_model_coefficients(model, data['feature_names'])
    print("\n=== Feature Coefficients ===")
    print(coef_df.to_string(index=False))