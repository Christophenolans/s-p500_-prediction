import os
from src.correlation import correlation

if __name__ == "__main__":
    # 1. กำหนดพาทไฟล์เข้า
    SPY_DATA_PATH = "result/standardization_data/spy_data.csv"
    INDICATOR_DATA_PATH = "result/standardization_data/REALGDP.csv"
    
    # 2. กำหนดพาทไฟล์ออก
    OUTPUT_RESULT_PATH = "result/correlation_std_spy_ind/correlation_spy_REALGDP.csv"

    # สร้างโฟลเดอร์ปลายทางทั้งหมดให้อัตโนมัติก่อนบันทึก
    os.makedirs(os.path.dirname(OUTPUT_RESULT_PATH), exist_ok=True)

    # 3. รันการคำนวณ
    results_df = correlation(
        spy_path=SPY_DATA_PATH,
        indicator_path=INDICATOR_DATA_PATH,
        window_years=7,
        max_lag=14
    )

    # 4. บันทึกผลลัพธ์ลง CSV
    results_df.to_csv(OUTPUT_RESULT_PATH, index=False)
    print(f" บันทึกผลลัพธ์เรียบร้อยที่: {OUTPUT_RESULT_PATH}")