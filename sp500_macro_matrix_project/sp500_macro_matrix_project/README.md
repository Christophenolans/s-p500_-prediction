# S&P 500 Macro Matrix Project

วิเคราะห์ผลตอบแทน S&P 500 ด้วย Macro indicators: Interest Rate, CPI, PPI, NFP, GDP, VIX, 10Y Treasury และ Labor Productivity พร้อม PCA, SVD และ Matrix Regression.

Target = ผลตอบแทน S&P 500 วันถัดไป
Adaptive backtest = train 7 ปี -> test 1 ปี -> เลื่อนหน้าต่างไป 1 ปี

รัน:
`pip install -r requirements.txt`
`python main.py`

ผลลัพธ์อยู่ใน `outputs/`

หมายเหตุ: การ align macro แบบพื้นฐานในโปรเจกต์นี้ใช้ค่าล่าสุดที่มีในแต่ละวัน (forward fill) เพื่อให้โครงงานรันได้ง่าย สำหรับงานวิจัยฉบับสมบูรณ์ควรใช้ release-date/vintage data เพื่อป้องกัน look-ahead bias และ data revisions.
