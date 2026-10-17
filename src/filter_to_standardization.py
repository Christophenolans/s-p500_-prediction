import os
import pandas as pd

output_dir = "result/standardization_data"
os.makedirs(output_dir, exist_ok=True)

data_folder = "data" 
start_date = "2000-01-01"

csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

for file_name in sorted(csv_files):
    file_path = os.path.join(data_folder, file_name)
    
    if 'spy' in file_name.lower():
        df = pd.read_csv(
            file_path, 
            skiprows=3, 
            header=None, 
            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        )
        df['Date'] = pd.to_datetime(df['Date'])
        filtered_df = df[df['Date'] >= start_date].sort_values('Date')
        
        save_path = os.path.join(output_dir, 'spy_data.csv')
        filtered_df.to_csv(save_path, index=False)
        print(f"บันทึก {save_path} เรียบร้อย ({len(filtered_df)} แถว)")
        
    else:
        df = pd.read_csv(file_path)
        date_col = [c for c in df.columns if 'date' in c.lower()][0]
        val_col = [c for c in df.columns if c != date_col][0]
        
        df[date_col] = pd.to_datetime(df[date_col])
        filtered_df = df[df[date_col] >= start_date].sort_values(date_col).copy()
        
        filtered_df[val_col] = pd.to_numeric(filtered_df[val_col], errors='coerce')
        filtered_df[val_col] = filtered_df[val_col].ffill().bfill()
        
        clean_file_name = file_name.replace(" (1)", "")
        save_path = os.path.join(output_dir, clean_file_name)
        
        filtered_df.to_csv(save_path, index=False)
        print(f"บันทึก {save_path} เรียบร้อย (จำนวนค่าว่างคงเหลือ: {filtered_df[val_col].isna().sum()})")

print("\n คลีนค่าว่างและจัดเก็บข้อมูลใน 'standardization_data' สำเร็จเรียบร้อย 100%!")