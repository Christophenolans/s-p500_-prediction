import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_features_and_target(
    spy_path: str,
    data_dir: str,
    best_lags: dict,
    start_year: int = None,
    end_year: int = None,
    test_size: float = 0.2
):
    spy_df = pd.read_csv(spy_path)
    spy_df.columns = [c.lower() for c in spy_df.columns]
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df = spy_df.sort_values('date').set_index('date')
    
    target_col = 'close' if 'close' in spy_df.columns else 'open'
    df_combined = pd.DataFrame(index=spy_df.index)
    df_combined['target'] = pd.to_numeric(spy_df[target_col], errors='coerce')

    for ind_name, lag in best_lags.items():
        file_path = os.path.join(data_dir, f"{ind_name}.csv")
        if not os.path.exists(file_path):
            continue
            
        ind_df = pd.read_csv(file_path)
        date_col = ind_df.columns[0]
        val_col = ind_df.columns[1]
        
        ind_df[date_col] = pd.to_datetime(ind_df[date_col])
        ind_df = ind_df.sort_values(date_col).set_index(date_col)
        
        ind_series = pd.to_numeric(ind_df[val_col], errors='coerce').reindex(df_combined.index, method='ffill')
        df_combined[ind_name] = ind_series.shift(lag)

    #  กรองเฉพาะกรอบปีที่ต้องการ (เช่น 2000-2006)
    if start_year is not None and end_year is not None:
        df_combined = df_combined.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

    df_combined = df_combined.bfill().dropna()

    X = df_combined.drop(columns=['target'])
    y = df_combined['target']
    dates = df_combined.index

    split_idx = int(len(df_combined) * (1 - test_size))
    
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), index=X_train_raw.index, columns=X_train_raw.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), index=X_test_raw.index, columns=X_test_raw.columns)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'dates_train': dates_train,
        'dates_test': dates_test,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }