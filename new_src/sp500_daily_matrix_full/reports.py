import numpy as np
import pandas as pd
def annual_summary(p):
    rows=[]
    for (y,m),g in p.groupby(['test_year','method']):
        a=g.actual_return.to_numpy(); q=g.predicted_return.to_numpy(); rows.append({'year':y,'method':m,'actual_cumulative_return':np.prod(1+a)-1,'predicted_cumulative_return':np.prod(1+q)-1,'actual_mean_daily_return':a.mean(),'predicted_mean_daily_return':q.mean(),'direction_accuracy':np.mean(np.sign(a)==np.sign(q))})
    return pd.DataFrame(rows)
def comparison(m): return m.groupby('method').agg(mean_MSE=('MSE','mean'),mean_MAE=('MAE','mean'),mean_direction_accuracy=('DirectionAccuracy','mean'),mean_prediction_correlation=('PredictionCorrelation','mean')).reset_index()
