import numpy as np
def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); return {'MSE':float(np.mean((y-p)**2)),'MAE':float(np.mean(abs(y-p))),'DirectionAccuracy':float(np.mean(np.sign(y)==np.sign(p))),'PredictionCorrelation':float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else np.nan}
