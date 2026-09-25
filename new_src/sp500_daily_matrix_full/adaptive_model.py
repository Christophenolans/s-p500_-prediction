import numpy as np
import pandas as pd
from config import FEATURES,TARGET,METHODS,TRAIN_YEARS,START_TEST_YEAR,N_COMPONENTS
from pca import fit_pca,transform
from regression import fit_predict
from evaluation import metrics

def run(df):
    d=df.dropna(subset=FEATURES+[TARGET]).copy(); d['year']=d.date.dt.year
    preds=[]; met=[]; coefs=[]; loads=[]; vars=[]
    for year in sorted(d.year.unique()):
        if year<START_TEST_YEAR: continue
        train=d[(d.year>=year-TRAIN_YEARS)&(d.year<year)]; test=d[d.year==year]
        if len(train)<100 or test.empty: continue
        Xtr=train[FEATURES].to_numpy(float); Xte=test[FEATURES].to_numpy(float); ytr=train[TARGET].to_numpy(float); yte=test[TARGET].to_numpy(float)
        pm=fit_pca(Xtr,N_COMPONENTS); Ztr=pm['scores']; Zte=transform(Xte,pm)
        for i,v in enumerate(pm['eigenvalues'],1): vars.append({'test_year':year,'component':f'PC{i}','eigenvalue':v,'explained_variance':pm['explained_variance'][i-1]})
        for pc in range(pm['loadings'].shape[1]):
            for fi,f in enumerate(FEATURES): loads.append({'test_year':year,'component':f'PC{pc+1}','feature':f,'loading':pm['loadings'][fi,pc]})
        for method in METHODS:
            try:
                b,p=fit_predict(Ztr.tolist(),ytr.tolist(),Zte.tolist(),method); mm=metrics(yte,p)
                met.append({'test_year':year,'method':method,'train_start':year-TRAIN_YEARS,'train_end':year-1,'n_train':len(train),'n_test':len(test),**mm})
                for date,a,q in zip(test.date,yte,p): preds.append({'date':date,'test_year':year,'method':method,'actual_return':a,'predicted_return':q})
                for i,v in enumerate(b): coefs.append({'test_year':year,'method':method,'feature':'intercept' if i==0 else f'PC{i}','coefficient':v})
            except Exception as e: met.append({'test_year':year,'method':method,'train_start':year-TRAIN_YEARS,'train_end':year-1,'n_train':len(train),'n_test':len(test),'MSE':np.nan,'MAE':np.nan,'DirectionAccuracy':np.nan,'PredictionCorrelation':np.nan,'error':str(e)})
    return pd.DataFrame(preds),pd.DataFrame(met),pd.DataFrame(coefs),pd.DataFrame(loads),pd.DataFrame(vars)
