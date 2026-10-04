import numpy as np,pandas as pd
from config import FEATURES,METHODS,TRAIN_YEARS,START_TEST_YEAR,N_COMPONENTS
from matrix_core import beta

def run(d):
    metrics=[]; preds=[]; pcas=[]
    for year in sorted(d.index.year.unique()):
        if year<START_TEST_YEAR: continue
        tr=d[(d.index.year>=year-TRAIN_YEARS)&(d.index.year<year)].dropna(subset=FEATURES+['target'])
        te=d[d.index.year==year].dropna(subset=FEATURES+['target'])
        if len(tr)<300 or len(te)<10: continue
        X=tr[FEATURES].values; Z=te[FEATURES].values; y=tr.target.values; yt=te.target.values
        mu=X.mean(0); sd=X.std(0,ddof=1); sd[sd<1e-12]=1
        X=(X-mu)/sd; Z=(Z-mu)/sd
        C=X.T@X/(len(X)-1); vals,V=np.linalg.eigh(C); order=np.argsort(vals)[::-1][:N_COMPONENTS]; vals=vals[order]; W=V[:,order]
        X=X@W; Z=Z@W
        for i,v in enumerate(vals): pcas.append({'test_year':year,'pc':i+1,'eigenvalue':v,'explained_variance_ratio':v/max(vals.sum(),1e-12)})
        for m in METHODS:
            try:
                b=beta(X,y,m); pred=np.c_[np.ones(len(Z)),Z]@b; err=yt-pred
                metrics.append({'test_year':year,'method':m,'MSE':np.mean(err**2),'MAE':np.mean(abs(err)),'Direction_Accuracy_%':np.mean(np.sign(yt)==np.sign(pred))*100,'Prediction_Correlation':np.corrcoef(yt,pred)[0,1]})
                preds += [{'date':idx,'test_year':year,'method':m,'actual_return':a,'predicted_return':p} for idx,a,p in zip(te.index,yt,pred)]
            except Exception as e: print('[WARN]',year,m,e)
    return pd.DataFrame(metrics),pd.DataFrame(preds),pd.DataFrame(pcas)
