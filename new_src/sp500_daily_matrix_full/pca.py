import numpy as np
def standardize(X):
    mean=X.mean(0); std=X.std(0,ddof=1); std[std<1e-12]=1; return (X-mean)/std,mean,std
def fit_pca(X,k=5):
    Z,mean,std=standardize(X); C=Z.T@Z/(len(Z)-1); vals,V=np.linalg.eigh(C); order=np.argsort(vals)[::-1]; vals=np.clip(vals[order],0,None); V=V[:,order]; k=min(k,X.shape[1]); W=V[:,:k]; return {'mean':mean,'std':std,'covariance':C,'eigenvalues':vals[:k],'loadings':W,'scores':Z@W,'explained_variance':vals[:k]/vals.sum()}
def transform(X,m): return ((X-m['mean'])/m['std'])@m['loadings']
