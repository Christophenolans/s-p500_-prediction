import numpy as np
def svd_via_eigen(X):
    vals,V=np.linalg.eigh(X.T@X); order=np.argsort(vals)[::-1]; vals=np.clip(vals[order],0,None); V=V[:,order]; s=np.sqrt(vals); keep=s>1e-10; s=s[keep]; V=V[:,keep]; U=X@V/s; return U,s,V
