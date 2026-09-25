from matrix_core import least_squares
def add_intercept(X): return [[1.0]+list(r) for r in X]
def predict(X,b): return [sum(r[j]*b[j] for j in range(len(b))) for r in X]
def fit_predict(Xtr,ytr,Xte,method):
    b=least_squares(add_intercept(Xtr),ytr,method); return b,predict(add_intercept(Xte),b)
