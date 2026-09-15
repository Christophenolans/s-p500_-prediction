from .matrix import Matrix, solve_gaussian
from .stats import standardize
from .eigen import top_eigenpairs
from .features import read_csv, build_features

def fit_predict(path):
    rows=read_csv(path)
    data=build_features(rows)
    split=int(len(data)*0.7)
    train=data[:split]
    test=data[split:]
    Xraw=[x for _,x,_ in train]
    y=Matrix([[t] for _,_,t in train])
    Xz,mu,sd=standardize(Xraw)
    X=Matrix(Xz)
    Xt=X.transpose()
    A=Xt.multiply(X)
    b=Xt.multiply(y)
    beta=solve_gaussian(A,b)
    preds=[]
    for date,feat,target in test:
        z=[(feat[j]-mu[j])/(sd[j] if sd[j]>1e-12 else 1.0) for j in range(len(feat))]
        pred=sum(z[j]*beta.data[j][0] for j in range(len(z)))
        preds.append((date,target,pred))
    cov=A.scalar(1.0/max(1,X.rows-1))
    pairs=top_eigenpairs(cov,min(3,cov.rows))
    mse=sum((t-p)**2 for _,t,p in preds)/len(preds) if preds else 0
    mae=sum(abs(t-p) for _,t,p in preds)/len(preds) if preds else 0
    correct=sum(1 for _,t,p in preds if (t>=0)==(p>=0))
    acc=correct/len(preds) if preds else 0
    return beta,pairs,preds,mse,mae,acc

def write_result(path, outpath):
    beta,pairs,preds,mse,mae,acc=fit_predict(path)
    import csv, os
    os.makedirs(os.path.dirname(outpath),exist_ok=True)
    threshold=0.001
    with open(outpath,newline="",mode="w",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["date","actual_return","predicted_return","signal"])
        for d,t,p in preds:
            s="BUY" if p>threshold else ("SELL" if p<-threshold else "HOLD")
            w.writerow([d,t,p,s])
    with open(os.path.join(os.path.dirname(outpath),"metrics.txt"),"w",encoding="utf-8") as f:
        f.write(f"Test MSE: {mse}\nTest MAE: {mae}\nDirection Accuracy: {acc}\n")
        f.write("Eigenvalues:\n")
        for lam,_ in pairs: f.write(str(lam)+"\n")
    return beta,pairs,preds,mse,mae,acc