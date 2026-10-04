import numpy as np

def inv_gj(A):
    A=np.asarray(A,float); n=len(A); M=np.c_[A,np.eye(n)]
    for i in range(n):
        p=i+np.argmax(np.abs(M[i:,i]));
        if abs(M[p,i])<1e-12: raise ValueError('singular')
        M[[i,p]]=M[[p,i]]; M[i]/=M[i,i]
        for r in range(n):
            if r!=i: M[r]-=M[r,i]*M[i]
    return M[:,n:]

def gauss(A,b):
    A=np.c_[np.asarray(A,float),np.asarray(b,float)]; n=len(A)
    for i in range(n):
        p=i+np.argmax(abs(A[i:,i]));
        if abs(A[p,i])<1e-12: raise ValueError('singular')
        A[[i,p]]=A[[p,i]]
        for r in range(i+1,n): A[r]-=A[r,i]/A[i,i]*A[i]
    x=np.zeros(n)
    for i in range(n-1,-1,-1): x[i]=(A[i,-1]-A[i,i+1:-1]@x[i+1:])/A[i,i]
    return x

def lu(A,b):
    A=np.asarray(A,float); b=np.asarray(b,float); n=len(A); L=np.eye(n); U=A.copy()
    for i in range(n):
        if abs(U[i,i])<1e-12: raise ValueError('zero pivot')
        for r in range(i+1,n): L[r,i]=U[r,i]/U[i,i]; U[r]-=L[r,i]*U[i]
    y=np.zeros(n)
    for i in range(n): y[i]=b[i]-L[i,:i]@y[:i]
    x=np.zeros(n)
    for i in range(n-1,-1,-1): x[i]=(y[i]-U[i,i+1:]@x[i+1:])/U[i,i]
    return x

def adj(A):
    A=np.asarray(A,float); n=len(A); det=np.linalg.det(A)
    if abs(det)<1e-12: raise ValueError('singular')
    C=np.zeros_like(A)
    for i in range(n):
        for j in range(n): C[i,j]=(-1)**(i+j)*np.linalg.det(np.delete(np.delete(A,i,0),j,1))
    return C.T/det

def beta(X,y,method):
    X=np.c_[np.ones(len(X)),X]; A=X.T@X; b=X.T@y
    if method=='gaussian': return gauss(A,b)
    if method=='gauss_jordan': return inv_gj(A)@b
    if method=='lu': return lu(A,b)
    if method=='adjugate': return adj(A)@b
    if method=='least_squares': return np.linalg.lstsq(X,y,rcond=None)[0]
    if method=='svd':
        u,s,vt=np.linalg.svd(X,full_matrices=False); return vt.T@((u.T@y)/s)
    raise ValueError(method)
