def shape(A): return len(A),len(A[0]) if A else 0
def zeros(r,c): return [[0.0]*c for _ in range(r)]
def identity(n):
    I=zeros(n,n)
    for i in range(n): I[i][i]=1.0
    return I
def transpose(A): return [list(x) for x in zip(*A)]
def matmul(A,B):
    ar,ac=shape(A); br,bc=shape(B)
    if ac!=br: raise ValueError('Matrix dimensions do not match')
    return [[sum(A[i][k]*B[k][j] for k in range(ac)) for j in range(bc)] for i in range(ar)]
def determinant(A):
    n,m=shape(A)
    if n!=m: raise ValueError('Square matrix required')
    if n==0:return 1.0
    if n==1:return A[0][0]
    if n==2:return A[0][0]*A[1][1]-A[0][1]*A[1][0]
    return sum(((-1)**j)*A[0][j]*determinant([[A[r][c] for c in range(n) if c!=j] for r in range(1,n)]) for j in range(n))
def gaussian_solve(A,b):
    n=len(A); M=[list(map(float,A[i]))+[float(b[i])] for i in range(n)]
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(M[i][k]))
        if abs(M[p][k])<1e-12: raise ValueError('Singular matrix')
        M[k],M[p]=M[p],M[k]
        for i in range(k+1,n):
            f=M[i][k]/M[k][k]
            for j in range(k,n+1): M[i][j]-=f*M[k][j]
    x=[0.0]*n
    for i in range(n-1,-1,-1): x[i]=(M[i][n]-sum(M[i][j]*x[j] for j in range(i+1,n)))/M[i][i]
    return x
def gauss_jordan_inverse(A):
    n=len(A); I=identity(n); M=[A[i][:]+I[i] for i in range(n)]
    for c in range(n):
        p=max(range(c,n),key=lambda i:abs(M[i][c]))
        if abs(M[p][c])<1e-12: raise ValueError('Singular matrix')
        M[c],M[p]=M[p],M[c]; q=M[c][c]; M[c]=[v/q for v in M[c]]
        for i in range(n):
            if i!=c:
                f=M[i][c]; M[i]=[M[i][j]-f*M[c][j] for j in range(2*n)]
    return [r[n:] for r in M]
def adjugate_inverse(A):
    n=len(A); d=determinant(A)
    if abs(d)<1e-12: raise ValueError('Singular matrix')
    C=[]
    for i in range(n):
        row=[]
        for j in range(n):
            minor=[[A[r][c] for c in range(n) if c!=j] for r in range(n) if r!=i]
            row.append(((-1)**(i+j))*determinant(minor))
        C.append(row)
    adj=transpose(C); return [[adj[i][j]/d for j in range(n)] for i in range(n)]
def lu_solve(A,b):
    n=len(A); L=identity(n); U=[r[:] for r in A]
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(U[i][k]))
        if abs(U[p][k])<1e-12: raise ValueError('Singular matrix')
        if p!=k:
            U[k],U[p]=U[p],U[k]
            for j in range(k): L[k][j],L[p][j]=L[p][j],L[k][j]
        for i in range(k+1,n):
            L[i][k]=U[i][k]/U[k][k]
            for j in range(k,n): U[i][j]-=L[i][k]*U[k][j]
    y=[0.0]*n
    for i in range(n): y[i]=b[i]-sum(L[i][j]*y[j] for j in range(i))
    x=[0.0]*n
    for i in range(n-1,-1,-1): x[i]=(y[i]-sum(U[i][j]*x[j] for j in range(i+1,n)))/U[i][i]
    return x
def least_squares(X,y,method):
    Xt=transpose(X); A=matmul(Xt,X); b=[r[0] for r in matmul(Xt,[[v] for v in y])]
    if method=='gaussian': return gaussian_solve(A,b)
    if method=='lu': return lu_solve(A,b)
    inv=gauss_jordan_inverse(A) if method=='gauss_jordan' else adjugate_inverse(A)
    return [sum(inv[i][j]*b[j] for j in range(len(b))) for i in range(len(b))]
