import math
from .matrix import Matrix, outer

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def normalize(v):
    n = norm(v)
    return [x/n for x in v] if n else v

def mat_vec(A,v):
    return [sum(A.data[i][j]*v[j] for j in range(A.cols)) for i in range(A.rows)]

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))

def power_iteration(A, iterations=1000, tol=1e-10):
    n = A.rows
    v = normalize([1.0 + i/n for i in range(n)])
    last = 0.0
    for _ in range(iterations):
        w = mat_vec(A,v)
        v2 = normalize(w)
        Av = mat_vec(A,v2)
        lam = dot(v2,Av)
        if abs(lam-last) < tol:
            v = v2
            break
        v, last = v2, lam
    Av = mat_vec(A,v)
    lam = dot(v,Av)
    return lam, v

def deflate(A, lam, v):
    return A.subtract(outer(v).scalar(lam))

def top_eigenpairs(A, k):
    B = A
    out = []
    for _ in range(min(k,A.rows)):
        lam,v = power_iteration(B)
        out.append((lam,v))
        B = deflate(B,lam,v)
    return out