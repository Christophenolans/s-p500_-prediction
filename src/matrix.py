class Matrix:
    def __init__(self, data):
        self.data = [list(map(float, r)) for r in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows else 0

    def transpose(self):
        return Matrix([[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)])

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("dimension mismatch")
        return Matrix([[sum(self.data[i][k]*other.data[k][j] for k in range(self.cols))
                        for j in range(other.cols)] for i in range(self.rows)])

    def add_column(self, values):
        return Matrix([self.data[i] + [float(values[i])] for i in range(self.rows)])

    def subtract(self, other):
        return Matrix([[self.data[i][j]-other.data[i][j] for j in range(self.cols)]
                       for i in range(self.rows)])

    def scalar(self, s):
        return Matrix([[x*s for x in row] for row in self.data])

    def inverse_gauss_jordan(self):
        if self.rows != self.cols:
            raise ValueError("square matrix required")
        n = self.rows
        a = [self.data[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            p = max(range(col,n), key=lambda r: abs(a[r][col]))
            if abs(a[p][col]) < 1e-12:
                raise ValueError("singular matrix")
            a[col], a[p] = a[p], a[col]
            pivot = a[col][col]
            a[col] = [x/pivot for x in a[col]]
            for r in range(n):
                if r != col:
                    q = a[r][col]
                    a[r] = [a[r][c]-q*a[col][c] for c in range(2*n)]
        return Matrix([r[n:] for r in a])

def solve_gaussian(A, b):
    a = [A.data[i][:] + [float(b.data[i][0])] for i in range(A.rows)]
    n = A.rows
    m = A.cols
    for col in range(min(n,m)):
        p = max(range(col,n), key=lambda r: abs(a[r][col]))
        if abs(a[p][col]) < 1e-12:
            continue
        a[col], a[p] = a[p], a[col]
        for r in range(col+1,n):
            q = a[r][col]/a[col][col]
            for c in range(col,m+1):
                a[r][c] -= q*a[col][c]
    x = [0.0]*m
    for i in range(min(n,m)-1,-1,-1):
        pivot = a[i][i]
        if abs(pivot) < 1e-12:
            continue
        x[i] = (a[i][m]-sum(a[i][j]*x[j] for j in range(i+1,m)))/pivot
    return Matrix([[v] for v in x])

def outer(v):
    return Matrix([[v[i]*v[j] for j in range(len(v))] for i in range(len(v))])