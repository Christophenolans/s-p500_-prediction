import math

def mean(v):
    return sum(v)/len(v) if v else 0.0

def std(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))

def standardize(rows):
    cols = len(rows[0])
    mu = [mean([r[j] for r in rows]) for j in range(cols)]
    sd = [std([r[j] for r in rows]) for j in range(cols)]
    sd = [x if x > 1e-12 else 1.0 for x in sd]
    z = [[(r[j]-mu[j])/sd[j] for j in range(cols)] for r in rows]
    return z, mu, sd