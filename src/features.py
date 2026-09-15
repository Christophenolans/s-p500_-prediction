import csv, math

def read_csv(path):
    with open(path,newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def build_features(rows):
    out=[]
    for i in range(5,len(rows)-1):
        c=float(rows[i]["close"]); p=float(rows[i-1]["close"])
        c5=float(rows[i-5]["close"])
        o=float(rows[i]["open"]); h=float(rows[i]["high"]); l=float(rows[i]["low"])
        vol=float(rows[i]["volume"])
        ret=(c-p)/p
        mom5=(c-c5)/c5
        intraday=(c-o)/o
        range_pct=(h-l)/c
        vol_log=math.log(vol+1.0)
        target=(float(rows[i+1]["close"])-c)/c
        out.append((rows[i]["date"], [ret,mom5,intraday,range_pct,vol_log], target))
    return out