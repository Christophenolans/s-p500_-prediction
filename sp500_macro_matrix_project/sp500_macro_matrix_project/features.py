import numpy as np

def build(sp,macro):
    d=sp.copy()
    for n,s in macro.items(): d[n]=s.reindex(d.index).ffill()
    for n in ['interest_rate','vix','bond10y']:
        d[n+'_change']=d[n].diff()
    for n in ['cpi','ppi','nfp','gdp','productivity']:
        d[n+'_yoy']=d[n].pct_change(12)*100
        d[n+'_change']=d[n+'_yoy'].diff()
    d['term_spread_10y_ff']=d['bond10y']-d['interest_rate']
    d['target']=d['close'].pct_change().shift(-1)
    return d
