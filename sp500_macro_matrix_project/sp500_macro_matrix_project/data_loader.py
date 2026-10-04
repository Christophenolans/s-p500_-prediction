import requests,pandas as pd
from config import RAW_DIR,SERIES

def get(url,path):
    r=requests.get(url,timeout=60); r.raise_for_status(); path.write_bytes(r.content)

def fred(sid,name):
    p=RAW_DIR/f'{name}_{sid}.csv'; get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}',p)
    d=pd.read_csv(p); d.iloc[:,0]=pd.to_datetime(d.iloc[:,0]); d.iloc[:,1]=pd.to_numeric(d.iloc[:,1],errors='coerce')
    return d.set_index(d.columns[0]).iloc[:,0].rename(name).sort_index()

def load():
    RAW_DIR.mkdir(parents=True,exist_ok=True)
    sp=fred('SP500','sp500').rename('close').to_frame()
    macro={}
    for name,sid in SERIES.items():
        try: macro[name]=fred(sid,name)
        except Exception as e: print('[WARN]',name,e); macro[name]=pd.Series(dtype=float,name=name)
    return sp,macro
