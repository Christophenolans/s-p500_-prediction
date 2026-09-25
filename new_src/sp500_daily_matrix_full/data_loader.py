import requests
import pandas as pd
from config import RAW_DIR,SP500_URL,PPI_URL

def download(url,path):
    r=requests.get(url,timeout=60); r.raise_for_status(); path.write_bytes(r.content)

def load_sp500():
    path=RAW_DIR/'sp500.csv'
    if not path.exists(): download(SP500_URL,path)
    df=pd.read_csv(path); df.columns=[str(c).strip().lower().replace(' ','_').replace('-','_') for c in df.columns]
    if 'date' not in df.columns: raise ValueError('S&P 500 CSV must contain Date')
    df['date']=pd.to_datetime(df['date'],errors='coerce'); df=df.dropna(subset=['date']).sort_values('date').drop_duplicates('date')
    if 'adj_close' not in df.columns: df['adj_close']=df['close']
    for c in ['open','high','low','close','adj_close','volume']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    missing=[c for c in ['open','high','low','close'] if c not in df.columns]
    if missing: raise ValueError(f'Missing columns: {missing}')
    return df.reset_index(drop=True)

def load_ppi():
    path=RAW_DIR/'ppiaco.csv'
    if not path.exists(): download(PPI_URL,path)
    df=pd.read_csv(path); df.columns=[str(c).strip().lower() for c in df.columns]
    df=df.rename(columns={df.columns[0]:'date',df.columns[1]:'ppi'})
    df['date']=pd.to_datetime(df['date'],errors='coerce'); df['ppi']=pd.to_numeric(df['ppi'],errors='coerce')
    return df.dropna().sort_values('date').drop_duplicates('date').reset_index(drop=True)

def align_ppi(sp500,ppi):
    a=pd.merge_asof(sp500.sort_values('date'),ppi.sort_values('date'),on='date',direction='backward')
    monthly=ppi.set_index('date')['ppi']; yoy=monthly.pct_change(12)
    a['ppi_yoy']=a['date'].map(yoy)
    return a
