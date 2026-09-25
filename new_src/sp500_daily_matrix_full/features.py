import numpy as np

def rsi(c,p=14):
    d=c.diff(); g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean(); rs=g/l.replace(0,np.nan); return 100-100/(1+rs)
def atr(df,p=14):
    pc=df.close.shift(1); tr=np.maximum(df.high-df.low,np.maximum((df.high-pc).abs(),(df.low-pc).abs())); return tr.rolling(p).mean()
def stochastic(df,p=14):
    lo=df.low.rolling(p).min(); hi=df.high.rolling(p).max(); return 100*(df.close-lo)/(hi-lo).replace(0,np.nan)
def williams(df,p=14):
    lo=df.low.rolling(p).min(); hi=df.high.rolling(p).max(); return -100*(hi-df.close)/(hi-lo).replace(0,np.nan)
def cci(df,p=20):
    tp=(df.high+df.low+df.close)/3; m=tp.rolling(p).mean(); mad=tp.rolling(p).apply(lambda x:np.mean(np.abs(x-np.mean(x))),raw=True); return (tp-m)/(0.015*mad.replace(0,np.nan))

def add_features(df):
    x=df.copy(); c=x.close
    x['return_1d']=c.pct_change(); x['target_next_return']=c.shift(-1)/c-1
    x['sma_5']=c.rolling(5).mean(); x['sma_20']=c.rolling(20).mean(); x['sma_50']=c.rolling(50).mean()
    x['ema_12']=c.ewm(span=12,adjust=False).mean(); x['ema_26']=c.ewm(span=26,adjust=False).mean()
    x['momentum_5']=c/c.shift(5)-1; x['momentum_20']=c/c.shift(20)-1; x['roc_20']=x.momentum_20
    x['rsi_14']=rsi(c); x['macd']=x.ema_12-x.ema_26; x['stochastic_k']=stochastic(x); x['williams_r']=williams(x); x['cci_20']=cci(x)
    x['atr_14']=atr(x); x['volatility_20']=x.return_1d.rolling(20).std()*np.sqrt(252)
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); x['bollinger_band_width']=4*sd/mid.replace(0,np.nan)
    x['high_low_range']=(x.high-x.low)/c.replace(0,np.nan)
    if 'volume' in x:
        direction=np.sign(c.diff()).fillna(0); x['volume_change']=x.volume.pct_change(); x['obv']=(direction*x.volume).cumsum(); x['obv_change']=x.obv.pct_change()
    else: x['volume_change']=np.nan; x['obv_change']=np.nan
    return x

def indicator_forecast(df,years=7):
    x=df.copy(); x['year']=x.date.dt.year; annual=x.groupby('year').last(numeric_only=True)
    cols=[c for c in ['sma_5','sma_20','sma_50','ema_12','ema_26','momentum_5','momentum_20','roc_20','rsi_14','macd','stochastic_k','williams_r','cci_20','atr_14','volatility_20','bollinger_band_width','high_low_range','volume_change','obv_change','ppi_yoy'] if c in annual]
    rows=[]
    for col in cols:
        ch=annual[col].diff()
        for y in annual.index:
            hist=ch.loc[:y-1].tail(years); rows.append({'year':y,'indicator':col,'expected_change_7y_mean':hist.mean() if len(hist) else np.nan,'actual_change':ch.loc[y]})
    import pandas as pd
    return pd.DataFrame(rows)
