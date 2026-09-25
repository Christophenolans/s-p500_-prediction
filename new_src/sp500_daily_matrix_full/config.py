from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/'data'; RAW_DIR=DATA_DIR/'raw'; PROCESSED_DIR=DATA_DIR/'processed'; OUTPUT_DIR=BASE_DIR/'outputs'
for d in [RAW_DIR,PROCESSED_DIR,OUTPUT_DIR]: d.mkdir(parents=True,exist_ok=True)
SP500_URL="https://raw.githubusercontent.com/vijinho/sp500/master/csv/sp500.csv"
PPI_URL="https://fred.stlouisfed.org/graph/fredgraph.csv?id=PPIACO"
TRAIN_YEARS=7; START_TEST_YEAR=2020; N_COMPONENTS=5
METHODS=['gaussian','gauss_jordan','lu','adjugate']
FEATURES=['sma_5','sma_20','sma_50','ema_12','ema_26','momentum_5','momentum_20','roc_20','rsi_14','macd','stochastic_k','williams_r','cci_20','atr_14','volatility_20','bollinger_band_width','high_low_range','volume_change','obv_change','ppi_yoy']
TARGET='target_next_return'
