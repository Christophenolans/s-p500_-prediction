from pathlib import Path
ROOT=Path(__file__).resolve().parent
RAW_DIR=ROOT/'data/raw'; PROCESSED_DIR=ROOT/'data/processed'; OUTPUT_DIR=ROOT/'outputs'
SERIES={'interest_rate':'DFF','cpi':'CPIAUCSL','ppi':'PPIACO','nfp':'PAYEMS','gdp':'GDPC1','vix':'VIXCLS','bond10y':'DGS10','productivity':'OPHPBS'}
FEATURES=['interest_rate','interest_rate_change','cpi_yoy','cpi_change','ppi_yoy','ppi_change','nfp_yoy','nfp_change','gdp_yoy','gdp_change','vix','vix_change','bond10y','bond10y_change','term_spread_10y_ff','productivity_yoy','productivity_change']
METHODS=['gaussian','gauss_jordan','lu','adjugate','least_squares','svd']
TRAIN_YEARS=7; START_TEST_YEAR=2024; N_COMPONENTS=5
