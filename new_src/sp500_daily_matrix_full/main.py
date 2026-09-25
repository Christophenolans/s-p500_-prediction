from config import OUTPUT_DIR,PROCESSED_DIR,FEATURES
from data_loader import load_sp500,load_ppi,align_ppi
from features import add_features,indicator_forecast
from adaptive_model import run
from reports import annual_summary,comparison
from plots import corr,pca_var,loadings,predictions,metrics_plot

def main():
    print('=== S&P 500 DAILY MATRIX PROJECT ===')
    sp=load_sp500(); print('S&P rows:',len(sp)); ppi=load_ppi(); print('PPI rows:',len(ppi))
    df=add_features(align_ppi(sp,ppi)); df.to_csv(PROCESSED_DIR/'daily_features.csv',index=False)
    clean=df.dropna(subset=FEATURES+['target_next_return']); clean[FEATURES+['target_next_return']].corr().to_csv(OUTPUT_DIR/'correlation_matrix.csv'); corr(clean)
    ind=indicator_forecast(df,7); ind.to_csv(OUTPUT_DIR/'indicator_7year_change_forecast.csv',index=False)
    pred,met,coef,load,var=run(df); pred.to_csv(OUTPUT_DIR/'daily_predictions.csv',index=False); met.to_csv(OUTPUT_DIR/'adaptive_backtest_metrics.csv',index=False); coef.to_csv(OUTPUT_DIR/'adaptive_model_coefficients.csv',index=False); load.to_csv(OUTPUT_DIR/'pca_loadings_by_year.csv',index=False); var.to_csv(OUTPUT_DIR/'pca_variance_by_year.csv',index=False)
    annual_summary(pred).to_csv(OUTPUT_DIR/'annual_prediction_summary.csv',index=False); comparison(met).to_csv(OUTPUT_DIR/'method_comparison.csv',index=False)
    pca_var(var); loadings(load); predictions(pred); metrics_plot(met)
    print('Completed. Outputs:',OUTPUT_DIR)
    if not met.empty: print(comparison(met).to_string(index=False))
if __name__=='__main__': main()
