import pandas as pd,matplotlib.pyplot as plt
from config import PROCESSED_DIR,OUTPUT_DIR,FEATURES
from data_loader import load
from features import build
from model import run

def main():
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    sp,macro=load(); d=build(sp,macro); d.to_csv(PROCESSED_DIR/'model_dataset.csv')
    metrics,preds,pca=run(d)
    metrics.to_csv(OUTPUT_DIR/'backtest_metrics.csv',index=False); preds.to_csv(OUTPUT_DIR/'daily_predictions.csv',index=False); pca.to_csv(OUTPUT_DIR/'pca_report.csv',index=False)
    d[FEATURES+['target']].corr()['target'].sort_values().to_csv(OUTPUT_DIR/'feature_target_correlation.csv')
    if not metrics.empty:
        metrics.groupby('method')[['MSE','MAE','Direction_Accuracy_%','Prediction_Correlation']].mean().to_csv(OUTPUT_DIR/'method_comparison.csv')
        metrics.groupby('method')['Direction_Accuracy_%'].mean().plot(kind='barh',figsize=(9,5),title='Average Direction Accuracy by Matrix Method'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'method_direction_accuracy.png',dpi=160); plt.close()
    if not pca.empty:
        pca.groupby('pc').explained_variance_ratio.mean().plot(kind='bar',figsize=(8,5),title='PCA Explained Variance'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'pca_explained_variance.png',dpi=160); plt.close()
    print('DONE:',OUTPUT_DIR)
if __name__=='__main__': main()
