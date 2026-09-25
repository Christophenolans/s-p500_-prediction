import matplotlib.pyplot as plt
from config import OUTPUT_DIR,FEATURES
def corr(df):
    c=df[FEATURES+['target_next_return']].corr(); fig,ax=plt.subplots(figsize=(14,11)); im=ax.imshow(c,aspect='auto'); ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c))); ax.set_xticklabels(c.columns,rotation=90,fontsize=7); ax.set_yticklabels(c.columns,fontsize=7); fig.colorbar(im,ax=ax); ax.set_title('Daily Indicator Correlation'); fig.tight_layout(); fig.savefig(OUTPUT_DIR/'01_correlation_heatmap.png',dpi=160); plt.close(fig)
def pca_var(v):
    if v.empty:return
    g=v.groupby('component').explained_variance.mean(); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(g.index,g.values); ax.set_title('PCA Explained Variance'); ax.set_ylabel('Mean Ratio'); fig.tight_layout(); fig.savefig(OUTPUT_DIR/'02_pca_explained_variance.png',dpi=160); plt.close(fig)
def loadings(l):
    if l.empty:return
    y=l.test_year.max(); p=l[l.test_year==y].pivot(index='feature',columns='component',values='loading'); fig,ax=plt.subplots(figsize=(9,9)); im=ax.imshow(p.values,aspect='auto'); ax.set_xticks(range(len(p.columns))); ax.set_xticklabels(p.columns); ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index,fontsize=8); fig.colorbar(im,ax=ax); ax.set_title(f'PCA Loadings - {y}'); fig.tight_layout(); fig.savefig(OUTPUT_DIR/'03_pca_loadings.png',dpi=160); plt.close(fig)
def predictions(p):
    for m,g in p.groupby('method'):
        g=g.sort_values('date'); fig,ax=plt.subplots(figsize=(12,5)); ax.plot(g.date,g.actual_return,label='Actual'); ax.plot(g.date,g.predicted_return,label='Predicted',alpha=.7); ax.axhline(0,linewidth=.8); ax.legend(); ax.set_title(f'Actual vs Predicted Next-Day Return - {m}'); fig.tight_layout(); fig.savefig(OUTPUT_DIR/f'04_prediction_{m}.png',dpi=160); plt.close(fig)
def metrics_plot(m):
    for method,g in m.groupby('method'):
        fig,ax=plt.subplots(figsize=(9,5)); ax.plot(g.test_year,g.DirectionAccuracy,marker='o'); ax.set_ylim(0,1); ax.grid(True); ax.set_title(f'Direction Accuracy - {method}'); fig.tight_layout(); fig.savefig(OUTPUT_DIR/f'05_direction_accuracy_{method}.png',dpi=160); plt.close(fig)
