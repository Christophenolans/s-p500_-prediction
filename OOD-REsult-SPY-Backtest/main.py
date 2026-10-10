from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
RESULT_DIR = ROOT / 'result' / 'backtest'
TRAIN_DAYS = 7 * 252   # approximately 7 trading years
TEST_DAYS = 252        # approximately 1 trading year
N_COMPONENTS = 5
RIDGE = 1e-6           # numerical stability for normal equations


def read_csv(path):
    return pd.read_csv(path)


def load_sp500():
    """Load SPY history from data/spy_data.csv, including its multi-row header."""
    path = DATA_DIR / 'spy_data.csv'
    if not path.exists():
        url = 'https://raw.githubusercontent.com/Frikzxyz/OOD-REsult/main/data/spy_data.csv'
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            downloaded = pd.read_csv(url, skiprows=[1, 2])
            downloaded.to_csv(path, index=False)
            print(f'Downloaded SPY data to {path}')
        except Exception as exc:
            raise FileNotFoundError(
                f'Cannot find {path} and automatic download failed ({exc}). '
                'Download spy_data.csv from the repository and place it in the data/ folder.'
            ) from exc

    # Original file: first row is column names, next two rows are ticker/date metadata.
    df = pd.read_csv(path, skiprows=[1, 2])
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = next((c for c in ['date', 'price', 'observation_date', 'datetime'] if c in df.columns), None)
    close_col = next((c for c in ['close', 'adj close', 'adj_close'] if c in df.columns), None)
    if date_col is None or close_col is None:
        # Support a conventional CSV with one header row as well.
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        date_col = next((c for c in ['date', 'price', 'observation_date', 'datetime'] if c in df.columns), None)
        close_col = next((c for c in ['close', 'adj close', 'adj_close'] if c in df.columns), None)
    if date_col is None or close_col is None:
        raise ValueError(f'Unexpected spy_data.csv columns: {list(df.columns)}')
    df['date'] = pd.to_datetime(df[date_col], utc=True, errors='coerce').dt.tz_convert(None).dt.normalize()
    df['close'] = pd.to_numeric(df[close_col], errors='coerce')
    df = df[['date', 'close']].dropna().drop_duplicates('date').sort_values('date').set_index('date')
    print(f"SPY history loaded: {df.index.min().date()} to {df.index.max().date()} ({len(df)} rows)")
    return df


def load_indicators(index):
    frames = []
    excluded = {'spy_data.csv', 'SPX500_1d.csv', 'sp500_sample.csv'}
    for path in sorted(DATA_DIR.glob('*.csv')):
        if path.name in excluded:
            continue
        try:
            raw = read_csv(path)
            date_col = 'observation_date' if 'observation_date' in raw.columns else ('date' if 'date' in raw.columns else None)
            if not date_col:
                continue
            value_cols = [c for c in raw.columns if c != date_col]
            if not value_cols:
                continue
            # Use the first numeric series in each file.
            col = value_cols[0]
            tmp = raw[[date_col, col]].copy()
            tmp['date'] = pd.to_datetime(tmp[date_col], utc=True, errors='coerce').dt.tz_convert(None).dt.normalize()
            tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
            tmp = tmp[['date', col]].dropna().drop_duplicates('date').sort_values('date').set_index('date')
            if tmp.empty:
                continue
            # Conservative availability approximation: delay low-frequency observations by one release interval.
            if len(tmp.index) > 2 and tmp.index.to_series().diff().dt.days.median() > 5:
                tmp[col] = tmp[col].shift(1)
            name = path.stem.lower().replace('icebofaushighyieldindexoptionadjustedspread', 'credit_spread')
            tmp.columns = [name]
            frames.append(tmp.reindex(index).ffill())
        except Exception as exc:
            warnings.warn(f'Skipping {path.name}: {exc}')
    if not frames:
        return pd.DataFrame(index=index)
    out = pd.concat(frames, axis=1)
    # Avoid duplicate credit spread series if both alias files exist.
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def make_dataset():
    sp = load_sp500()
    macro = load_indicators(sp.index)
    x = pd.DataFrame(index=sp.index)
    daily_return = sp['close'].pct_change()
    x['sp500_return_lag1'] = daily_return.shift(1)
    x['sp500_return_lag5'] = sp['close'].pct_change(5).shift(1)
    for col in macro.columns:
        s = macro[col]
        # Both level and change help capture different indicator behaviour.
        x[f'{col}_level'] = s
        x[f'{col}_change'] = s.pct_change().replace([np.inf, -np.inf], np.nan)
    y = daily_return.shift(-1).rename('actual_return')
    dataset = x.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dataset) < 100:
        raise ValueError(f'Only {len(dataset)} complete vectors available. Need at least 100; inspect date overlap and missing data.')
    return dataset


def transpose(a): return [list(row) for row in zip(*a)]
def matmul(a, b):
    bt = transpose(b)
    return [[sum(x*y for x, y in zip(row, col)) for col in bt] for row in a]
def eye(n): return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def determinant(a):
    n = len(a)
    if n == 1: return a[0][0]
    if n == 2: return a[0][0]*a[1][1] - a[0][1]*a[1][0]
    total = 0.0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in a[1:]]
        total += ((-1)**j) * a[0][j] * determinant(minor)
    return total


def inverse_adjugate(a):
    n = len(a); det = determinant(a)
    if abs(det) < 1e-12: raise ValueError('Matrix is singular; adjugate inverse is unstable.')
    if n == 1: return [[1.0 / a[0][0]]]
    cof = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = [r[:j] + r[j+1:] for k, r in enumerate(a) if k != i]
            row.append(((-1)**(i+j)) * determinant(minor))
        cof.append(row)
    return [[v/det for v in row] for row in transpose(cof)]


def solve_gaussian(a, b):
    a = [list(map(float, row)) for row in a]; b = list(map(float, b)); n = len(a)
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[p][i]) < 1e-12: raise ValueError('Singular system')
        a[i], a[p] = a[p], a[i]; b[i], b[p] = b[p], b[i]
        for r in range(i+1, n):
            f = a[r][i] / a[i][i]
            for c in range(i, n): a[r][c] -= f*a[i][c]
            b[r] -= f*b[i]
    x = [0.0]*n
    for i in range(n-1, -1, -1): x[i] = (b[i] - sum(a[i][j]*x[j] for j in range(i+1, n))) / a[i][i]
    return x


def solve_gauss_jordan(a, b):
    aug = [list(map(float, row)) + [float(rhs)] for row, rhs in zip(a, b)]; n = len(aug)
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[p][i]) < 1e-12: raise ValueError('Singular system')
        aug[i], aug[p] = aug[p], aug[i]
        d = aug[i][i]; aug[i] = [v/d for v in aug[i]]
        for r in range(n):
            if r == i: continue
            f = aug[r][i]; aug[r] = [u-f*v for u, v in zip(aug[r], aug[i])]
    return [row[-1] for row in aug]


def solve_lu(a, b):
    n = len(a); L = eye(n); U = [list(map(float, row)) for row in a]
    for i in range(n):
        if abs(U[i][i]) < 1e-12: raise ValueError('Zero pivot in LU')
        for r in range(i+1, n):
            f = U[r][i] / U[i][i]; L[r][i] = f
            for c in range(i, n): U[r][c] -= f*U[i][c]
    y = [0.0]*n
    for i in range(n): y[i] = b[i] - sum(L[i][j]*y[j] for j in range(i))
    x = [0.0]*n
    for i in range(n-1, -1, -1): x[i] = (y[i]-sum(U[i][j]*x[j] for j in range(i+1,n))) / U[i][i]
    return x


def fit_pca(x_train, x_test, n_components):
    mu = x_train.mean(axis=0); sigma = x_train.std(axis=0, ddof=0)
    sigma[sigma < 1e-12] = 1.0
    z_train = (x_train-mu)/sigma; z_test = (x_test-mu)/sigma
    cov = z_train.T @ z_train / max(len(z_train)-1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    k = max(1, min(n_components, len(vals), len(x_train)-1))
    w = vecs[:, :k]
    return z_train @ w, z_test @ w, vals[:k], (mu, sigma, w)


def fit_predict_all_methods(x_train, y_train, x_test):
    # Add intercept; solve ridge-stabilized normal equations for comparable methods.
    a = np.column_stack([np.ones(len(x_train)), x_train])
    at = np.column_stack([np.ones(len(x_test)), x_test])
    A = a.T @ a; rhs = a.T @ y_train
    A = A + np.eye(A.shape[0])*RIDGE; A[0,0] -= RIDGE
    A_list = A.tolist(); rhs_list = rhs.tolist()
    methods = {
        'Gaussian': lambda: solve_gaussian(A_list, rhs_list),
        'Gauss-Jordan': lambda: solve_gauss_jordan(A_list, rhs_list),
        'LU': lambda: solve_lu(A_list, rhs_list),
        'Adjugate': lambda: np.asarray(matmul(inverse_adjugate(A_list), [[v] for v in rhs_list])).ravel().tolist(),
        'Least Squares': lambda: np.linalg.lstsq(a, y_train, rcond=None)[0].tolist(),
        'SVD': lambda: (np.linalg.pinv(a) @ y_train).tolist(),
    }
    preds = {}
    for name, fn in methods.items():
        try:
            beta = np.asarray(fn(), dtype=float)
            preds[name] = at @ beta
        except Exception as exc:
            warnings.warn(f'{name} failed this round: {exc}')
    return preds


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    direction = np.mean(np.sign(y) == np.sign(p))*100
    corr = float(np.corrcoef(y, p)[0,1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {'MSE': float(np.mean((y-p)**2)), 'MAE': float(np.mean(np.abs(y-p))), 'Direction_Accuracy_pct': float(direction), 'Correlation': corr}


def run_backtest():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = make_dataset()
    feature_cols = [c for c in data.columns if c != 'actual_return']
    # Require the configured training window plus at least one test vector.
    if len(data) <= TRAIN_DAYS:
        raise ValueError(
            f'Only {len(data)} complete vectors available; need more than {TRAIN_DAYS} '
            'for a 7-year training window. Check spy_data.csv and date overlap with macro indicators.'
        )
    all_predictions = []; all_metrics = []; round_no = 0
    train_start = 0
    while train_start + TRAIN_DAYS < len(data):
        train_end = train_start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, len(data))
        train = data.iloc[train_start:train_end]
        test = data.iloc[train_end:test_end]
        if len(test) == 0: break
        round_no += 1
        xtr = train[feature_cols].to_numpy(float); xte = test[feature_cols].to_numpy(float)
        ytr = train['actual_return'].to_numpy(float); yte = test['actual_return'].to_numpy(float)
        pc_tr, pc_te, eigenvalues, _ = fit_pca(xtr, xte, N_COMPONENTS)
        predictions = fit_predict_all_methods(pc_tr, ytr, pc_te)
        for method, pred in predictions.items():
            for date, actual, estimate in zip(test.index, yte, pred):
                all_predictions.append({'round': round_no, 'train_start': train.index[0].date(), 'train_end': train.index[-1].date(), 'test_date': date.date(), 'method': method, 'actual_return': actual, 'predicted_return': estimate})
            row = {'round': round_no, 'train_start': train.index[0].date(), 'train_end': train.index[-1].date(), 'test_start': test.index[0].date(), 'test_end': test.index[-1].date(), 'method': method, 'train_vectors': len(train), 'test_vectors': len(test), 'PCA_components': pc_tr.shape[1], 'explained_variance_ratio': float(np.sum(eigenvalues)/max(np.sum(np.linalg.eigvalsh(np.cov((xtr-xtr.mean(0))/(xtr.std(0)+1e-12), rowvar=False))), 1e-12)) if len(eigenvalues) else np.nan}
            row.update(metrics(yte, pred)); all_metrics.append(row)
        # Non-overlapping quarterly tests; retrain with one quarter more history.
        train_start += TEST_DAYS
    pred_df = pd.DataFrame(all_predictions)
    metric_df = pd.DataFrame(all_metrics)
    pred_df.to_csv(RESULT_DIR/'predictions.csv', index=False)
    metric_df.to_csv(RESULT_DIR/'metrics_by_round.csv', index=False)
    if not metric_df.empty:
        summary = metric_df.groupby('method')[['MSE','MAE','Direction_Accuracy_pct','Correlation']].mean().reset_index()
        summary.to_csv(RESULT_DIR/'method_comparison.csv', index=False)
        summary.to_string(RESULT_DIR/'summary.txt', index=False)
        # Plot actual vs prediction for each method, with all test rounds concatenated.
        for method in pred_df['method'].unique():
            part = pred_df[pred_df.method == method].copy()
            plt.figure(figsize=(12,5)); plt.plot(pd.to_datetime(part.test_date), part.actual_return, label='Actual', linewidth=1)
            plt.plot(pd.to_datetime(part.test_date), part.predicted_return, label='Predicted', linewidth=1)
            plt.title(f'Walk-forward Backtest - {method}'); plt.xlabel('Date'); plt.ylabel('Next-day return'); plt.legend(); plt.tight_layout()
            plt.savefig(RESULT_DIR/f'prediction_{method.lower().replace(" ", "_").replace("-", "_")}.png', dpi=150); plt.close()
    print(f'Rows/vectors available: {len(data)}')
    print(f'Feature count: {len(feature_cols)}')
    print(f'Backtest rounds: {round_no}')
    print(f'Results saved to: {RESULT_DIR}')
    if len(data.index) and (data.index[-1]-data.index[0]).days < 365*7:
        print('NOTE: configured for 7-year training and 1-year testing; the final test block may be shorter.')

if __name__ == '__main__':
    run_backtest()
