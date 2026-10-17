"""Walk-forward backtest for the S&P 500 Matrix Decision Model.

Core matrix calculations use Python lists and the standard library only.
This module is educational research code; it does not produce trading advice.
"""
from __future__ import annotations
import csv
import math
import os
import urllib.request
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "result", "backtest")
SPY_URL = "https://raw.githubusercontent.com/Frikzxyz/OOD-REsult/main/data/spy_data.csv"

# Macro observations are shifted forward to approximate when the observation
# would have become available. Exact release calendars/vintages are preferable
# for formal research; these fixed lags are a documented approximation.
MACRO_FILES = {
    "CPI": ("CPI.csv", 45),
    "DGS10": ("DGS10.csv", 1),
    "USD_INDEX": ("DTWEXBGS.csv", 1),
    "FEDFUNDS": ("FEDFUNDS.csv", 30),
    "GOLD": ("GOLD.csv", 30),
    "NFP": ("NFP.csv", 45),
    "OIL": ("OIL.csv", 1),
    "PCEPI": ("PCEPI.csv", 45),
    "PPI": ("PPI.csv", 45),
    "REALGDP": ("REALGDP.csv", 120),
    "UNEMPLOYRATE": ("UNEMPLOYRATE.csv", 45),
    "VIX": ("VIX.csv", 1),
}


def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip().replace("\ufeff", "")
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def as_float(value: str) -> Optional[float]:
    try:
        value = (value or "").strip()
        if not value or value in {".", "NA", "N/A", "nan", "NaN"}:
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def read_dict_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_col(row: dict, names: List[str]) -> Optional[str]:
    lookup = {str(k).strip().lower().replace(" ", "_"): k for k in row.keys()}
    for name in names:
        key = lookup.get(name.lower().replace(" ", "_"))
        if key is not None:
            return key
    return None


def ensure_market_file() -> str:
    candidates = [
        os.path.join(DATA_DIR, "spy_data.csv"),
        os.path.join(DATA_DIR, "SPY.csv"),
        os.path.join(DATA_DIR, "spy.csv"),
        os.path.join(DATA_DIR, "SPX500_1d.csv"),
        os.path.join(DATA_DIR, "sp500_sample.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            if os.path.basename(path).lower() not in {"spy_data.csv", "spy.csv", "spy.csv"}:
                print(f"NOTICE: data/spy_data.csv not found; using {os.path.relpath(path, ROOT)} instead.")
            return path
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        print("Downloading spy_data.csv from the configured GitHub raw URL...")
        urllib.request.urlretrieve(SPY_URL, os.path.join(DATA_DIR, "spy_data.csv"))
        return os.path.join(DATA_DIR, "spy_data.csv")
    except Exception as exc:
        raise FileNotFoundError(
            "Could not find data/spy_data.csv and automatic download failed. "
            "Download the CSV from the project repository and place it at data/spy_data.csv. "
            f"Download error: {exc}"
        )


def load_market(path: str) -> List[dict]:
    raw = read_dict_csv(path)
    if not raw:
        raise ValueError(f"Market data is empty: {path}")
    sample = raw[0]
    date_col = find_col(sample, ["date", "datetime", "timestamp", "observation_date"])
    open_col = find_col(sample, ["open"])
    high_col = find_col(sample, ["high"])
    low_col = find_col(sample, ["low"])
    close_col = find_col(sample, ["close", "adj_close", "adjclose"])
    volume_col = find_col(sample, ["volume", "vol"])
    if not all([date_col, open_col, high_col, low_col, close_col]):
        raise ValueError(
            f"Unsupported market CSV columns: {list(sample.keys())}. "
            "Expected date/open/high/low/close and optional volume."
        )
    out = []
    for row in raw:
        d = parse_date(row.get(date_col, ""))
        o, h, l, c = [as_float(row.get(col, "")) for col in (open_col, high_col, low_col, close_col)]
        v = as_float(row.get(volume_col, "")) if volume_col else 0.0
        if d and all(x is not None for x in (o, h, l, c)) and c > 0 and o > 0:
            out.append({"date": d, "open": o, "high": h, "low": l, "close": c, "volume": max(0.0, v or 0.0)})
    out.sort(key=lambda x: x["date"])
    dedup = {}
    for row in out:
        dedup[row["date"]] = row
    out = [dedup[d] for d in sorted(dedup)]
    if len(out) < 30:
        raise ValueError(f"Only {len(out)} valid market rows; at least 30 are required.")
    return out


def load_macro_series() -> Dict[str, List[Tuple[date, float]]]:
    series = {}
    for feature_name, (filename, lag_days) in MACRO_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            print(f"NOTICE: macro file missing, skipping {filename}")
            continue
        raw = read_dict_csv(path)
        if not raw:
            continue
        sample = raw[0]
        dcol = find_col(sample, ["observation_date", "date", "datetime"])
        value_col = next((k for k in sample.keys() if k != dcol), None)
        if not dcol or not value_col:
            print(f"NOTICE: could not identify columns in {filename}; skipping it.")
            continue
        vals = []
        for row in raw:
            d = parse_date(row.get(dcol, ""))
            value = as_float(row.get(value_col, ""))
            if d is not None and value is not None:
                available = d + timedelta(days=lag_days)
                vals.append((available, value))
        vals.sort(key=lambda x: x[0])
        if vals:
            # Keep the last value for duplicate available dates.
            dedup = {}
            for d, v in vals:
                dedup[d] = v
            series[feature_name] = sorted(dedup.items())
    return series


def latest_known(series: List[Tuple[date, float]], d: date, pointer: int, previous: Optional[float]):
    while pointer < len(series) and series[pointer][0] <= d:
        previous = series[pointer][1]
        pointer += 1
    return pointer, previous


def build_dataset(market: List[dict], macro: Dict[str, List[Tuple[date, float]]]):
    macro_names = list(macro.keys())
    pointers = {name: 0 for name in macro_names}
    previous = {name: None for name in macro_names}
    rows = []
    for i in range(5, len(market) - 1):
        cur = market[i]
        prev = market[i - 1]
        five = market[i - 5]
        for name in macro_names:
            pointers[name], previous[name] = latest_known(
                macro[name], cur["date"], pointers[name], previous[name]
            )
        # Do not create fake early macro values: wait until all selected series are available.
        if macro_names and any(previous[name] is None for name in macro_names):
            continue
        close, open_, high, low = cur["close"], cur["open"], cur["high"], cur["low"]
        features = [
            (close - prev["close"]) / prev["close"],
            (close - five["close"]) / five["close"],
            (close - open_) / open_,
            (high - low) / close,
            math.log(cur["volume"] + 1.0),
        ]
        features.extend(previous[name] for name in macro_names)
        next_row = market[i + 1]
        target = (next_row["close"] - close) / close
        rows.append({
            "date": cur["date"], "target_date": next_row["date"],
            "features": features, "target": target,
        })
    if len(rows) < 50:
        raise ValueError(
            f"Only {len(rows)} rows remain after feature alignment. Check the market and macro CSV files."
        )
    return rows, ["return_1d", "momentum_5d", "intraday_return", "range_pct", "log_volume"] + macro_names


def standardize_fit(X: List[List[float]]):
    p = len(X[0])
    means, stds = [], []
    for j in range(p):
        col = [r[j] for r in X]
        mu = sum(col) / len(col)
        var = sum((v - mu) ** 2 for v in col) / max(1, len(col) - 1)
        sd = math.sqrt(var)
        means.append(mu)
        stds.append(sd if sd > 1e-12 else 1.0)
    Z = [[(r[j] - means[j]) / stds[j] for j in range(p)] for r in X]
    return Z, means, stds


def standardize_apply(X: List[List[float]], means: List[float], stds: List[float]):
    return [[(r[j] - means[j]) / stds[j] for j in range(len(means))] for r in X]


def transpose(A):
    return [list(row) for row in zip(*A)] if A else []


def matmul(A, B):
    if not A or not B or len(A[0]) != len(B):
        raise ValueError("Matrix dimension mismatch")
    BT = transpose(B)
    return [[sum(x * y for x, y in zip(row, col)) for col in BT] for row in A]


def jacobi_eigen_symmetric(A, tolerance=1e-10, max_iter=10000):
    n = len(A)
    M = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    if n == 1:
        return [M[0][0]], V
    for _ in range(max_iter):
        p, q, largest = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(M[i][j]) > largest:
                    p, q, largest = i, j, abs(M[i][j])
        if largest < tolerance:
            break
        if abs(M[p][q]) < tolerance:
            continue
        tau = (M[q][q] - M[p][p]) / (2.0 * M[p][q])
        t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        app, aqq, apq = M[p][p], M[q][q], M[p][q]
        M[p][p] = app - t * apq
        M[q][q] = aqq + t * apq
        M[p][q] = M[q][p] = 0.0
        for k in range(n):
            if k != p and k != q:
                mkp, mkq = M[k][p], M[k][q]
                M[k][p] = M[p][k] = c * mkp - s * mkq
                M[k][q] = M[q][k] = s * mkp + c * mkq
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq
    eigenvalues = [M[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigenvalues[i], reverse=True)
    vals = [max(0.0, eigenvalues[i]) for i in order]
    vecs = [[V[r][i] for i in order] for r in range(n)]
    return vals, vecs


def pca_fit(Z: List[List[float]], components: int):
    n, p = len(Z), len(Z[0])
    cov = [[sum(row[i] * row[j] for row in Z) / max(1, n - 1) for j in range(p)] for i in range(p)]
    values, vectors = jacobi_eigen_symmetric(cov)
    k = max(1, min(components, p))
    W = [[vectors[i][j] for j in range(k)] for i in range(p)]
    total = sum(values)
    explained = [values[j] / total if total > 0 else 0.0 for j in range(k)]
    return W, explained


def project(Z, W):
    return matmul(Z, W)


def determinant(A):
    n = len(A)
    if n == 0:
        return 1.0
    if n == 1:
        return A[0][0]
    total = 0.0
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in A[1:]]
        total += ((-1.0) ** j) * A[0][j] * determinant(minor)
    return total


def inverse_adjugate(A):
    n = len(A)
    det = determinant(A)
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix in adjugate inverse")
    cof = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = [A[r][:j] + A[r][j + 1:] for r in range(n) if r != i]
            row.append(((-1.0) ** (i + j)) * determinant(minor))
        cof.append(row)
    adj = transpose(cof)
    return [[v / det for v in row] for row in adj]


def solve_gaussian(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("Singular system")
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def inverse_gauss_jordan(A):
    n = len(A)
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix")
        M[col], M[pivot] = M[pivot], M[col]
        scale = M[col][col]
        M[col] = [v / scale for v in M[col]]
        for r in range(n):
            if r != col:
                factor = M[r][col]
                M[r] = [M[r][j] - factor * M[col][j] for j in range(2 * n)]
    return [row[n:] for row in M]


def solve_lu(A, b):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        if abs(U[i][i]) < 1e-12:
            raise ValueError("LU zero pivot; matrix may be singular")
        L[i][i] = 1.0
        for k in range(i + 1, n):
            L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def fit_regression(X, y, method, ridge=1e-8):
    # X already contains an intercept column. Use a tiny ridge term on slopes
    # to stabilize the normal equations; do not penalize the intercept.
    XT = transpose(X)
    A = matmul(XT, X)
    bmat = matmul(XT, [[v] for v in y])
    A = [row[:] for row in A]
    for i in range(1, len(A)):
        A[i][i] += ridge
    b = [row[0] for row in bmat]
    if method == "Gaussian":
        return solve_gaussian(A, b)
    if method == "Gauss-Jordan":
        inv = inverse_gauss_jordan(A)
        return [sum(inv[i][j] * b[j] for j in range(len(b))) for i in range(len(b))]
    if method == "LU":
        return solve_lu(A, b)
    if method == "Adjugate":
        inv = inverse_adjugate(A)
        return [sum(inv[i][j] * b[j] for j in range(len(b))) for i in range(len(b))]
    if method == "Least Squares":
        # Ordinary least squares via the normal equations, solved by Gaussian elimination.
        return solve_gaussian(A, b)
    if method == "SVD":
        # Pseudoinverse-equivalent solution using the eigensystem of X^T X.
        # For symmetric X^T X, its eigenvectors are right singular vectors;
        # very small eigenvalues are truncated to avoid unstable division.
        vals, vecs = jacobi_eigen_symmetric(A)
        beta = [0.0] * len(b)
        maxv = max(vals) if vals else 0.0
        for k, lam in enumerate(vals):
            if lam <= max(1e-12, maxv * 1e-10):
                continue
            v = [vecs[i][k] for i in range(len(b))]
            weight = sum(v[i] * b[i] for i in range(len(b))) / lam
            for i in range(len(b)):
                beta[i] += v[i] * weight
        return beta
    raise ValueError(f"Unknown method: {method}")


def add_intercept(X):
    return [[1.0] + row[:] for row in X]


def predict(X, beta):
    return [sum(a * b for a, b in zip(row, beta)) for row in X]


def metrics(actual, predicted):
    n = len(actual)
    if not n:
        return {"MSE": float("nan"), "MAE": float("nan"), "Direction_Accuracy": float("nan"), "Correlation": float("nan")}
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
    da = sum(1 for a, p in zip(actual, predicted) if (a >= 0) == (p >= 0)) / n
    am, pm = sum(actual) / n, sum(predicted) / n
    cov = sum((a - am) * (p - pm) for a, p in zip(actual, predicted))
    av = sum((a - am) ** 2 for a in actual)
    pv = sum((p - pm) ** 2 for p in predicted)
    corr = cov / math.sqrt(av * pv) if av > 0 and pv > 0 else 0.0
    return {"MSE": mse, "MAE": mae, "Direction_Accuracy": da, "Correlation": corr}


def make_rounds(rows):
    years = sorted({r["date"].year for r in rows})
    # Preferred setup: train on seven calendar years and test the next year.
    if len(years) >= 9:
        rounds = []
        for test_year in range(years[0] + 7, years[-1] + 1):
            train_start = date(test_year - 7, 1, 1)
            test_start = date(test_year, 1, 1)
            test_end = date(test_year + 1, 1, 1)
            train = [r for r in rows if train_start <= r["date"] < test_start and r["target_date"] < test_start]
            test = [r for r in rows if test_start <= r["date"] < test_end]
            if len(train) >= 100 and len(test) >= 20:
                rounds.append((f"{test_year}", train, test))
        if rounds:
            return rounds, "7-year rolling training / 1-year testing"
    # Fallback for short local data: 252 rows training and 63 rows per test block.
    # This makes the code runnable, but is not a substitute for the 7-year study.
    rounds = []
    train_size, test_size = 252, 63
    start = train_size
    round_no = 1
    while start < len(rows):
        end = min(start + test_size, len(rows))
        train = rows[max(0, start - train_size):start]
        test = rows[start:end]
        if len(train) >= 100 and len(test) >= 20:
            label = f"short-data-round-{round_no}"
            rounds.append((label, train, test))
        start = end
        round_no += 1
    return rounds, "SHORT-DATA FALLBACK: 252-row training / 63-row testing"


def write_csv(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def run_backtest():
    os.makedirs(OUT_DIR, exist_ok=True)
    market_path = ensure_market_file()
    market = load_market(market_path)
    macro = load_macro_series()
    dataset, feature_names = build_dataset(market, macro)
    rounds, mode = make_rounds(dataset)
    if not rounds:
        raise ValueError(
            f"Not enough data for a backtest ({len(dataset)} aligned rows). "
            "Use a longer historical spy_data.csv. For a 7-year train / 1-year test, at least 8 years are needed."
        )
    methods = ["Gaussian", "Gauss-Jordan", "LU", "Adjugate", "Least Squares", "SVD"]
    pred_rows, metric_rows, pca_rows = [], [], []
    errors = []
    components = min(5, len(feature_names))
    for round_name, train, test in rounds:
        print(f"Running {round_name}: train={len(train)} test={len(test)}")
        Xraw = [r["features"] for r in train]
        ytrain = [r["target"] for r in train]
        Xtest_raw = [r["features"] for r in test]
        ytest = [r["target"] for r in test]
        Ztrain, means, stds = standardize_fit(Xraw)
        Ztest = standardize_apply(Xtest_raw, means, stds)
        W, explained = pca_fit(Ztrain, components)
        Ptrain = add_intercept(project(Ztrain, W))
        Ptest = add_intercept(project(Ztest, W))
        pca_rows.append([round_name, len(feature_names), components, sum(explained)] + explained)
        for method in methods:
            try:
                beta = fit_regression(Ptrain, ytrain, method)
                pred = predict(Ptest, beta)
                met = metrics(ytest, pred)
                metric_rows.append([round_name, method, len(train), len(test), met["MSE"], met["MAE"], met["Direction_Accuracy"], met["Correlation"]])
                for row, actual, estimate in zip(test, ytest, pred):
                    pred_rows.append([round_name, method, row["date"].isoformat(), row["target_date"].isoformat(), actual, estimate, "UP" if estimate >= 0 else "DOWN"])
            except Exception as exc:
                errors.append([round_name, method, str(exc)])
                print(f"  WARNING {method} failed in {round_name}: {exc}")
    write_csv(os.path.join(OUT_DIR, "predictions.csv"),
              ["round", "method", "feature_date", "target_date", "actual_return", "predicted_return", "predicted_direction"], pred_rows)
    write_csv(os.path.join(OUT_DIR, "metrics_by_round.csv"),
              ["round", "method", "train_rows", "test_rows", "MSE", "MAE", "Direction_Accuracy", "Correlation"], metric_rows)
    # Average each method over its out-of-sample rounds (rounds are equally weighted here).
    summary = []
    for method in methods:
        rows = [r for r in metric_rows if r[1] == method]
        if rows:
            summary.append([method, len(rows), sum(float(r[4]) for r in rows) / len(rows),
                            sum(float(r[5]) for r in rows) / len(rows),
                            sum(float(r[6]) for r in rows) / len(rows),
                            sum(float(r[7]) for r in rows) / len(rows)])
    write_csv(os.path.join(OUT_DIR, "method_comparison.csv"),
              ["method", "rounds", "mean_MSE", "mean_MAE", "mean_Direction_Accuracy", "mean_Correlation"], summary)
    pca_headers = ["round", "original_features", "components", "explained_variance_ratio_sum"] + [f"PC{i+1}_variance_ratio" for i in range(components)]
    write_csv(os.path.join(OUT_DIR, "pca_summary.csv"), pca_headers, pca_rows)
    write_csv(os.path.join(OUT_DIR, "errors.csv"), ["round", "method", "error"], errors)
    with open(os.path.join(OUT_DIR, "README_RESULTS.txt"), "w", encoding="utf-8") as f:
        f.write("S&P 500 Matrix Model - Walk-forward Backtest\n")
        f.write("=============================================\n")
        f.write(f"Market file: {os.path.relpath(market_path, ROOT)}\n")
        f.write(f"Market rows: {len(market)}\nAligned feature rows: {len(dataset)}\n")
        f.write(f"Features: {', '.join(feature_names)}\n")
        f.write(f"Backtest mode: {mode}\nRounds completed: {len(rounds)}\n")
        f.write("PCA and standardization are fitted on each training set only.\n")
        f.write("Macro release lags are fixed approximations, not exact release-date/vintage data.\n")
        f.write("Metrics are historical out-of-sample evaluation, not a forecast guarantee or trading advice.\n")
        if "SHORT-DATA" in mode:
            f.write("WARNING: Full 7-year/1-year evaluation was not possible with available history.\n")
        if errors:
            f.write(f"Methods with errors: {len(errors)}; see errors.csv.\n")
    print("\nBacktest finished.")
    print(f"Mode: {mode}")
    print(f"Rounds: {len(rounds)} | aligned rows: {len(dataset)}")
    print(f"Results: {os.path.relpath(OUT_DIR, ROOT)}")
    print("Files: predictions.csv, metrics_by_round.csv, method_comparison.csv, pca_summary.csv, errors.csv")
    if "SHORT-DATA" in mode:
        print("WARNING: this is a short-data fallback, not the requested 7-year/1-year backtest.")
    return metric_rows, summary
