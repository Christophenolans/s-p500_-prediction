"""Score calibration and next-session return predictions."""
from __future__ import annotations
import math
from collections import defaultdict
from .horizon1_data import as_float, parse_date, compute_score, realized_return_pct

BIN_SIZE = 0.1


def score_bin(score):
    return round(math.floor(score / BIN_SIZE) * BIN_SIZE, 10)


def build_labeled_rows(rows, weights, target_before=None):
    """Build training examples and exclude labels whose target date reaches test period."""
    out = []
    for row in rows:
        d = parse_date(row.get("Date"))
        target_date = parse_date(row.get("Future_Date"))
        score = compute_score(row, weights)
        realized = realized_return_pct(row)
        if target_before is not None and (target_date is None or target_date >= target_before):
            continue
        if d and score is not None and realized is not None:
            out.append({"date": d, "score": score, "return_pct": realized})
    return out


def fit_calibration(train_rows, min_bin_count=10):
    """Fit score-bin means and a linear fallback using training observations only."""
    bins = defaultdict(list)
    for row in train_rows:
        bins[score_bin(row["score"])].append(row["return_pct"])
    bin_model = {}
    for key, values in bins.items():
        bin_model[key] = {
            "n": len(values),
            "mean_return_pct": sum(values) / len(values),
            "probability_up_pct": 100.0 * sum(v > 0 for v in values) / len(values),
            "std_return_pct": math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / max(1, len(values)-1)) if len(values) > 1 else 0.0,
        }
    if not train_rows:
        return {"bins": bin_model, "min_bin_count": min_bin_count, "intercept": 0.0, "slope": 0.0, "global_mean": 0.0, "global_up_pct": 50.0, "n": 0}
    xs = [r["score"] for r in train_rows]
    ys = [r["return_pct"] for r in train_rows]
    xm, ym = sum(xs)/len(xs), sum(ys)/len(ys)
    denom = sum((x-xm)**2 for x in xs)
    slope = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/denom if denom > 1e-15 else 0.0
    return {
        "bins": bin_model, "min_bin_count": min_bin_count,
        "intercept": ym - slope*xm, "slope": slope,
        "global_mean": ym, "global_up_pct": 100.0*sum(y > 0 for y in ys)/len(ys), "n": len(ys)
    }


def predict_one(score, model):
    key = score_bin(score)
    b = model["bins"].get(key)
    if b and b["n"] >= model["min_bin_count"]:
        return b["mean_return_pct"], b["probability_up_pct"], b["n"], "SCORE_BIN"
    estimate = model["intercept"] + model["slope"] * score
    return estimate, model["global_up_pct"], model["n"], "LINEAR_FALLBACK"


def predict_rows(test_rows, weights, model, train_window, test_year, threshold_pct=0.0):
    predictions = []
    for row in test_rows:
        d = parse_date(row.get("Date"))
        if not d or d.year != test_year:
            continue
        score = compute_score(row, weights)
        realized = realized_return_pct(row)
        open_price = as_float(row.get("Open"))
        future_open = as_float(row.get("Future_Open"))
        future_date = parse_date(row.get("Future_Date"))
        if score is None or realized is None or open_price is None or future_open is None:
            continue
        expected, prob_up, n_cal, method = predict_one(score, model)
        if expected > threshold_pct:
            signal = "BUY/HOLD"
            position = 1
        else:
            signal = "CASH"
            position = 0
        predictions.append({
            "train_window": train_window,
            "test_year": test_year,
            "date": d.isoformat(),
            "target_date": future_date.isoformat() if future_date else "",
            "open_price": open_price,
            "future_open_price": future_open,
            "score": score,
            "predicted_return_pct": expected,
            "predicted_probability_up_pct": prob_up,
            "calibration_samples": n_cal,
            "prediction_method": method,
            "signal": signal,
            "position": position,
            "actual_return_pct": realized,
            "direction_correct": int((expected >= 0) == (realized >= 0)),
        })
    return predictions
