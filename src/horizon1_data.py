"""Data loading helpers for the Horizon-1 S&P 500 score model."""
from __future__ import annotations
import csv, math, os
from datetime import date

INDICATORS = ["CPI", "DTWEXBGS", "FEDFUNDS", "GOLD", "NFP", "OIL", "PCEPI", "PPI", "REALGDP", "UNEMPLOYRATE", "VIX"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        s = str(value or "").strip()
        if not s or s.lower() in {"nan", "none", "null", "na", "n/a", "."}:
            return None
        value = float(s)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def parse_date(value):
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except (TypeError, ValueError):
        return None


def load_weights(path):
    weights = {}
    for row in read_csv(path):
        window = str(row.get("window", "")).strip()
        indicator = str(row.get("indicator", "")).strip()
        weight = as_float(row.get("weight"))
        if window and indicator and weight is not None:
            weights.setdefault(window, {})[indicator] = weight
    return weights


def compute_score(row, weights):
    """Recalculate score with weights from the training window, never test-window weights."""
    total = 0.0
    used = 0
    for indicator in INDICATORS:
        value = as_float(row.get(indicator))
        weight = weights.get(indicator)
        if value is None or weight is None:
            return None
        total += value * weight
        used += 1
    return total if used == len(INDICATORS) else None


def realized_return_pct(row):
    open_price = as_float(row.get("Open"))
    future_open = as_float(row.get("Future_Open"))
    current_date = parse_date(row.get("Date"))
    future_date = parse_date(row.get("Future_Date"))
    if open_price is None or future_open is None or open_price <= 0 or future_open <= 0:
        return None
    if current_date is None or future_date is None or future_date <= current_date:
        return None
    return (future_open / open_price - 1.0) * 100.0


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
