
import csv
import os
import re
import math
from datetime import datetime

# =========================================================
# 1. SETTINGS
# =========================================================

# Main folders
BASE_DIR = r"D:\Deengam\linear project"
INDICATOR_DIR = os.path.join(BASE_DIR, "Indicator data")
CODE_DIR = os.path.join(BASE_DIR, "code")

# Input files
SPY_FILE = os.path.join(INDICATOR_DIR, "spy_data.csv")
WEIGHT_FILE = os.path.join(CODE_DIR, "weighted_correlation_lag-14.csv")

# Output files
OUTPUT_DAILY = os.path.join(CODE_DIR, "spy_daily_score.csv")
OUTPUT_STATS = os.path.join(CODE_DIR, "spy_probability_by_score.csv")


INDICATORS = [
    "CPI",
    "DTWEXBGS",
    "FEDFUNDS",
    "GOLD",
    "NFP",
    "OIL",
    "PCEPI",
    "PPI",
    "REALGDP",
    "UNEMPLOYRATE",
    "VIX"
]

TARGET_LAG = -14
HORIZON = 14
BIN_SIZE = 0.1


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def parse_date(value):
    """Support common CSV date formats."""
    value = value.strip()

    formats = [
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    return None


def parse_number(value):
    """Convert a CSV value into float."""
    if value is None:
        return None

    value = value.strip()

    if value == "" or value.lower() in {
        "nan", "none", "null", "n/a", "na"
    }:
        return None

    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except ValueError:
        return None


def parse_lag(value):
    number = parse_number(value)
    if number is None:
        return None
    return int(number)


def get_window_years(window):
    """
    Extract years from a window such as:
    2000-2006
    2007-2013
    """
    years = re.findall(r"\d{4}", window)

    if len(years) >= 2:
        return int(years[0]), int(years[1])

    return None


def find_window(date, window_names):
    """
    Find the window containing the given date.
    If windows overlap, choose the one with the latest start year.
    """
    matches = []

    for name in window_names:
        years = get_window_years(name)

        if years is None:
            continue

        start_year, end_year = years

        if start_year <= date.year <= end_year:
            matches.append((start_year, end_year, name))

    if not matches:
        return None

    matches.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return matches[0][2]


# =========================================================
# 3. LOAD SPY OPEN PRICES
# =========================================================

def load_spy(filename):
    """
    Example input:
    Price,Close,High,Low,Open,Volume
    Ticker,SPY,SPY,SPY,SPY,SPY
    Date,,,,,
    1/3/2000,92.14,93.92,91.15,93.92,8164300

    The actual Open value is column index 4.
    """
    spy = []

    with open(filename, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        rows = list(reader)

    for row in rows[3:]:
        if len(row) < 5:
            continue

        date = parse_date(row[0])

        if date is None:
            continue

        open_price = parse_number(row[4])

        if open_price is None or open_price <= 0:
            continue

        spy.append({
            "date": date,
            "open": open_price
        })

    spy.sort(key=lambda item: item["date"])

    # Remove duplicate dates, keeping the last valid record.
    unique = {}

    for row in spy:
        unique[row["date"]] = row["open"]

    return [
        {"date": date, "open": open_price}
        for date, open_price in sorted(unique.items())
    ]


# =========================================================
# 4. LOAD INDICATOR CSV FILES
# =========================================================

def load_indicator(filename):
    """
    Reads a CSV whose first column is the date.
    Uses the first non-empty numeric field after the date
    as the indicator value.

    Blank/missing observations are skipped. Later, the
    latest available past value is carried forward.
    """
    observations = {}

    with open(filename, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        rows = list(reader)

    for row in rows[1:]:
        if len(row) < 2:
            continue

        date = parse_date(row[0])

        if date is None:
            continue

        value = None

        for cell in row[1:]:
            number = parse_number(cell)

            if number is not None:
                value = number
                break

        if value is not None:
            observations[date] = value

    return sorted(observations.items())


# =========================================================
# 5. LOAD WEIGHTS FROM weighted_correlation_lag-14.csv
# =========================================================

def load_weights(filename):
    """
    Expected columns:
    window, lag, indicator, correlation,
    sum_abs_correlation, weight

    Use the existing weight column directly.
    """
    weights = {}

    with open(filename, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"window", "lag", "indicator", "weight"}
        available = set(reader.fieldnames or [])

        if not required.issubset(available):
            raise ValueError(
                "Weight CSV must contain columns: "
                "window, lag, indicator, weight"
            )

        for row in reader:
            lag = parse_lag(row.get("lag", ""))

            if lag != TARGET_LAG:
                continue

            window = row["window"].strip()
            indicator = row["indicator"].strip().upper()
            weight = parse_number(row.get("weight", ""))

            if weight is None:
                continue

            weights[(window, indicator)] = weight

    return weights


# =========================================================
# 6. ALIGN INDICATORS TO SPY TRADING DATES
# =========================================================

def align_indicators(spy, indicator_data):
    """
    For each SPY trading date, use the most recent
    indicator observation dated on or before that date.

    Never use a future indicator observation.
    """
    aligned = {}

    for indicator, observations in indicator_data.items():
        values_by_date = {}
        pointer = 0
        latest_value = None

        for spy_row in spy:
            spy_date = spy_row["date"]

            while (
                pointer < len(observations)
                and observations[pointer][0] <= spy_date
            ):
                latest_value = observations[pointer][1]
                pointer += 1

            values_by_date[spy_date] = latest_value

        aligned[indicator] = values_by_date

    return aligned


# =========================================================
# 7. CALCULATE DAILY SCORE AND OPEN-TO-OPEN RETURN
# =========================================================

def calculate_daily_scores(spy, aligned, weights):
    results = []
    window_names = sorted(set(key[0] for key in weights))

    for i, spy_row in enumerate(spy):
        date = spy_row["date"]
        open_price = spy_row["open"]

        window = find_window(date, window_names)

        indicator_values = {}
        score = 0.0
        complete = True

        if window is None:
            complete = False

        for indicator in INDICATORS:
            value = aligned[indicator].get(date)

            indicator_values[indicator] = value

            weight = weights.get((window, indicator)) if window else None

            # Require every indicator and its weight.
            if value is None or weight is None:
                complete = False
            else:
                score += weight * value

        # Return measured after 14 subsequent SPY trading sessions.
        future_open = None
        future_date = None
        future_return = None

        future_index = i + HORIZON

        if future_index < len(spy):
            future_row = spy[future_index]
            future_open = future_row["open"]
            future_date = future_row["date"]

            future_return = (
                (future_open / open_price) - 1
            ) * 100

        results.append({
            "date": date,
            "open": open_price,
            "window": window,
            "score": score if complete else None,
            "indicator_values": indicator_values,
            "future_date": future_date,
            "future_open": future_open,
            "future_return": future_return
        })

    return results


# =========================================================
# 8. SAVE DAILY RESULTS
# =========================================================

def save_daily_results(results, filename):
    headers = [
        "Date",
        "Open",
        "Window",
        "Score"
    ]

    headers += INDICATORS

    headers += [
        "Future_Date",
        "Future_Open",
        "Return_14_Trading_Days_Percent"
    ]

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for row in results:
            output = {
                "Date": row["date"].isoformat(),
                "Open": row["open"],
                "Window": row["window"] or "",
                "Score": (
                    row["score"] if row["score"] is not None else ""
                ),
                "Future_Date": (
                    row["future_date"].isoformat()
                    if row["future_date"] else ""
                ),
                "Future_Open": (
                    row["future_open"]
                    if row["future_open"] is not None else ""
                ),
                "Return_14_Trading_Days_Percent": (
                    row["future_return"]
                    if row["future_return"] is not None else ""
                )
            }

            for indicator in INDICATORS:
                value = row["indicator_values"][indicator]
                output[indicator] = value if value is not None else ""

            writer.writerow(output)


# =========================================================
# 9. GROUP SCORES INTO 0.1-WIDE BINS
# =========================================================

def calculate_probability(results):
    groups = {}

    for row in results:
        score = row["score"]
        ret = row["future_return"]

        # Skip rows without a complete score or future return.
        if score is None or ret is None:
            continue

        # Bins are [lower, upper), for example [0.2, 0.3).
        bin_lower = math.floor(score / BIN_SIZE) * BIN_SIZE
        bin_lower = round(bin_lower, 10)

        if bin_lower not in groups:
            groups[bin_lower] = {
                "count": 0,
                "up_count": 0,
                "returns": []
            }

        group = groups[bin_lower]
        group["count"] += 1
        group["returns"].append(ret)

        if ret > 0:
            group["up_count"] += 1

    output = []

    for bin_lower in sorted(groups):
        group = groups[bin_lower]
        returns = group["returns"]
        n = group["count"]

        mean_return = sum(returns) / n

        # Sample variance: sum((x - mean)^2) / (n - 1).
        if n >= 2:
            return_variance = sum(
                (value - mean_return) ** 2 for value in returns
            ) / (n - 1)

            variance_of_mean = return_variance / n
            standard_error = math.sqrt(variance_of_mean)
        else:
            return_variance = None
            variance_of_mean = None
            standard_error = None

        probability_up = group["up_count"] / n * 100

        output.append({
            "Score_Bin_Lower": bin_lower,
            "Score_Bin_Upper": round(bin_lower + BIN_SIZE, 10),
            "Count": n,
            "Up_Count": group["up_count"],
            "Probability_Up_Percent": probability_up,
            "Mean_Return_Percent": mean_return,
            "Return_Variance": return_variance,
            "Variance_Of_Mean": variance_of_mean,
            "Standard_Error": standard_error
        })

    return output


# =========================================================
# 10. SAVE PROBABILITY AND VARIANCE RESULTS
# =========================================================

def save_probability_results(results, filename):
    headers = [
        "Score_Bin_Lower",
        "Score_Bin_Upper",
        "Count",
        "Up_Count",
        "Probability_Up_Percent",
        "Mean_Return_Percent",
        "Return_Variance",
        "Variance_Of_Mean",
        "Standard_Error"
    ]

    with open(filename, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


# =========================================================
# 11. MAIN
# =========================================================

def main():
    print("Loading SPY data...")
    spy = load_spy(SPY_FILE)

    if not spy:
        raise ValueError("No valid SPY rows found. Check spy_data.csv.")

    print("Loading weights...")
    weights = load_weights(WEIGHT_FILE)

    if not weights:
        raise ValueError(
            "No weights found for lag = " + str(TARGET_LAG)
        )

    print("Loading indicator data...")
    indicator_data = {}

    for indicator in INDICATORS:
        filename = os.path.join(INDICATOR_DIR, indicator + ".csv")

        if not os.path.exists(filename):
            raise FileNotFoundError(
                "Indicator file not found: " + filename
            )

        indicator_data[indicator] = load_indicator(filename)

        if not indicator_data[indicator]:
            raise ValueError(
                "No valid observations in " + filename
            )

    print("Aligning indicator dates...")
    aligned = align_indicators(spy, indicator_data)

    print("Calculating daily scores...")
    daily_results = calculate_daily_scores(
        spy, aligned, weights
    )

    print("Saving daily results...")
    save_daily_results(daily_results, OUTPUT_DAILY)

    print("Calculating probability and variance...")
    probability_results = calculate_probability(daily_results)

    print("Saving probability results...")
    save_probability_results(probability_results, OUTPUT_STATS)

    valid_scores = sum(
        1 for row in daily_results if row["score"] is not None
    )

    valid_returns = sum(
        1 for row in daily_results if row["future_return"] is not None
    )

    print("\nFinished!")
    print("SPY trading days:", len(spy))
    print("Rows with complete scores:", valid_scores)
    print("Rows with 14-day future returns:", valid_returns)
    print("Daily output:", OUTPUT_DAILY)
    print("Probability output:", OUTPUT_STATS)


if __name__ == "__main__":
    main()
