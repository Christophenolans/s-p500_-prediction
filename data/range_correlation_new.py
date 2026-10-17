
import csv
import os
import math
from datetime import datetime


# =========================================================
# 1. SETTINGS
# =========================================================

# Main folder
BASE_DIR = r"D:\Deengam\linear project"

# Input folders
NORMALIZED_DIR = os.path.join(BASE_DIR, "Normalized data")
INDICATOR_DIR = os.path.join(BASE_DIR, "Indicator data")
CODE_DIR = os.path.join(BASE_DIR, "code")

# Input files
SPY_FILE = os.path.join(INDICATOR_DIR, "spy_data.csv")

WEIGHT_FILE = os.path.join(
    CODE_DIR,
    "weighted_correlation_lag3.csv"
)

# Output folder
OUTPUT_DIR = os.path.join(CODE_DIR, "results 3 horizon 14")

# Rolling windows
START_YEAR = 2006
END_YEAR = 2025
WINDOW_SIZE = 7

# Model settings
TARGET_LAG = 3
HORIZON = 14
BIN_SIZE = 0.1

# Indicators
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


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def parse_date(value):
    """Convert a supported date string to date."""

    if value is None:
        return None

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
    """Convert a CSV value to a finite float."""

    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in {
        "", "nan", "none", "null", "n/a", "na"
    }:
        return None

    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (ValueError, TypeError):
        return None


def parse_lag(value):
    """Convert a lag value to an integer."""

    number = parse_number(value)

    if number is None:
        return None

    return int(number)


# =========================================================
# 3. LOAD SPY OPEN PRICES
# =========================================================

def load_spy(filename):
    """
    Load SPY data.

    Expected columns:
    Price, Close, High, Low, Open, Volume
    Ticker, SPY, SPY, SPY, SPY, SPY
    Date, ...
    """

    spy_by_date = {}

    with open(
        filename, "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) < 5:
                continue

            date = parse_date(row[0])

            if date is None:
                continue

            # Open is column index 4
            open_price = parse_number(row[4])

            if open_price is None or open_price <= 0:
                continue

            spy_by_date[date] = open_price

    spy = [
        {"date": date, "open": open_price}
        for date, open_price in sorted(spy_by_date.items())
    ]

    return spy


# =========================================================
# 4. LOAD NORMALIZED INDICATOR FILES
# =========================================================

def load_indicator(filename):
    """
    Load normalized indicator data.

    Expected columns:
    Date, Original_Value, Normalized_Value

    IMPORTANT:
    Use Normalized_Value, not Original_Value.
    """

    observations = {}

    with open(
        filename, "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return []

        # Find the date column, ignoring capitalization
        date_column = next(
            (
                col for col in reader.fieldnames
                if col.strip().lower() == "date"
            ),
            None
        )

        # Find Normalized_Value
        normalized_column = next(
            (
                col for col in reader.fieldnames
                if col.strip().lower() == "normalized_value"
            ),
            None
        )

        # Fallback if the file does not have Normalized_Value
        if date_column is None:
            raise ValueError(
                f"No Date column found in {filename}"
            )

        value_column = normalized_column

        if value_column is None:
            value_column = next(
                (
                    col for col in reader.fieldnames
                    if col.strip().lower() != "date"
                ),
                None
            )

        if value_column is None:
            raise ValueError(
                f"No indicator value column found in {filename}"
            )

        for row in reader:

            date = parse_date(row.get(date_column, ""))

            if date is None:
                continue

            value = parse_number(
                row.get(value_column, "")
            )

            if value is not None:
                observations[date] = value

    return sorted(observations.items())


# =========================================================
# 5. LOAD WEIGHTS
# =========================================================

def load_weights(filename):
    """
    Expected columns:
    window, lag, indicator, correlation,
    sum_abs_correlation, weight

    Use the existing weight column directly.
    Do not recalculate weights.
    """

    weights = {}

    with open(
        filename, "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "Weight CSV has no header."
            )

        # Normalize header names
        header_map = {
            name.strip().lower(): name
            for name in reader.fieldnames
        }

        required = {
            "window", "lag", "indicator", "weight"
        }

        if not required.issubset(header_map):
            raise ValueError(
                "Weight CSV must contain columns: "
                "window, lag, indicator, weight"
            )

        for row in reader:

            lag = parse_lag(
                row.get(header_map["lag"], "")
            )

            if lag != TARGET_LAG:
                continue

            window = row.get(
                header_map["window"], ""
            ).strip()

            indicator = row.get(
                header_map["indicator"], ""
            ).strip().upper()

            weight = parse_number(
                row.get(header_map["weight"], "")
            )

            if not window or not indicator:
                continue

            if weight is None:
                continue

            weights[(window, indicator)] = weight

    return weights


# =========================================================
# 6. ALIGN INDICATORS TO SPY TRADING DATES
# =========================================================

def align_indicators(spy, indicator_data):
    """
    For each SPY date, use the most recent indicator
    observation dated on or before that date.

    No future indicator observations are used.
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
# 7. CALCULATE DAILY SCORE AND FUTURE RETURN
# =========================================================

def calculate_daily_scores(
    spy,
    aligned,
    weights,
    window_name,
    start_year,
    end_year
):
    """
    Calculate scores only for the specified year window.

    Score = sum(normalized_indicator * window_weight)

    Future return:
    Open price after 14 subsequent SPY trading sessions
    compared with the current Open price.
    """

    results = []

    for i, spy_row in enumerate(spy):

        date = spy_row["date"]

        # Only include dates within this window
        if not (start_year <= date.year <= end_year):
            continue

        open_price = spy_row["open"]

        indicator_values = {}
        score = 0.0
        complete = True

        for indicator in INDICATORS:

            value = aligned[indicator].get(date)

            indicator_values[indicator] = value

            # Use weights belonging to this window
            weight = weights.get(
                (window_name, indicator)
            )

            if value is None or weight is None:
                complete = False
            else:
                score += weight * value

        # Find the SPY price 14 trading sessions later
        future_index = i + HORIZON

        future_open = None
        future_date = None
        future_return = None

        if future_index < len(spy):

            future_row = spy[future_index]

            future_open = future_row["open"]
            future_date = future_row["date"]

            future_return = (
                future_open / open_price - 1
            ) * 100

        results.append({
            "date": date,
            "open": open_price,
            "window": window_name,
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

    with open(
        filename, "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers
        )

        writer.writeheader()

        for row in results:

            output = {
                "Date": row["date"].isoformat(),
                "Open": row["open"],
                "Window": row["window"],
                "Score": (
                    row["score"]
                    if row["score"] is not None
                    else ""
                ),
                "Future_Date": (
                    row["future_date"].isoformat()
                    if row["future_date"] is not None
                    else ""
                ),
                "Future_Open": (
                    row["future_open"]
                    if row["future_open"] is not None
                    else ""
                ),
                "Return_14_Trading_Days_Percent": (
                    row["future_return"]
                    if row["future_return"] is not None
                    else ""
                )
            }

            for indicator in INDICATORS:

                value = row["indicator_values"][indicator]

                output[indicator] = (
                    value if value is not None else ""
                )

            writer.writerow(output)


# =========================================================
# 9. GROUP SCORES INTO BINS
# =========================================================

def calculate_probability(results):

    groups = {}

    for row in results:

        score = row["score"]
        ret = row["future_return"]

        # Skip incomplete scores and missing future returns
        if score is None or ret is None:
            continue

        # Example: [0.2, 0.3)
        bin_lower = math.floor(
            score / BIN_SIZE
        ) * BIN_SIZE

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

        # Sample variance of individual returns
        if n >= 2:

            return_variance = sum(
                (value - mean_return) ** 2
                for value in returns
            ) / (n - 1)

            # Variance of the sample mean
            variance_of_mean = return_variance / n

            standard_error = math.sqrt(
                variance_of_mean
            )

        else:
            return_variance = None
            variance_of_mean = None
            standard_error = None

        probability_up = (
            group["up_count"] / n
        ) * 100

        output.append({
            "Score_Bin_Lower": bin_lower,
            "Score_Bin_Upper": round(
                bin_lower + BIN_SIZE, 10
            ),
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
# 10. SAVE PROBABILITY RESULTS
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

    with open(
        filename, "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers
        )

        writer.writeheader()
        writer.writerows(results)


# =========================================================
# 11. PROCESS ONE WINDOW
# =========================================================

def process_window(
    spy,
    weights,
    start_year,
    end_year
):

    window_name = f"{start_year}-{end_year}"

    # Folder for this window's normalized indicators
    indicator_dir = os.path.join(
        NORMALIZED_DIR,
        window_name
    )

    if not os.path.isdir(indicator_dir):
        print(
            f"Skipping missing folder: {indicator_dir}"
        )
        return False

    print("\n" + "=" * 55)
    print("Processing window:", window_name)
    print("=" * 55)

    # Check that all 11 weights exist for this window
    missing_weights = [
        indicator
        for indicator in INDICATORS
        if (window_name, indicator) not in weights
    ]

    if missing_weights:
        raise ValueError(
            f"Missing weights for {window_name}: "
            + ", ".join(missing_weights)
        )

    # Load the 11 normalized indicator files
    indicator_data = {}

    for indicator in INDICATORS:

        filename = os.path.join(
            indicator_dir,
            indicator + ".csv"
        )

        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"Missing indicator file: {filename}"
            )

        data = load_indicator(filename)

        if not data:
            raise ValueError(
                f"No valid indicator data: {filename}"
            )

        indicator_data[indicator] = data

        print(
            f"Loaded {indicator}: {len(data)} observations"
        )

    # Align normalized indicators to SPY trading dates
    aligned = align_indicators(
        spy,
        indicator_data
    )

    # Calculate daily scores for this window
    daily_results = calculate_daily_scores(
        spy,
        aligned,
        weights,
        window_name,
        start_year,
        end_year
    )

    # Calculate probability and variance statistics
    probability_results = calculate_probability(
        daily_results
    )

    # Save each window in its own output folder
    window_output_dir = os.path.join(
        OUTPUT_DIR,
        window_name
    )

    os.makedirs(
        window_output_dir,
        exist_ok=True
    )

    output_daily = os.path.join(
        window_output_dir,
        "spy_daily_score.csv"
    )

    output_stats = os.path.join(
        window_output_dir,
        "spy_probability_by_score.csv"
    )

    save_daily_results(
        daily_results,
        output_daily
    )

    save_probability_results(
        probability_results,
        output_stats
    )

    valid_scores = sum(
        1
        for row in daily_results
        if row["score"] is not None
    )

    valid_returns = sum(
        1
        for row in daily_results
        if row["future_return"] is not None
    )

    print("SPY rows in window:", len(daily_results))
    print("Complete scores:", valid_scores)
    print("Rows with future returns:", valid_returns)
    print("Daily output:", output_daily)
    print("Probability output:", output_stats)

    return True


# =========================================================
# 12. MAIN
# =========================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("Loading SPY data...")

    spy = load_spy(SPY_FILE)

    if not spy:
        raise ValueError(
            "No valid SPY rows found. "
            "Check spy_data.csv and its Open column."
        )

    print("SPY trading rows:", len(spy))

    print("\nLoading weights...")

    weights = load_weights(WEIGHT_FILE)

    if not weights:
        raise ValueError(
            f"No weights found for lag = {TARGET_LAG}"
        )

    print("Weight records loaded:", len(weights))

    total_windows = 0
    skipped_windows = 0

    # 2000-2006, 2001-2007, ..., 2019-2025
    last_start_year = END_YEAR - WINDOW_SIZE + 1

    for start_year in range(
        START_YEAR,
        last_start_year + 1
    ):

        end_year = start_year + WINDOW_SIZE - 1

        success = process_window(
            spy,
            weights,
            start_year,
            end_year
        )

        if success:
            total_windows += 1
        else:
            skipped_windows += 1

    print("\n" + "=" * 55)
    print("ALL WINDOWS FINISHED")
    print("=" * 55)
    print("Windows processed:", total_windows)
    print("Windows skipped:", skipped_windows)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
