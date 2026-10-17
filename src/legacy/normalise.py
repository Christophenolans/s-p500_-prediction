
import csv
import os
from datetime import datetime

# =====================================================
# 1. SETTINGS
# =====================================================

BASE_DIR = r"D:\Deengam\linear project"

INPUT_DIR = os.path.join(BASE_DIR, "Indicator data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Normalized data")

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

START_YEAR = 2000
END_YEAR = 2025
WINDOW_SIZE = 7


# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def parse_date(value):
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
            continue

    return None


def parse_number(value):
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        number = float(value)

        if number != number:  # NaN
            return None

        if number == float("inf") or number == float("-inf"):
            return None

        return number

    except ValueError:
        return None


# =====================================================
# 3. LOAD INDICATOR DATA
# =====================================================

def load_indicator(filename):
    """
    Reads a CSV with:
    - First column: date
    - One or more columns after the date

    Uses the first non-empty numeric value after the date.
    """

    data = {}

    with open(
        filename,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

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
            data[date] = value

    return sorted(data.items())


# =====================================================
# 4. MIN-MAX NORMALIZATION
# =====================================================

def normalize_window(data, start_year, end_year):
    """
    Normalize values in one window to [-1, 1].

    Formula:
    normalized = 2 * (x - min) / (max - min) - 1

    If all values are identical, normalized = 0.
    """

    window_data = [
        (date, value)
        for date, value in data
        if start_year <= date.year <= end_year
    ]

    if not window_data:
        return []

    values = [value for date, value in window_data]

    minimum = min(values)
    maximum = max(values)

    results = []

    for date, value in window_data:

        if maximum == minimum:
            normalized = 0.0

        else:
            normalized = (
                2 * (value - minimum) / (maximum - minimum)
                - 1
            )

        results.append({
            "Date": date.isoformat(),
            "Original_Value": value,
            "Normalized_Value": normalized
        })

    return results


# =====================================================
# 5. SAVE NORMALIZED DATA
# =====================================================

def save_csv(results, filename):

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "Date",
            "Original_Value",
            "Normalized_Value"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


# =====================================================
# 6. PROCESS ALL WINDOWS AND INDICATORS
# =====================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_files = 0

    # Example: 2000-2006, 2001-2007, ..., 2019-2025
    last_start_year = END_YEAR - WINDOW_SIZE + 1

    for start_year in range(
        START_YEAR,
        last_start_year + 1
    ):

        end_year = start_year + WINDOW_SIZE - 1

        window_name = f"{start_year}-{end_year}"

        window_dir = os.path.join(
            OUTPUT_DIR,
            window_name
        )

        os.makedirs(window_dir, exist_ok=True)

        print(f"\nProcessing window: {window_name}")

        for indicator in INDICATORS:

            input_file = os.path.join(
                INPUT_DIR,
                indicator + ".csv"
            )

            if not os.path.exists(input_file):
                print(f"WARNING: Missing file: {input_file}")
                continue

            data = load_indicator(input_file)

            normalized_data = normalize_window(
                data,
                start_year,
                end_year
            )

            output_file = os.path.join(
                window_dir,
                indicator + ".csv"
            )

            save_csv(normalized_data, output_file)

            total_files += 1

            print(
                f"  {indicator}: "
                f"{len(normalized_data)} rows saved"
            )

    print("\n================================")
    print("Normalization completed!")
    print("Windows:", last_start_year - START_YEAR + 1)
    print("CSV files created:", total_files)
    print("Output directory:", OUTPUT_DIR)
    print("================================")


if __name__ == "__main__":
    main()
