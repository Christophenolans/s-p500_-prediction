
import csv
import os
from collections import defaultdict

input_files = [
    r"D:\Deengam\linear project\code\average_rolling_corr_result_CPI.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_DTWEXBGS.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_FEDFUNDS.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_GOLD.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_NFP.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_OIL.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_PCEPI.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_PPI.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_REALGDP.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_UNEMPLOYRATE.csv",
    r"D:\Deengam\linear project\code\average_rolling_corr_result_VIX.csv"
]

output_file = (
    r"D:\Deengam\linear project\code\weighted_correlation_lag14.csv"
)

TARGET_LAG = 14

# Store correlations by window and indicator
data = defaultdict(list)

# 1. Read all files and select lag = 14
for input_file in input_files:
    indicator = os.path.basename(input_file)
    indicator = indicator.replace(
        "average_rolling_corr_result_", ""
    ).replace(".csv", "")

    with open(input_file, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if int(row["lag"]) != TARGET_LAG:
                continue

            window = row["window"]
            correlation = float(row["average_correlation"])

            data[(window, indicator)].append(correlation)

# 2. Average duplicate correlations for each window and indicator
correlations = {}

for key, values in data.items():
    correlations[key] = sum(values) / len(values)

# 3. Group correlations by window
window_data = defaultdict(dict)

for (window, indicator), correlation in correlations.items():
    window_data[window][indicator] = correlation

# 4. Calculate weights using sum of absolute correlations
results = []

for window, indicators in sorted(window_data.items()):
    denominator = sum(abs(cor) for cor in indicators.values())

    for indicator, correlation in sorted(indicators.items()):
        weight = (
            correlation / denominator
            if denominator != 0
            else ""
        )

        results.append({
            "window": window,
            "lag": TARGET_LAG,
            "indicator": indicator,
            "correlation": correlation,
            "sum_abs_correlation": denominator,
            "weight": weight
        })

# 5. Save output CSV
fieldnames = [
    "window",
    "lag",
    "indicator",
    "correlation",
    "sum_abs_correlation",
    "weight"
]

with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("Done! Saved to:", output_file)
