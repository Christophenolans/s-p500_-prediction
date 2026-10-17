
import csv

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

output_file = r"D:\Deengam\linear project\code\average_correlation_all.csv"

N = 0

# Store correlations by lag across all indicators
data = {}

for lag in range(N - 14, N + 15):
    data[lag] = []

for input_file in input_files:
    with open(input_file, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            lag = int(row["lag"])
            correlation = float(row["average_correlation"])

            if N - 14 <= lag <= N + 14:
                data[lag].append(correlation)

# Write one average for each lag
with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
    fieldnames = [
        "lag",
        "number_of_correlations",
        "average_correlation"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for lag in range(N - 14, N + 15):
        values = data[lag]

        if values:
            average = sum(values) / len(values)
        else:
            average = ""

        writer.writerow({
            "lag": lag,
            "number_of_correlations": len(values),
            "average_correlation": average
        })

print("Done! Saved to:", output_file)
