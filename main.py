from src.pipeline import write_result
write_result("data/SPX500_1d.csv","result/predictions.csv")
print("Finished. See result/predictions.csv and result/metrics.txt")