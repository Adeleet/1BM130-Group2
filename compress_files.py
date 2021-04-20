import pandas as pd

pd.read_csv(
    "./Data/Fact_bids1.csv",
    sep=';').to_csv(
        "./Data/Fact_bids1.csv.gz",
    index=False)
pd.read_csv("./Data/DIM_LOT.csv", sep=';').to_csv("./Data/DIM_LOT.csv.gz", index=False)
