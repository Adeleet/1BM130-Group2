# %%
import pandas as pd

# %%
df_auctions = pd.read_csv(
    "./Data/Dim_auction.csv",
    sep=";",
    parse_dates=["startdate", "closedate"],
    low_memory=False,
)
# %%
df_lots = pd.read_csv(
    "./Data/DIM_LOT.csv.gz",
    sep=";",
    parse_dates=["startingdate", "closingdate", "uitleverdatum"],
    low_memory=False,
)
# %%
df_projects = pd.read_csv(
    "./Data/Dim_projects.csv",
    sep=",",
    parse_dates=[
        "project_auction_start",
        "project_auction_end",
        "project_auction_online",
    ],
    low_memory=False,
)
# %%
df_fact_bids1 = pd.read_csv(
    "./Data/Fact_bids1.csv.gz",
    sep=";",
    parse_dates=["bid_date", "lot_closingdate", "auction_closingdate"],
    low_memory=False,
)
# %%
df_public_auction_data = pd.read_csv(
    "./Data/publicAuctionData.csv",
    sep=";",
    parse_dates=["startdate", "closedate", "onlinedate"],
    low_memory=False,
)
# %%

df_fact_lots1 = pd.read_excel(
    "./Data/fact_lots_1.xlsx",
    skiprows=2,
    parse_dates=["auction_closingdate", "closingdate"],
)
df_fact_lots2 = pd.read_excel(
    "./Data/fact_lots_2.xlsx",
    skiprows=2,
    parse_dates=["auction_closingdate", "closingdate"],
)

# %%
