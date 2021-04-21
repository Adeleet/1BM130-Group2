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

# %% Merge two dataframes into single dataframe for lots
df_fact_lots = pd.concat([df_fact_lots1, df_fact_lots2])
# %% 1 MERGE WITH BIDS (lots-bids: 1-n relationship)
df_fact_lots_bids = pd.merge(df_fact_lots, df_fact_bids1)
# %% 2 MERGE WITH AUCTIONS
df_fact_lots_bids_auctions = pd.merge(df_fact_lots_bids, df_auctions)
# %% 3 MERGE WITH LOTS

# df_lots contains an auction id which is a string.
# This does not occur in the other data so will be excluded when merging anyways
# So, before merging, remove entries with this auction_id and convert type to int
df_lots = df_lots[df_lots.auction_id != '3667-'].astype({'auction_id': 'int64'})
df = pd.merge(df_fact_lots_bids_auctions, df_lots)

# %% 4 MERGE WITH PROJECTS
df.to_csv("./Data/data_1bm130.csv.gz", index=False)
# %%
