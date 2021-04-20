import pandas as pd

# %%
df_bids = pd.read_csv("./Data/Fact_bids1.csv.gz")
df_lots = pd.read_csv("./Data/DIM_LOT.csv.gz")

# %%
df_lots.groupby("auction_id")["lot_id"].nunique()
df_top_auction = df_lots[df_lots["auction_id"] == 37596]
df_bids_top_auction = df_bids[df_bids["auction_id"] == 37596]
df_bids_top_auction[df_bids_top_auction["latest_bid"] == 1]
# %%
df_bids.set_index("lot_id")
# %%
df_bids["bid_amount"].max()
# %%
df_auctions = pd.read_csv("./Data/Dim_auction.csv", sep=";")
df_projects = pd.read_csv("./Data/Dim_projects.csv", sep=",")

# %%
df_auctions
# %%
df_projects
# %%
