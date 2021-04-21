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


def get_common_cols(df1, df2):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    return list(cols1.intersection(cols2))


# %% Merge two dataframes into single dataframe for lots
df_fact_lots = pd.concat([df_fact_lots1, df_fact_lots2])
# %% 1 MERGE WITH BIDS (lots-bids: 1-n relationship)
common_cols_lots_bids = get_common_cols(df_fact_bids1, df_fact_lots)
df_fact_lots_bids = pd.merge(
    df_fact_lots,
    df_fact_bids1,
    left_on=common_cols_lots_bids,
    right_on=common_cols_lots_bids)
# %% 2 MERGE WITH AUCTIONS
common_cols_auctions_bids = get_common_cols(df_auctions, df_fact_lots_bids)
df_fact_lots_bids_auctions = pd.merge(
    df_fact_lots_bids,
    df_auctions,
    left_on=common_cols_auctions_bids,
    right_on=common_cols_auctions_bids)
# %% 3 MERGE WITH LOTS
common_cols_auctions_bids_lots = get_common_cols(df_fact_lots_bids_auctions, df_lots)
df = pd.merge(
    df_fact_lots_bids_auctions,
    df_lots,
    left_on=['lot_id'],
    right_on=['lot_id'])

# %% 4 MERGE WITH PROJECTS
df.to_csv("./Data/data_1bm130.csv.gz", index=False)
# %%
