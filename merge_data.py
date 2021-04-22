# %%
import pandas as pd

# %%
df_auctions = pd.read_csv(
    "./Data/Dim_auction.csv",
    sep=";",
    parse_dates=["startdate", "closedate"],
    low_memory=False,
)
# Drop is_active and open_for_bidding, because they are not applicable
# And drop is_public and onlinedate as they consists of only NaN values
df_auctions.drop(labels=['is_active',
                         'open_for_bidding',
                         'onlinedate',
                         'is_public'],
                 axis=1,
                 inplace=True)
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
df_projects.drop(labels=["project_auction_start",
                         "project_auction_end",
                         "project_auction_online"], axis=1, inplace=True)

# %%
df_fact_bids1 = pd.read_csv(
    "./Data/Fact_bids1.csv.gz",
    sep=";",
    parse_dates=["bid_date", "lot_closingdate", "auction_closingdate"],
    low_memory=False,
)

df_fact_bids1.drop(labels=['seller_id', 'channel_id'], axis=1, inplace=True)
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
    parse_dates=["startingdate", "closingdate"],
)
df_fact_lots2 = pd.read_excel(
    "./Data/fact_lots_2.xlsx",
    skiprows=2,
    parse_dates=["startingdate", "closingdate"],
)

# %% Merge two dataframes into single dataframe for lots
df_fact_lots = pd.concat([df_fact_lots1, df_fact_lots2])
df_fact_lots.drop(labels=['seller_id', 'auction_closingdate'], axis=1, inplace=True)
df_fact_lots.rename(
    columns={
        "startingdate": "lot_startingdate",
        "closingdate": "lot_closingdate"},
    inplace=True)
# %% Merge df_public_auction_data and df_auctions
df_auctions_public_auction = pd.merge(df_auctions, df_public_auction_data)


# %% 1 MERGE WITH BIDS (lots-bids: 1-n relationship)
df_fact_lots_bids = pd.merge(df_fact_lots, df_fact_bids1)
# %% 2 MERGE WITH AUCTIONS
df_fact_lots_bids_auctions = pd.merge(df_fact_lots_bids, df_auctions_public_auction)
# %% 3 MERGE WITH LOTS

# df_lots contains an auction id which is a string.
# This does not occur in the other data so will be excluded when merging anyways
# So, before merging, remove entries with this auction_id and convert type to int
df_lots = df_lots[df_lots.auction_id != '3667-'].astype({'auction_id': 'int64'})
df = pd.merge(df_fact_lots_bids_auctions, df_lots)
# %% #TODO FIX INCORRECT MERGE (RESULTS IN ONLY 1300 ROWS)
pd.merge(df_fact_lots_bids_auctions, df_lots, left_on=["auction_id"], right_on=["auction_id"])
# %% 4 MERGE WITH AUCTION CLOSE TIMES
df_auction_close_times = pd.read_csv(
    "./Data/AuctionCloseTimes.csv",
    parse_dates=[
        'AUCTIONSTARTDATE',
        'AUCTIONENDDATE',
        'LOTMINSTARTDATE',
        'LOTMAXSTARTDATE',
        'LOTMINENDDATE',
        'LOTMAXENDDATE'])
df_auction_close_times['auction_id'] = df_auction_close_times["AUCTIONID"]
df_auction_close_times.drop("AUCTIONID", axis=1, inplace=True)
datetime_cols = [col for col in df_auction_close_times.columns if "DATE" in col]
df = pd.merge(df, df_auction_close_times)
df.drop(["auction_closingdate", "closingdate", "startingdate",
         "startdate", "closedate"], axis=1, inplace=True)


# %% Export to csv
df.to_csv("./Data/data_1bm130.csv.gz", chunksize=1000)
# %%
# %%
