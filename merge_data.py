# %%
import pandas as pd
import constants

# %% Dataset 1: Auction Platform
df_auctions = pd.read_csv(
    "data/raw_data/Dim_auction.csv",
    sep=";",
    infer_datetime_format=True,
)

# DROP 'is_active', 'open_for_bidding': Not Applicable (Data Description)
df_auctions.drop(["is_active", "open_for_bidding"], axis=1, inplace=True)
# DROP 'is_public' ,'is_private', 'onlinedate': consists of only NaN
# values/obtained from publicAuctionData.txt
df_auctions.drop(["onlinedate", "is_public", "is_private"], axis=1, inplace=True)
# DROP date columns, more precise date+time is obtained from 'AuctionCloseTimes.csv'
df_auctions.drop(["startdate", "closedate"], axis=1, inplace=True)

# RENAME columns
df_auctions.rename(columns=constants.COLNAMES_DIM_AUCTION, inplace=True)
# %% Dataset 1-A: 'publicAuctionData.txt' to obtain correct 'is_public' values for auctions
df_public_auction_data = pd.read_csv(
    "data/raw_data/publicAuctionData.txt",
    sep=";",
    index_col=["auction_id"],
    usecols=["auction_id", "is_public"],
    na_values=["na"],
)
df_public_auction_data.fillna(0, inplace=True)
# # Obtain 'auction.is_public' from 'publicAuctionData.txt' and assign to Dataset 1
df_auctions["auction.is_public"] = df_auctions["auction.id"].apply(
    lambda auction_id: df_public_auction_data["is_public"][auction_id]
)


# %% Dataset 1-B: 'AuctionCloseTimes.csv' to obtain start/end dates for auctions
df_auction_close_times = pd.read_csv(
    "data/raw_data/AuctionCloseTimes.csv",
    parse_dates=[
        "AUCTIONSTARTDATE",
        "AUCTIONENDDATE",
        "LOTMINSTARTDATE",
        "LOTMAXSTARTDATE",
        "LOTMINENDDATE",
        "LOTMAXENDDATE",
    ],
    dayfirst=True,
)
df_auction_close_times.rename(columns=constants.COLNAMES_AUCTION_CLOSE_TIMES, inplace=True)

# %%
AUCTIONS_MISSING_CLOSETIMES = set(df_auctions["auction.id"]).difference(
    set(df_auction_close_times["auction.id"])
)
print(f"{len(AUCTIONS_MISSING_CLOSETIMES)} auction IDs not found in 'publicAuctionData.txt'")
# %% Merge to add datetime information to all auctions
df_auctions = pd.merge(df_auctions, df_auction_close_times)
# %%
df_lots = pd.read_csv(
    "data/raw_data/DIM_LOT.csv.gz",
    sep=";",
    parse_dates=["startingdate", "closingdate", "uitleverdatum"],
    low_memory=False,
    decimal=",",
    dtype={"valid_bid_count": "Int64"},
)
df_lots = df_lots[df_lots["auction_id"] != "3667-"].astype({"auction_id": "int64"})
df_lots.rename(columns=constants.COLNAMES_LOTS, inplace=True)

# Drop date-only 'lot.closingdate' as 'Fact_bids1.csv.gz' has 'lot.closingdate' with date+time
df_lots.drop("lot.closingdate", axis=1, inplace=True)
# %%
df = pd.merge(df_auctions, df_lots)
# %%
df_projects = pd.read_csv("data/raw_data/Dim_projects.csv", sep=",")

# Drop project_auction_start, end as they will be taken from other df. Online irrelevant
df_projects.drop(
    labels=["project_auction_start", "project_auction_end", "project_auction_online"],
    axis=1,
    inplace=True,
)
df_projects.rename(columns=constants.COLNAMES_PROJECTS, inplace=True)
# %%
df = pd.merge(df, df_projects)
# %%
df_bids = pd.read_csv(
    "data/raw_data/Fact_bids1.csv.gz",
    sep=";",
    parse_dates=["bid_date", "lot_closingdate", "auction_closingdate"],
    low_memory=False,
    decimal=",",
)
# Drop 'seller_id' and 'channel_id' as they are Not Applicable
df_bids.drop(["seller_id", "channel_id"], axis=1, inplace=True)

# Drop 'auction_closingdate', 'opportunity_id' as these are already included
df_bids.drop(["auction_closingdate", "opportunity_id"], axis=1, inplace=True)
df_bids.rename(columns=constants.COLNAMES_BIDS, inplace=True)

# %%
df = pd.merge(df, df_bids, how="outer")

# %%
df_fact_lots = pd.read_csv("data/raw_data/fact_lots.csv.gz")
# Drop seller id, since irrelevant and auction closing date as other df will be used for this
df_fact_lots.drop(
    ["seller_id", "auction_closingdate", "closingdate", "startingdate", "efficy_business_line"],
    axis=1,
    inplace=True,
)
df_fact_lots.rename(columns=constants.COLNAMES_FACT_LOTS, inplace=True)
# %%
df = pd.merge(df, df_fact_lots)

# %% Drop column 'lot.type' and 'auction.main_category' (mixed dtypes, many missing)
df.drop(["lot.type", "auction.main_category"], axis=1, inplace=True)

# %% Drop column auction.is_closed: This has value 1 for each row
df.drop(["auction.is_closed"], axis=1, inplace=True)
# Drop column lot.is_open: This has value 0 for each row
df.drop(["lot.is_open"], axis=1, inplace=True)
# Drop column lot.has_bid: This has value 1 for each row
df.drop(["lot.has_bid"], axis=1, inplace=True)
#%%
df.to_csv("data/data_merged.csv.gz", index=False)
