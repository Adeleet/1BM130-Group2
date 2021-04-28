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
#From public auction data we only need is_public (and auction_id for merging)
df_public_auction_data = pd.read_csv(
    "./Data/publicAuctionData.txt",
    sep=";",
    usecols= ['auction_id', 'is_public'],
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
#Drop project_auction_start, end as they will be taken from other df. Online irrelevant
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
#Drop seller id and channel id as they are irrelevant
df_fact_bids1.drop(labels=['seller_id', 'channel_id'], axis=1, inplace=True)

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
#Drop seller id, since irrelevant and auction closing date as other df will be used for this
df_fact_lots.drop(
    labels=['seller_id', 'auction_closingdate'], axis=1, inplace=True)  
df_fact_lots.rename(
    columns={
        "startingdate": "lot_startingdate",
        "closingdate": "lot_closingdate"},
    inplace=True)

# %% Add the correct is_public column to df_auctions
df_auctions_correct = pd.merge(df_auctions, df_public_auction_data, on = 'auction_id')


# %% 1 MERGE WITH BIDS (lots-bids: 1-n relationship)
df_fact_lots.drop(columns=['lot_closingdate'], inplace=True)
df_fact_lots_bids = pd.merge(df_fact_bids1, df_fact_lots)
print(df_fact_lots_bids.shape)
#business line zorgt voor 2000 minder rijen

not_overlapping_ids = set(df_fact_bids1["lot_id"].values).difference(set(df_fact_lots["lot_id"].values))
print(f'The following number of lot_ids ({len(not_overlapping_ids)}) is not available in both dataframes')

# %% 2 MERGE WITH AUCTIONS
df_fact_lots_bids_auctions = pd.merge(df_fact_lots_bids, df_auctions_correct)

# %% 3 MERGE WITH LOTS

# df_lots contains an auction id which is a string.
# This does not occur in the other data so will be excluded when merging anyways
# So, before merging, remove entries with this auction_id and convert type to int
df_lots = df_lots[df_lots.auction_id !=
                  '3667-'].astype({'auction_id': 'int64'})
#df_fact_lots_bids_auctions categorycode column consists of almost only NaNs
# Remove these before merging, and use the correct categorycodes from df_lots
df_fact_lots_bids_auctions.drop(labels = ['categorycode'], axis = 1, inplace=True)
df_lots.drop(columns=['opportunity_id'], inplace=True)
df = pd.merge(df_fact_lots_bids_auctions, df_lots)

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

df_auction_close_times.rename(
    columns={"AUCTIONID": "auction_id"}, inplace=True)
df = pd.merge(df, df_auction_close_times)
df.drop(["auction_closingdate", "closingdate", "startingdate",
         "startdate", "closedate", "type", "lot_newprice"], axis=1, inplace=True)
df['brand'] = df['brand'].fillna('no_brand')
df = df[df['efficy_business_line'] != '006.Automotive']
df = df.reset_index(drop=True)

print(df['efficy_business_line'].unique())
# column type bevat opmerkelijk waardes zoals "grijs" en "baby car seat cover"
# brand values zijn niet legit
# %% Export to csv
df.to_csv("./Data/data_1bm130.csv.gz", chunksize=1000)
# %%
# %%
