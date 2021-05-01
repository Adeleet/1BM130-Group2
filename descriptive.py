#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
df = pd.read_csv(
    "data/data_merged.csv.gz",
    parse_dates=[
        "auction.start_date",
        "auction.end_date",
        "auction.lot_min_start_date",
        "auction.lot_max_start_date",
        "auction.lot_min_end_date",
        "auction.lot_max_end_date",
        "lot.closingdate",
        "lot.startingdate",
        "lot.collection_date",
        "bid.date",
    ],
)
#%%
df = df.sort_values(by=["auction.id", "lot.id", "bid.date"]).reset_index(drop=True)

#%% Calculate lot price difference (end price - starting price)
df_valid_bids = df[(df["bid.is_valid"] == 1) & (df["lot.is_sold"] == 1)]
df_lot_start_price = df_valid_bids.groupby("lot.id")["lot.start_amount"].min()
df_lot_end_price = df_valid_bids.groupby("lot.id")["bid.amount"].max()
df_lot_price_diff = df_lot_end_price - df_lot_start_price
df_lot_price_diff = df_lot_price_diff.reset_index(name="lot.price_diff")
df = pd.merge(df, df_lot_price_diff, how="outer")
# %% Auction Performance
df_auctions_dates = df.groupby("auction.id")["auction.end_date"].min().reset_index()

# Performance A - % sold
df_auc_perf = df.groupby("auction.id")["lot.is_sold"].mean().reset_index(name="auction.pct_sold")

# Performance B - Number of bids
df_auc_num_bids = (
    df.drop_duplicates("lot.id")
    .groupby("auction.id")["lot.valid_bid_count"]
    .sum()
    .reset_index(name="auction.num_bids")
)
# Performance C - Price difference (end vs. start price)
df_auction_price_diff = (
    df.drop_duplicates("lot.id")
    .groupby("auction.id")["lot.price_diff"]
    .sum()
    .reset_index(name="auction.price_diff")
)
#%%
df_auc_perf = pd.merge(df_auc_perf, df_auctions_dates, how="outer")
df_auc_perf = pd.merge(df_auc_perf, df_auc_num_bids, how="outer")
df_auc_perf = pd.merge(df_auc_perf, df_auction_price_diff, how="outer")

#%% Performance: Percentage Sold (Histogram)
sns.histplot(df_auc_perf["auction.pct_sold"])
plt.title("Histogram of percentage sold for Auctions"), plt.xlabel("% Sold")

#%% Performance: Num Bids (Histogram)
sns.hist(df_auc_perf["auction.num_bids"])
plt.title("Histogram of number of bids for Auctions"), plt.xlabel("Number of bids")
#%% Performance: Price Differencce (Histogram)
sns.histplot(
    df_auc_perf["auction.price_diff"],
)
plt.xlim(0, 30000)
plt.title("Histogram of End Price - Start Price for Auctions"), plt.xlabel(
    "End Price - Start Price"
)
#%% Performance - timeseries
df_auc_perf_ts = df_auc_perf.set_index("auction.end_date").sort_index()
df_auc_perf_ts = df_auc_perf_ts.resample(pd.Timedelta("1d")).mean()
df_auc_perf_ts = df_auc_perf_ts.rolling(window="14d").mean()
#%% Performance: Percentage Sold (Timeseries)
df_auc_perf_ts["auction.pct_sold"].plot()
plt.title("Timeseries of percentage sold for Auctions"), plt.xlabel("Date"), plt.ylabel("% Sold")

#%% Performance: Number of Bids (Timeseries)
df_auc_perf_ts["auction.num_bids"].plot()
plt.title("Timeseries of number of bids for Auctions"), plt.xlabel("Date"), plt.ylabel(
    "Number of Bids"
)
#%% Performance: Price Difference (Timeseries))
df_auc_perf_ts["auction.price_diff"].plot()
plt.title("Timeseries of End Price vs. Start Price for Auctions"), plt.xlabel("Date"), plt.ylabel(
    "End Price - Start Price "
)

# %%
df = pd.merge(df, df_auc_perf)

#%%
df_auctions = df.drop_duplicates("auction.id")
df_auc_perf_hd = df_auctions.groupby("auction.is_homedelivery")[
    ["auction.pct_sold", "auction.num_bids", "auction.price_diff"]
].mean()
#%%
df_auc_perf_hd["auction.pct_sold"].plot.bar(ylim=(0, 1))
#%%
df_auc_perf_hd["auction.num_bids"].plot.bar()
#%%
df_auc_perf_hd["auction.price_diff"].plot.bar()

# %% Scarcity
df_lots_scarcity = (
    df.groupby(["lot.category", "lot.closingdate"])["lot.id"].nunique()
).reset_index(name="lot.closing_count")
# %%
df_lots_scarcity_bids = (
    df.groupby(["lot.category", "lot.closingdate"])["lot.valid_bid_count"]
    .sum()
    .reset_index(name="lot.num_bids")
)
# %%
pd.merge(df_lots_scarcity, df_lots_scarcity_bids).corr()

# %%
