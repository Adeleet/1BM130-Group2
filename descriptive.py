# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# %%
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

# %%
df = df.sort_values(by=["auction.id", "lot.id", "bid.date"]).reset_index(drop=True)

# %% Calculate lot price difference (end price - starting price)
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

# %%
df_auc_perf = pd.merge(df_auc_perf, df_auctions_dates, how="outer")
df_auc_perf = pd.merge(df_auc_perf, df_auc_num_bids, how="outer")
df_auc_perf = pd.merge(df_auc_perf, df_auction_price_diff, how="outer")

# %% Load scarcity measures
df_new_scarcity = pd.read_csv("data/scarcity_measures.csv.gz")

# center scarcity per category 
lot_category_mean_closing_count = (
    df_new_scarcity.groupby(["lot.category"])["lot.category_count_in_auction"].describe()[["mean", "std"]].reset_index()
)
lot_category_mean_closing_count.rename(
    columns={"mean": "lot.category_scarcity_mean", "std": "lot.category_scarcity_std"},
    inplace=True,
)
df_new_scarcity = pd.merge(df_new_scarcity, lot_category_mean_closing_count)
df_new_scarcity["lot.scarcity_cat"] = (df_new_scarcity["lot.category_count_in_auction"] - df_new_scarcity["lot.category_scarcity_mean"]) / df_new_scarcity[
    "lot.category_scarcity_std"
]

# center scarcity per subcategory
lot_subcategory_mean_closing_count = (
    df_new_scarcity.groupby(["lot.subcategory"])["lot.subcategory_count_in_auction"].describe()[["mean", "std"]].reset_index()
)
lot_subcategory_mean_closing_count.rename(
    columns={"mean": "lot.subcategory_scarcity_mean", "std": "lot.subcategory_scarcity_std"},
    inplace=True,
)
df_new_scarcity = pd.merge(df_new_scarcity, lot_subcategory_mean_closing_count)
df_new_scarcity["lot.subscarcity_cat"] = (df_new_scarcity["lot.subcategory_count_in_auction"] - df_new_scarcity["lot.subcategory_scarcity_mean"]) / df_new_scarcity[
    "lot.subcategory_scarcity_std"
]


# add price difference
price_diff_only = df.drop_duplicates("lot.id")[["lot.price_diff", 'lot.id']]
df_new_scarcity = df_new_scarcity.merge(price_diff_only)

#plot centered scarcity per category 
#%%is_sold
sns.boxplot(data=df_new_scarcity, x="lot.is_sold", y="lot.scarcity_cat")
plt.ylabel("Lot scarcity")
plt.xlabel("Lot sold")
plt.savefig("./figures/descriptive_scarcity_vs_is_sold.pdf")

# %% bid count
sns.scatterplot(data=df_new_scarcity, y="lot.valid_bid_count", x="lot.scarcity_cat")
plt.xlabel("Lot scarcity")
plt.ylabel("Lot bid count")
plt.savefig("./figures/descriptive_scarcity_vs_bid_count.pdf")

# %% price difference
sns.scatterplot(data=df_new_scarcity, y="lot.price_diff", x="lot.scarcity_cat")
plt.xlabel("Lot scarcity")
plt.ylabel("Lot price difference")
plt.savefig("./figures/descriptive_scarcity_vs_price_difference.pdf")

#plot center scarcity per SUBcategory
# %%is_sold
sns.boxplot(data=df_new_scarcity, x="lot.is_sold", y="lot.subscarcity_cat")
plt.ylabel("Lot scarcity")
plt.xlabel("Lot sold")
plt.savefig("./figures/descriptive_subscarcity_vs_is_sold.pdf")

# %%bid count
sns.scatterplot(data=df_new_scarcity, y="lot.valid_bid_count", x="lot.subscarcity_cat")
plt.xlabel("Lot scarcity")
plt.ylabel("Lot bid count")
plt.savefig("./figures/descriptive_subscarcity_vs_bid_count.pdf")

# %%price difference
sns.scatterplot(data=df_new_scarcity, y="lot.price_diff", x="lot.subscarcity_cat")
plt.xlabel("Lot scarcity")
plt.ylabel("Lot price difference")
plt.savefig("./figures/descriptive_subscarcity_vs_price_difference.pdf")

# %% Performance: Percentage Sold (Histogram)
sns.histplot(df_auc_perf["auction.pct_sold"])
# plt.title("Histogram of percentage sold for Auctions")
plt.xlabel("% Sold")
plt.savefig("./figures/descriptive_auction_pct_sold_hist.svg")

# %% Performance: Num Bids (Histogram)
sns.histplot(df_auc_perf["auction.num_bids"])
# plt.title("Histogram of number of bids for Auctions")
plt.xlabel("Number of bids")
plt.savefig("./figures/descriptive_auction_num_bids_hist.svg")

# %% Performance: Price Differencce (Histogram)
sns.histplot(df_auc_perf["auction.price_diff"])
plt.xlim(0, 30000)
# plt.title("Histogram of End Price - Start Price for Auctions")
plt.xlabel("End Price - Start Price")
plt.savefig("./figures/descriptive_auction_price_diff_hist.svg")

# %% Performance - timeseries
df_auc_perf_ts = df_auc_perf.set_index("auction.end_date").sort_index()
df_auc_perf_ts = df_auc_perf_ts.resample(pd.Timedelta("1d")).mean()
df_auc_perf_ts = df_auc_perf_ts.rolling(window="14d").mean()
# %% Performance: Percentage Sold (Timeseries)
df_auc_perf_ts["auction.pct_sold"].plot()
# plt.title("Timeseries of percentage sold for Auctions")
plt.xlabel("")
plt.ylabel("% Sold")
plt.savefig("./figures/descriptive_auction_perc_sold_timeseries.svg")


# %% Performance: Number of Bids (Timeseries)
df_auc_perf_ts["auction.num_bids"].plot()
# plt.title("Timeseries of number of bids for Auctions")
plt.xlabel("")
plt.ylabel("Number of Bids")
plt.savefig("./figures/descriptive_auction_num_bids_timeseries.svg")

# %% Performance: Price Difference (Timeseries))
df_auc_perf_ts["auction.price_diff"].plot()
# plt.title("Timeseries of End Price vs. Start Price for Auctions")
plt.xlabel("")
plt.ylabel("End Price - Start Price")
plt.savefig("./figures/descriptive_auction_price_diff_timeseries.svg")


# %% Impact of scarcity
df_lots_scarcity_bids = (
    df.groupby(["lot.category", "lot.closingdate"])["lot.valid_bid_count"]
    .sum()
    .reset_index(name="lot.num_bids")
)


# %% Calculate scarcity
df_lots_scarcity = (
    df.groupby(["lot.category", "lot.closingdate"])["lot.id"].nunique()
).reset_index(name="lot.closing_count")
# %%
df_scarcity = pd.merge(df_lots_scarcity, df_lots_scarcity_bids)
df = pd.merge(df_scarcity, df)
# %%
df_scarcity
# %%
df_total_scarcity = (
    df.groupby(["lot.category", "lot.closingdate"])["lot.id"]
    .nunique()
    .unstack()
    .sum()
    .reset_index(name="scarcity")
)
df = pd.merge(df_total_scarcity, df)


# %%
df_total_scarcity.set_index("lot.closingdate").plot()
# %%
df.drop_duplicates(subset=["lot.id"]).sort_values(by="lot.number")
# df["lot.name"]
# %%
df["lot.num_bids_log"] = np.log(df["lot.num_bids"])
# %%
df.drop_duplicates("lot.id").groupby("scarcity")["lot.num_bids"].median().plot()


# %%
most_common_categories = df["lot.category"].value_counts().sample(10).plot.bar()
# %%
scarcity_is_sold = (
    df.groupby(["lot.category", "lot.is_sold"])["lot.closing_count"].mean().unstack().round(2)
)
scarcity_is_sold.sample(10).plot.bar()

# %%
most_common_categories = df["lot.category"].value_counts().head(10).index
most_common_categories
# %%
df["lot.category"].value_counts().head(20)
# %%

df.drop_duplicates("lot.id").groupby("lot.category")["lot.price_diff"].mean()[
    most_common_categories
].plot.bar()
# %% center scarcity
lot_category_mean_closing_count = (
    df.groupby(["lot.category"])["lot.closing_count"].describe()[["mean", "std"]].reset_index()
)
lot_category_mean_closing_count.rename(
    columns={"mean": "lot.category_scarcity_mean", "std": "lot.category_scarcity_std"},
    inplace=True,
)
df = pd.merge(df, lot_category_mean_closing_count)
df["lot.scarcity"] = (df["lot.closing_count"] - df["lot.category_scarcity_mean"]) / df[
    "lot.category_scarcity_std"
]

# %% center valid_bid_count
lot_category_mean_valid_bid_count = (
    df.groupby(["lot.category"])["lot.valid_bid_count"].describe()[["mean", "std"]].reset_index()
)
lot_category_mean_valid_bid_count.rename(
    columns={
        "mean": "lot.category_valid_bid_count_mean",
        "std": "lot.category_valid_bid_count_std",
    },
    inplace=True,
)
df = pd.merge(df, lot_category_mean_valid_bid_count)
df["lot.valid_bid_count_relative"] = (
    df["lot.valid_bid_count"] - df["lot.category_valid_bid_count_mean"]
) / df["lot.category_valid_bid_count_std"]
# %%
sns.boxplot(data=df.drop_duplicates("lot.id"), x="lot.is_sold", y="lot.scarcity")
plt.ylim(-4, 5)
plt.ylabel("Lot scarcity (category-relative)")
plt.xlabel("Lot sold")
plt.savefig("./figures/descriptive_scarcity_is_sold.svg")


# %%
train_data = df.drop_duplicates("lot.id")[["lot.valid_bid_count_relative", "lot.scarcity"]]
sns.scatterplot(
    data=df.drop_duplicates("lot.id").sample(10000),
    y="lot.valid_bid_count_relative",
    x="lot.scarcity",
)
plt.ylim(-2, 3)
plt.xlim(-2, 3)
plt.xlabel("Lot Scarcity")
plt.ylabel("Number of Bids (category-relative)")
plt.savefig("./figures/descriptive_scarcity_bid_count.svg")

# train_data.quantile(0.05)
# %%
lot_category_mean_price_diff = (
    df.groupby(["lot.category"])["lot.price_diff"].describe()[["mean", "std"]].reset_index()
)
lot_category_mean_price_diff.rename(
    columns={"mean": "lot.category_price_diff_mean", "std": "lot.category_price_diff_std"},
    inplace=True,
)
df = pd.merge(df, lot_category_mean_price_diff)
df["lot.price_diff_relative"] = (df["lot.price_diff"] - df["lot.category_price_diff_mean"]) / df[
    "lot.category_price_diff_std"
]
# %%
df

# %%
train_data = df.drop_duplicates("lot.id")[["lot.price_diff_relative", "lot.scarcity"]]
sns.scatterplot(
    data=df.drop_duplicates("lot.id").sample(10000), y="lot.price_diff_relative", x="lot.scarcity"
)
plt.ylim(-2, 4)
plt.xlim(-2, 4)
plt.xlabel("Lot Scarcity")
plt.ylabel("Price Difference (category-relative)")
plt.savefig("./figures/descriptive_scarcity_price_diff.svg")
# %%
train_data = df.dropna(subset=["lot.scarcity", "lot.price_diff_relative"])
pearsonr(train_data["lot.price_diff_relative"], train_data["lot.scarcity"])

# %% plot starting price
df["lot.start_amount"].plot.hist(bins=30000, xlim=(0, 100))

# %% Impact of starting prices
df = pd.merge(df, df_auc_perf)
df["lot.start_amount_log"] = np.log1p(df["lot.start_amount"])
df["lot.price_diff_log"] = np.log1p(df["lot.price_diff"])
# %%
df["auction.num_bids"]
# %%

df_corr = df.corr()[["auction.pct_sold", "auction.num_bids", "auction.price_diff"]]

df_corr_sig = df_corr[df_corr.abs().max(axis=1).between(0.45, 0.99)]  # significant correlations
plt.figure(figsize=(8, 8))
sns.heatmap(df_corr_sig.round(2), annot=True)
plt.savefig("./figures/descriptive_corr_heatmap.svg")
# %%
# sns.boxplot(data=df.drop_duplicates('lot.id'),x='lot.starting_at_1EUR', y='lot.valid_bid_count_log')
df_lots = df.drop_duplicates("lot.id")

df_corr2 = df.corr()[["lot.starting_at_1EUR", "lot.start_amount"]]
df_corr2[df_corr2.abs().max(axis=1).between(0.5, 0.99)].round(2)


# %%
# %%

df_lots["lot.valid_bid_count_log"] = np.log(df_lots["lot.valid_bid_count"])
# %%
sns.boxplot(data=df_lots, x="lot.starting_at_1EUR", y="lot.valid_bid_count_log")
plt.xlabel("Lot starting at 1 EUR")
plt.ylabel("log (Number of bids)")
plt.savefig("./figures/descriptive_lot1EUR_logNumBids_boxplot.svg")

# %%
sns.boxplot(data=df_lots, x="lot.starting_at_1EUR", y="lot.price_diff_log", whis=3)
plt.xlabel("Lot starting at 1 EUR")
plt.ylabel("log (Price Difference)")
plt.savefig("./figures/descriptive_lot1EUR_logPriceDiff_boxplot.svg")

# %%
df.groupby("lot.starting_at_1EUR")["lot.is_sold"].describe().round(2)
# %%
sns.boxplot(data=df_lots, x="auction.is_homedelivery", y="lot.valid_bid_count_log")
plt.xlabel("Lot homedelivery")
plt.ylabel("log (Number of bids)")
plt.savefig("./figures/descriptive_lotHomedelivery_logNumBids_boxplot.svg")

# %%
sns.boxplot(data=df_lots, x="auction.is_homedelivery", y="lot.price_diff_log", whis=3)
plt.xlabel("Lot homedelivery")
plt.ylabel("log (Price Difference)")
plt.savefig("./figures/descriptive_lotHomedelivery_logPriceDiff_boxplot.svg")

# %%
df.groupby("auction.is_homedelivery")["lot.is_sold"].describe().round(2)


# %%
sns.boxplot(data=df_lots, x="auction.is_public", y="lot.valid_bid_count_log")
plt.xlabel("Public Auction")
plt.ylabel("log (Number of bids)")
plt.savefig("./figures/descriptive_lotIsPublic_logNumBids_boxplot.svg")
# %%

sns.boxplot(data=df_lots, x="auction.is_public", y="lot.price_diff_log", whis=3)
plt.xlabel("Public Auction")
plt.ylabel("log (Price Difference)")
plt.savefig("./figures/descriptive_lotIsPublic_logPriceDiff_boxplot.svg")
# %%
df.groupby("auction.is_public")["lot.is_sold"].describe().round(2)

# %%

# %%

# %%
