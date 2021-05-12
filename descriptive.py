#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

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

#%%
df_auc_perf
#%% Performance: Percentage Sold (Histogram)
sns.histplot(df_auc_perf["auction.pct_sold"])
plt.title("Histogram of percentage sold for Auctions"), plt.xlabel("% Sold")

#%% Performance: Num Bids (Histogram)
sns.histplot(df_auc_perf["auction.num_bids"])
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

# %%
df.groupby("lot.category")["auction.pct_sold"].mean().sample(20).plot.bar()
#%%
df_corr = df.corr().abs()[["auction.pct_sold", "auction.num_bids", "auction.price_diff"]]
#%%
sns.heatmap(df_corr[df_corr.max(axis=1).between(0.5, 0.99)].round(2), annot=True)
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

# %% Calculate scarcity
df_lots_scarcity = (
    df.groupby(["lot.category", "lot.closingdate"])["lot.id"].nunique()
).reset_index(name="lot.closing_count")

sns.histplot(df_lots_scarcity["lot.closing_count"], bins=32)
plt.title("Histogram of scarcity"), plt.xlabel("Scarcity")

# %% I don't know why we would need this
# Number of bids per category per closing date?
df_lots_scarcity_bids = (
    df.groupby(["lot.category", "lot.closingdate"])["lot.valid_bid_count"]
    .sum()
    .reset_index(name="lot.num_bids") 
)

# %%
df_scarcity = pd.merge(df_lots_scarcity, df_lots_scarcity_bids)
df = pd.merge(df_scarcity, df)
per_category = df_scarcity.groupby("lot.category")["lot.closing_count"].mean().reset_index()

per_category.nlargest(15, columns=["lot.closing_count"]).plot.bar(x="lot.category")
plt.xticks(rotation=90)
plt.title("15 least scarce categories")
plt.ylabel("Avg. number of items per closing date that the category is present") 
plt.xlabel("Lot Category")

per_category.nsmallest(15, columns=["lot.closing_count"]).plot.bar(x="lot.category")
plt.xticks(rotation=90)
plt.title("15 most scarce categories")
plt.ylabel("Avg. number of items per closing date that the category is present") 
plt.xlabel("Lot Category")

# %% Log-transform
df["lot.closing_count_log"] = np.log(df['lot.closing_count'])
sns.histplot(df["lot.closing_count_log"], bins=32)
plt.title("Histogram of log-transformed scarcity"), plt.xlabel("Scarcity")

# deze plot duurt eeuwen
#sns.regplot(x=df['lot.closing_count_log'], y=df['lot.num_bids'])

# %% Regression per lot
print(  'REGRESSION PER LOT: \n'
        'num_bids = per category per closing date! \n')

for alt in ['valid_bid_count', 'num_bids', 'is_sold']:
    model = LinearRegression()
    X = df[['lot.closing_count_log']].values
    y = df[[f'lot.{alt}']].values          
    model.fit(X, y)
    print(f'Scarcity per lot on {alt} per lot')
    print(f'r2: {round(model.score(X, y), 2)}')
    print(f'coef: {round(model.coef_[0][0], 2)} \n')

# %% Create auction scarcity measures
df_auctions_scarcity = (
    df.groupby(["auction.id"])["lot.closing_count"].mean()
).reset_index(name="auction.closing_count")
df_auctions_scarcity["auction.closing_count_log"] = np.log(df_auctions_scarcity['auction.closing_count'])

df_auctions_scarcity_bids = (
    df.groupby(["auction.id"])["lot.valid_bid_count"].sum() 
).reset_index(name="auction.num_bids")

df_auctions_scarcity_perc_sold = (
    df.groupby(["auction.id"])["lot.is_sold"].mean()
).reset_index(name="auction.perc_sold")

df_scarcity2 = pd.merge(df_auctions_scarcity, df_auctions_scarcity_bids)
df_scarcity2 = pd.merge(df_scarcity2, df_auctions_scarcity_perc_sold)
df = pd.merge(df_scarcity2, df)

sns.histplot(df["auction.num_bids"])
plt.title("Histogram of the number of bids per auction"), plt.xlabel("Number of bids per auction")
print("Check hoe laag deze aantallen zijn, maar met num_bids wordt het niet beter")

# %%
sns.histplot(df["auction.perc_sold"])
plt.title("Histogram of the percentage of lots sold per auction"), plt.xlabel("Percentage sold")

# %% Regression per auction
print(  'REGRESSION PER AUCTION: \n')

for alt in ['num_bids', 'perc_sold']:
    model = LinearRegression()
    X = df[['auction.closing_count_log']].values
    y = df[[f'auction.{alt}']].values          
    model.fit(X, y)
    print(f'Scarcity per auction on {alt} per auction')
    print(f'r2: {round(model.score(X, y), 2)}')
    print(f'coef: {round(model.coef_[0][0], 2)} \n')

# %%
df_scarcity

# %% The effect of the starting price
sns.histplot(df["lot.start_amount"], bins=32)
plt.title("Histogram of the start amount per lot"), plt.xlabel("Amount in euros")


# %%
df_lowest_bids = (
    df.groupby(["lot.id"])["bid.amount"].min() 
).reset_index(name="lot.min_bid")

sns.histplot(df_lowest_bids["lot.min_bid"])
plt.title("Histogram of the lowest bid per lot"), plt.xlabel("Amount in euros")

# %%
