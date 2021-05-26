# %%
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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
df = df.sort_values(
    by=["auction.id", "lot.id", "bid.date"]).reset_index(drop=True)


# %%
clf = DecisionTreeClassifier(max_depth=4)
train_data = df.dropna()
y = train_data['lot.is_sold']
X = train_data.drop('lot.is_sold', axis=1).select_dtypes("float")
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf.fit(X_train, y_train)
clf.score(X_train, y_train), clf.score(X_test, y_test)


# %%
lin = LinearRegression()
y = train_data['lot.valid_bid_count']
X = train_data.drop('lot.valid_bid_count', axis=1).select_dtypes("float")
X_train, X_test, y_train, y_test = train_test_split(X, y)
lin.fit(X_train, y_train)
lin.score(X_train, y_train), lin.score(X_test, y_test)

# %% TODO FIX INCONSISTENT NUMBER OF BIDS
df = df[df['bid.is_valid'] == 1]
lot_vbid_count = df.groupby('lot.id')['bid.id'].nunique().reset_index()
lot_vbid_count.rename(
    columns={"bid.id": "bid.valid_count_chingchang"}, inplace=True)

df = pd.merge(df, lot_vbid_count)
# %%
(df['bid.valid_count_chingchang'] !=
 df['lot.valid_bid_count']).value_counts(normalize=True)
CHINGCHANG_COLS = ['bid.valid_count_chingchang', 'lot.valid_bid_count']
# %%
(df['bid.valid_count_chingchang'] - df['lot.valid_bid_count']).describe().round(3)
# %%
