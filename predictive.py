# %%
from multiprocessing.sharedctypes import Value
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    plot_roc_curve,
    r2_score,
    plot_confusion_matrix,
)
import hyperopt
from helpers import run_hyperopt
from constants import TRAIN_COLS_IS_SOLD, TRAIN_COLS_REVENUE

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
        "lot.closingdate_day",
        "lot.startingdate",
        "lot.collection_date",
        "bid.date",
    ],
)

# %% Add feature 'lot.rel_nr': lot position on website corrected for total #nr lots in auction
df.sort_values(["auction.id", "lot.number"], inplace=True)
auction_size = df.groupby("auction.id").size().reset_index(name="auction.num_lots")
df = pd.merge(df, auction_size)
df["lot.rel_nr"] = (df.groupby("auction.id").cumcount() + 1) / df["auction.num_lots"]
# %% Add feature 'lot.revenue': revenue per lot
df["lot.revenue"] = df["bid.amount"] * df["bid.is_latest"] * df["bid.is_valid"]
lot_revenue = df.groupby("lot.id")["lot.revenue"].max().reset_index()
lot_revenue = lot_revenue.replace(0, np.nan)
df = pd.merge(df, lot_revenue)
# %%
df.sort_values(by='lot.closingdate', inplace=True)


# %% Add feature 'lot.closing_day_of_week': day of week that lot closes
df['lot.closing_day_of_week'] = df['lot.closingdate_day'].dt.dayofweek.astype('str')


# %% Drop duplicates with 'lot.id', 1 row per lot
df.drop_duplicates("lot.id", inplace=True)
# %% Add feature 'lot.closingdate_day': lot closing date without time
# df["lot.closingdate_day"] = pd.to_datetime(df["lot.closingdate"].dt.date)
# %% Sort first by auction, then by lot, then by bid date
df = df.sort_values(by=["auction.id", "lot.id", "bid.date"]).reset_index(drop=True)
# %% Add feature 'lot.days_open': number of days a lot is open
df["lot.days_open"] = (df["lot.closingdate_day"] - df["lot.startingdate"]).dt.days
# %% Remove outlier with 384 days (mean=7.04, median=7)
df = df[df['lot.days_open'] < 50]

# %% Convert float accountmanager ID to string (categorical)
df['project.accountmanager'] = df['project.accountmanager'].astype('int').astype('str')
# %% Add feature 'lot.category_count_in_auction': number of lots of same category in auction
auction_cat_counts = df.groupby(["auction.id", "lot.category"]).size().reset_index()
auction_cat_counts.rename(columns={0: "lot.category_count_in_auction"}, inplace=True)
df = pd.merge(df, auction_cat_counts)
# %% Add feature 'lot.category_closing_count': number of lots of same category closing on same day
df_lots_scarcity = (df.groupby(["lot.category", "lot.closingdate_day"])["lot.id"].nunique()).reset_index(
    name="lot.category_closing_count"
)
df = pd.merge(df_lots_scarcity, df)


# %% Add feature 'lot.subcategory_count_in_auction': number of lots of same subcategory in auction
auction_cat_counts = df.groupby(["auction.id", "lot.subcategory"]).size().reset_index()
auction_cat_counts.rename(columns={0: "lot.subcategory_count_in_auction"}, inplace=True)
df = pd.merge(df, auction_cat_counts)
# %% Add feature 'lot.subcategory_closing_count': number of lots of same subcategory closing on same day
df_lots_scarcity = (df.groupby(["lot.subcategory", "lot.closingdate_day"])["lot.id"].nunique()).reset_index(
    name="lot.subcategory_closing_count"
)
df = pd.merge(df_lots_scarcity, df)

# %% Add feature 'lot.closing_count': number of lots closing on same day
df_lots_closingcount = (
    df.groupby(["lot.closingdate_day"])["lot.id"]
    .nunique()
    .reset_index()
    .rename(columns={"lot.id": "lot.closing_count"})
)
df = pd.merge(df_lots_closingcount, df)
# %%
# df.dropna(subset=['lot.closingdate'], inplace=True)
df['lot.closing_timeslot'] = (
    df['lot.closingdate'] -
    df['lot.closingdate'].min()).apply(
        lambda td: td.total_seconds() /
    3600)

# %% Add lot closing_day
df['lot.closing_dayslot'] = lot_closing_day = (
    df['lot.closingdate_day'] -
    df['lot.closingdate_day'].min()).apply(
        lambda td: td.total_seconds() /
        (
            3600 *
            24)).astype('int')

# %%

# %%
sampled_hourslots_idx = []
sampled_hourslots = []

for auction_id in df['auction.id'].unique():
    df_auction = df[df['auction.id'] == auction_id]
    for closing_dayslot in df_auction['lot.closing_dayslot'].unique():
        df_auction_closing_dayslot = df_auction[df_auction['lot.closing_dayslot'] == closing_dayslot]
        timeslot_probabilities = df_auction_closing_dayslot['lot.closing_timeslot'].value_counts(
            normalize=True)
        N = df_auction_closing_dayslot.shape[0]
        if len(timeslot_probabilities.index) > 0:
            sampled_timeslots = np.random.choice(
                timeslot_probabilities.index, N, p=list(timeslot_probabilities))
            sampled_hourslots_idx += list(df_auction_closing_dayslot.index)
            sampled_hourslots += list(sampled_timeslots)

# %%
df.loc[sampled_hourslots_idx, "lot.closing_timeslot_interpolated"] = np.array(sampled_hourslots)
# for i, idx in enumerate(sampled_hourslots_idx):
#     df.loc[idx] = sampled_hourslots[i]
# %%
df_lot_closing_hourslots = pd.DataFrame(
    sampled_hourslots,
    index=sampled_hourslots_idx,
    columns=['lot.closing_hourslot'])
df = pd.merge(df, df_lot_closing_hourslots, left_index=True, right_index=True)
df['lot.closing_hourslot'] = df['lot.closing_hourslot'].astype('int')
# %% Add feature 'lot.timeslot_category_closing_count': number of lots of same category closing on same day
df_lots_scarcity = (df.groupby(["lot.category", "lot.closing_timeslot"])["lot.id"].nunique()).reset_index(
    name="lot.timeslot_category_closing_count"
)
df = pd.merge(df_lots_scarcity, df)
# %% Add feature 'lot.timeslot_subcategory_closing_count': number of lots of same subcategory closing on same day
df_lots_scarcity = (df.groupby(["lot.subcategory", "lot.closing_timeslot"])["lot.id"].nunique()).reset_index(
    name="lot.timeslot_subcategory_closing_count"
)
df = pd.merge(df_lots_scarcity, df)


# %% Add feature 'lot.closing_count': number of lots closing on same day
df_lots_closingcount = (
    df.groupby(["lot.closing_timeslot"])["lot.id"]
    .nunique()
    .reset_index()
    .rename(columns={"lot.id": "lot.timeslot_closing_count"})
)
df = pd.merge(df_lots_closingcount, df)
# %%
data = df[list(set(TRAIN_COLS_IS_SOLD).union(TRAIN_COLS_REVENUE))]
# %%
nunique_per_var = data.select_dtypes(exclude="O").nunique()
num_vars = nunique_per_var[nunique_per_var > 3].index
df[num_vars].skew()[df[num_vars].skew().abs() > 2].index
# %%
skewed_vars = df[num_vars].skew()[df[num_vars].skew().abs() > 1.5].index
df[skewed_vars].describe().round(1).T


# TODO check multivariate by lot.subcategory

# %%


# %% Classification for 'is_sold': train/test/validation split
train_data = pd.get_dummies(df[TRAIN_COLS_IS_SOLD].dropna())
y = train_data["lot.is_sold"]
X = train_data.drop("lot.is_sold", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# %%
train_test_split_undersampling(df, 'lot.is_sold')['lot.is_sold'].value_counts()

# %%
# %%
space = {
    "criterion": hyperopt.hp.choice("criterion", ["gini", "entropy"]),
    "splitter": hyperopt.hp.choice("splitter", ["best", "random"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 25, 50),
    "min_samples_split": hyperopt.hp.uniformint("min_samples_split", 20, 100),
    "max_features": hyperopt.hp.uniform("max_features", 0.6, 0.99),
}
run_hyperopt(DecisionTreeClassifier, X_train, y_train, space, mode="clf", max_evals=100)

# %% Train/score optimal Decision Tree
clf = DecisionTreeClassifier(
    criterion="entropy", max_depth=43, max_features=0.85, min_samples_split=36, splitter="random"
)
clf.fit(X_train, y_train)
plot_confusion_matrix(clf, X_test, y_test)
plot_roc_curve(clf, X_test, y_test)
clf.score(X_test, y_test)


# %% Regression for 'lot.revenue': train/test/validation split
space = {
    "criterion": "friedman_mse",
    "splitter": hyperopt.hp.choice("splitter", ["best", "random"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 30),
    "min_samples_leaf": hyperopt.hp.uniformint("min_samples_leaf", 1, 1000),
    "max_features": hyperopt.hp.uniform("max_features", 0.3, 0.99),
}

train_data_reg = pd.get_dummies(df[TRAIN_COLS_REVENUE].dropna())
y = np.log1p(train_data_reg["lot.revenue"])
X = train_data_reg.drop("lot.revenue", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
run_hyperopt(DecisionTreeRegressor, X_train, y_train, space, mode="reg", max_evals=500)
# %%
reg = DecisionTreeRegressor(
    criterion="friedman_mse", splitter="best", max_depth=15, max_features=0.91, min_samples_leaf=3
)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)
r2_score(np.expm1(y_test), np.expm1(reg.predict(X_test)))
# %%
train_data[['lot.revenue']]
# %%


# %%
pd.Series(reg.predict(X_reg)).nunique()
# %%

# TODO pickle DecisionTreeClassifier & DecisionTreeRegressor
# TODO decision variables:
#  - lot closing timeslot (hourly)
#  - lot.starting_price
