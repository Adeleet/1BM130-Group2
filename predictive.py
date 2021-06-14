# %% Import packages
import tqdm
from datetime import datetime
import pickle
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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

# %% Read dataset
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
# %% Add feature 'lot.closing_day_of_week': day of week that lot closes
df["lot.closing_day_of_week"] = df["lot.closingdate_day"].dt.dayofweek.astype("str")
# %% Drop duplicates with 'lot.id', 1 row per lot
df.drop_duplicates("lot.id", inplace=True)
# %% Sort first by auction, then by lot, then by bid date
df = df.sort_values(by=["auction.id", "lot.id", "bid.date"]).reset_index(drop=True)
# %% Add feature 'lot.days_open': number of days a lot is open
df["lot.days_open"] = (df["lot.closingdate_day"] - df["lot.startingdate"]).dt.days
# %% Remove outlier with 384 days (mean=7.04, median=7)
df = df[df["lot.days_open"] < 50]
# %% Convert float accountmanager ID to string (categorical)
df["project.accountmanager"] = df["project.accountmanager"].astype("int").astype("str")
# %% Add feature 'lot.category_count_in_auction' and 'lot.subcategory_count_in_auction' number of lots of same (sub) category in auction
auction_cat_counts = df.groupby(["auction.id", "lot.category"]).size().reset_index()
auction_cat_counts.rename(columns={0: "lot.category_count_in_auction"}, inplace=True)
df = pd.merge(df, auction_cat_counts)

auction_subcat_counts = df.groupby(["auction.id", "lot.subcategory"]).size().reset_index()
auction_subcat_counts.rename(columns={0: "lot.subcategory_count_in_auction"}, inplace=True)
df = pd.merge(df, auction_subcat_counts)


# %% Take closing dayslot (day) and timeslot (hour) for scarcity
TIMESTAMP_MIN = pd.Timestamp("2019-12-09 00:00:00")

# %%
df["lot.closing_dayslot"] = (df["lot.closingdate_day"] - TIMESTAMP_MIN).dt.days
df["lot.closing_timeslot"] = (df["lot.closingdate"] - TIMESTAMP_MIN).dt.total_seconds() / 3600

# Initialize array for indices to be interpolated and interpolated values
sampled_hourslots_idx = []
sampled_hourslots = []

# for each auction
for auction_id in df["auction.id"].unique():
    df_auction = df[df["auction.id"] == auction_id]
    # for each closing day within this auction
    for closing_dayslot in df_auction["lot.closing_dayslot"].unique():
        df_auction_closing_dayslot = df_auction[df_auction["lot.closing_dayslot"] == closing_dayslot]
        # compute closing timeslot (hour) probabilities
        timeslot_probabilities = df_auction_closing_dayslot["lot.closing_timeslot"].value_counts(
            normalize=True
        )

        timeslots_pop = timeslot_probabilities.index
        N = df_auction_closing_dayslot.shape[0]
        if len(timeslots_pop) > 0:
            sampled_timeslots = np.random.choice(timeslots_pop, N, p=list(timeslot_probabilities))
            sampled_hourslots_idx += list(df_auction_closing_dayslot.index)
            sampled_hourslots += list(sampled_timeslots)
df.loc[sampled_hourslots_idx, "lot.closing_timeslot_interpolated"] = np.array(sampled_hourslots)
df["lot.closing_timeslot"].fillna(df["lot.closing_timeslot_interpolated"], inplace=True)

# %%
df["auction.end_timeslot"] = np.ceil((df["auction.end_date"] - TIMESTAMP_MIN).dt.total_seconds() / 3600)
df["auction.lot_min_end_timeslot"] = np.ceil(
    (df["auction.lot_min_end_date"] - TIMESTAMP_MIN).dt.total_seconds() / 3600
)
df["auction.lot_max_end_timeslot"] = np.ceil(
    (df["auction.lot_max_end_date"] - TIMESTAMP_MIN).dt.total_seconds() / 3600
)

# %%
auction_ids = df["auction.id"].unique()


def datetime_to_nearest(dt):
    return datetime(dt.year, dt.month, dt.day, 23, 59)


df["auction.lot_max_end_date_adjusted"] = df["auction.lot_max_end_date"].apply(datetime_to_nearest)
auction_min_end_dates = (
    df[["auction.id", "auction.lot_min_end_date", "auction.lot_max_end_date_adjusted"]]
    .drop_duplicates()
    .values
)

# %%
# auction_timeslot_scarcity = []
# for id, min_end_date, max_end_date in auction_min_end_dates:
#     auction_closing_range = pd.date_range(min_end_date, max_end_date, freq='h')
#     for dt in auction_closing_range:
#         df[(df['auction.id'] != id) & (df['lot.closing_time'])]
#         auction_timeslot_scarcity.append([id, dt])
# lot_scarcity_other_auctions = []
# for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
#     lot_auction = row["auction.id"]
#     lot_closing_timeslot = row["lot.closing_timeslot"]
#     lot_cat = row["lot.category"]
#     lot_subcat = row["lot.subcategory"]
#     scarcity_other_auction = (
#         (df["auction.id"] != lot_auction) & (df["lot.closing_timeslot"] == lot_closing_timeslot)
#     ).sum()
#     scarcity_cat_other_auction = (
#         (df["auction.id"] != lot_auction)
#         & (df["lot.closing_timeslot"] == lot_closing_timeslot)
#         & (df["lot.category"] == lot_cat)
#     ).sum()
#     scarcity_subcat_other_auction = (
#         (df["auction.id"] != lot_auction)
#         & (df["lot.closing_timeslot"] == lot_closing_timeslot)
#         & (df["lot.subcategory"] == lot_subcat)
#     ).sum()
#     lot_scarcity_other_auctions.append([row["lot.id"], scarcity_other_auction])
# %%
df_closing_timeslot_within_auction = (
    df.groupby(["lot.closing_timeslot", "auction.id"])
    .size()
    .reset_index(name="lot.num_closing_timeslot_within_auction")
)
df_closing_timeslot_total = (
    df.groupby(["lot.closing_timeslot"]).size().reset_index(name="lot.num_closing_timeslot")
)

df_closing_timeslot_category_within_auction = (
    df.groupby(["lot.closing_timeslot", "lot.category", "auction.id"])
    .size()
    .reset_index(name="lot.num_closing_timeslot_category_within_auction")
)
df_closing_timeslot_category_total = (
    df.groupby(["lot.closing_timeslot"]).size().reset_index(name="lot.num_closing_timeslot_category")
)

df_closing_timeslot_subcategory_within_auction = (
    df.groupby(["lot.closing_timeslot", "lot.subcategory", "auction.id"])
    .size()
    .reset_index(name="lot.num_closing_timeslot_subcategory_within_auction")
)
df_closing_timeslot_subcategory_total = (
    df.groupby(["lot.closing_timeslot"]).size().reset_index(name="lot.num_closing_timeslot_subcategory")
)

df = (
    df.merge(df_closing_timeslot_within_auction)
    .merge(df_closing_timeslot_total)
    .merge(df_closing_timeslot_category_within_auction)
    .merge(df_closing_timeslot_category_total)
    .merge(df_closing_timeslot_subcategory_within_auction)
    .merge(df_closing_timeslot_subcategory_total)
)
#%%
df["lot.num_closing_timeslot_other_auctions"] = (
    df["lot.num_closing_timeslot"] - df["lot.num_closing_timeslot_within_auction"]
)
df["lot.num_closing_timeslot_category_other_auctions"] = (
    df["lot.num_closing_timeslot_category"] - df["lot.num_closing_timeslot_category_within_auction"]
)
df["lot.num_closing_timeslot_subcategory_other_auctions"] = (
    df["lot.num_closing_timeslot_subcategory"] - df["lot.num_closing_timeslot_subcategory_within_auction"]
)

# # %% Add feature: number of lots of same (sub) category closing on same day
# df_category_scarcity = (df.groupby(["lot.category", "lot.closing_timeslot"])["lot.id"].nunique()).reset_index(
#     name="lot.category_scarcity"
# )
# df = pd.merge(df, df_category_scarcity)

# df_subcategory_scarcity = (
#     df.groupby(["lot.subcategory", "lot.closing_timeslot"])["lot.id"].nunique()
# ).reset_index(name="lot.subcategory_scarcity")
# df = pd.merge(df, df_subcategory_scarcity)

# # add feature 'lot.closing_count': number of lots closing on same day
# df_lots_closingcount = (
#     df.groupby(["lot.closing_timeslot"])["lot.id"]
#     .nunique()
#     .reset_index()
#     .rename(columns={"lot.id": "lot.scarcity"})
# )
# df = pd.merge(df_lots_closingcount, df)
#%%
df["lot.condition"] = df["lot.condition"].fillna("None")

# %%
USED_COLS = list(set(TRAIN_COLS_IS_SOLD).union(TRAIN_COLS_REVENUE))
# %%
missing_revenue_per_auction = df.isna().groupby(df["auction.id"], sort=False).sum()["lot.revenue"]
data = df[USED_COLS + ["lot.id", "auction.id", "auction.lot_min_end_date", "auction.lot_max_end_date"]]
sample_auctions = (
    missing_revenue_per_auction[missing_revenue_per_auction == 0].sample(25, random_state=42).index
)
sample_data = data[data["auction.id"].isin(sample_auctions)]

pd.get_dummies(sample_data).to_csv("./data/sample_auctions_25.csv.gz", index=False)

# %% Classification for 'is_sold': train/test/validation split
train_data = pd.get_dummies(df[TRAIN_COLS_IS_SOLD].dropna())
y = train_data["lot.is_sold"]
X = train_data.drop("lot.is_sold", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# %%
space = {
    "criterion": hyperopt.hp.choice("criterion", ["gini", "entropy"]),
    "splitter": hyperopt.hp.choice("splitter", ["best", "random"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 50),
    "min_samples_split": hyperopt.hp.uniformint("min_samples_split", 20, 100),
    "max_features": hyperopt.hp.uniform("max_features", 0.6, 0.99),
}
run_hyperopt(DecisionTreeClassifier, X_train, y_train, space, mode="clf", max_evals=100)

# %% Train/score optimal Decision Tree
clf = DecisionTreeClassifier(
    criterion="gini", max_depth=28, max_features=0.79, min_samples_split=28, splitter="random"
)
clf.fit(X_train, y_train)
plot_confusion_matrix(clf, X_test, y_test)
plot_roc_curve(clf, X_test, y_test)
clf.score(X_test, y_test)


# %%
with open("./models/dec_tree_clf.pkl", "wb") as f:
    pickle.dump(clf, f)

#%% Classifier: gradient boosting
space = {
    "n_estimators": hyperopt.hp.uniformint("n_estimators", 5, 100),
    "max_samples": hyperopt.hp.uniform("max_samples", 0, 1),
    "max_features": hyperopt.hp.uniform("max_features", 0, 1),
    "bootstrap": hyperopt.hp.choice("bootstrap", [False, True]),
    "bootstrap_features": hyperopt.hp.choice("bootstrap_features", [False, True]),
    "oob_score": hyperopt.hp.choice("oob_score", [False, True]),
    "warm_start": hyperopt.hp.choice("warm_start", [False, True]),
}
run_hyperopt(GradientBoostingClassifier, X_train, y_train, space, mode="clf", max_evals=100)

#%% Classifier: random forest
space = {
    "n_estimators": hyperopt.hp.uniformint("n_estimators", 5, 200),
    "criterion": hyperopt.hp.choice("criterion", ["gini", "entropy"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 50),
    "min_samples_split": hyperopt.hp.uniform("min_samples_split", 0, 0.9999),
    "min_weight_fraction_leaf": hyperopt.hp.uniform("min_weight_fraction_leaf", 0, 0.9999),
    "max_features": hyperopt.hp.uniform("max_features", 0, 0.9999),
    "bootstrap": hyperopt.hp.choice("bootstrap", [False, True]),
    "oob_score": hyperopt.hp.choice("oob_score", [False, True]),
}
run_hyperopt(RandomForestClassifier, X_train, y_train, space, mode="clf", max_evals=100)
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
train_data[["lot.revenue"]]
# %%


# %%
pd.Series(reg.predict(X_reg)).nunique()
# %%

# TODO pickle DecisionTreeClassifier & DecisionTreeRegressor
# TODO decision variables:
#  - lot closing timeslot (hourly)
#  - lot.starting_price
