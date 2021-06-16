# %% Import packages
from ensurepip import bootstrap
from random import uniform
from scipy.sparse.construct import rand
from sympy import hyper
import tqdm
from datetime import datetime
import pickle
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    plot_roc_curve,
    r2_score,
    plot_confusion_matrix,
    mean_absolute_percentage_error as mape,
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


#%%
df.columns

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
# %% If lot is not sold, set revenue to 0 and convert to nan
df["lot.revenue"] = ((df["lot.is_sold"]) * df["lot.revenue"]).replace(0, np.nan)

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
df_closing_timeslot_total = (
    df.groupby(["lot.closing_timeslot"]).size().reset_index(name="lot.num_closing_timeslot")
)


df_closing_timeslot_category_total = (
    df.groupby(["lot.closing_timeslot", "lot.category"])
    .size()
    .reset_index(name="lot.num_closing_timeslot_category")
)


df_closing_timeslot_subcategory_total = (
    df.groupby(["lot.closing_timeslot", "lot.subcategory"])
    .size()
    .reset_index(name="lot.num_closing_timeslot_subcategory")
)

df = (
    df.merge(df_closing_timeslot_total)
    .merge(df_closing_timeslot_category_total)
    .merge(df_closing_timeslot_subcategory_total)
)


#%%
auction_closing_available_timeslots = []
for id, min_end_date, max_end_date in auction_min_end_dates:
    min_end_date = datetime(min_end_date.year, min_end_date.month, min_end_date.day, min_end_date.hour)
    auction_closing_range = pd.date_range(min_end_date, max_end_date, freq="h")

    aux_df = pd.DataFrame({"auction.closing_timeslot": auction_closing_range})
    aux_df["auction.id"] = id
    aux_df["auction.num_closing_timeslot_other_auctions"] = np.nan
    aux_df["auction.num_closing_timeslot_category_other_auctions"] = np.nan
    aux_df["auction.num_closing_timeslot_subcategory_other_auctions"] = np.nan
    auction_closing_available_timeslots.append(aux_df)
df_auction_closing_available_timeslots = pd.concat(auction_closing_available_timeslots)
# Change the timeslot from datetime to timeslot number
df_auction_closing_available_timeslots["auction.closing_timeslot"] = np.ceil(
    (df_auction_closing_available_timeslots["auction.closing_timeslot"] - TIMESTAMP_MIN).dt.total_seconds()
    / 3600
)
df_auction_closing_available_timeslots.reset_index(drop=True, inplace=True)
df_auction_closing_available_timeslots
# %%
for i, row in tqdm.tqdm(
    df_auction_closing_available_timeslots.iterrows(), total=df_auction_closing_available_timeslots.shape[0]
):
    id = row["auction.id"]
    # print(row)
    # raise ValueError("TEST")
    closing_timeslot = row["auction.closing_timeslot"]
    df_same_timeslot = df[df["lot.closing_timeslot"] == closing_timeslot]
    df_same_time_auction = df_same_timeslot[df_same_timeslot["auction.id"] == id]
    # if df_same_time_auction.shape[0] > 0:
    df_auction_closing_available_timeslots.iloc[i] = [
        closing_timeslot,
        id,
        (df_same_timeslot.shape[0] - df_same_time_auction.shape[0]),
        {
            category: (
                df_same_timeslot[df_same_timeslot["lot.category"] == category].shape[0]
                - df_same_time_auction[df_same_time_auction["lot.category"] == category].shape[0]
            )
            for category in df_same_time_auction["lot.category"].unique()
        },
        {
            sub_category: (
                df_same_timeslot[df_same_timeslot["lot.subcategory"] == sub_category].shape[0]
                - df_same_time_auction[df_same_time_auction["lot.subcategory"] == sub_category].shape[0]
            )
            for sub_category in df_same_time_auction["lot.subcategory"].unique()
        },
    ]
# %%
df_auction_closing_available_timeslots.to_csv("Data/auction_timeslot_info.csv", index=False)
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
# missing_revenue_per_auction = df.isna().groupby(df["auction.id"], sort=False).sum()["lot.revenue"]
incorrect_auctions_sold_revenue = df[(df["lot.revenue"].isna()) & (df["lot.is_sold"] == 1)][
    "auction.id"
].unique()
possible_samples = list(set(df["auction.id"].unique()).intersection(set(incorrect_auctions_sold_revenue)))
np.random.seed(42)
sample_auctions = np.random.choice(possible_samples, 25, replace=False)
data = df[USED_COLS + ["lot.id", "auction.id", "auction.lot_min_end_date", "auction.lot_max_end_date"]]

sample_data = data[data["auction.id"].isin(sample_auctions)]

pd.get_dummies(sample_data).to_csv("./data/sample_auctions_25.csv.gz", index=False)
# %% Drop the samples that we use for the prescriptive section
df = df[~df["auction.id"].isin(sample_auctions)].reset_index(drop=True)
# TODO Drop the samples that we use for the prescriptive section
# %% Classification for 'is_sold': train/test/validation split
train_data = pd.get_dummies(df[TRAIN_COLS_IS_SOLD].dropna())
y = train_data["lot.is_sold"]
X = train_data.drop("lot.is_sold", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)


# %%
space = {
    "criterion": hyperopt.hp.choice("criterion", ["gini", "entropy"]),
    "splitter": hyperopt.hp.choice("splitter", ["best", "random"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 50),
    "min_samples_split": hyperopt.hp.uniformint("min_samples_split", 20, 100),
    "max_features": hyperopt.hp.uniform("max_features", 0.6, 0.99),
    "random_state": 0,
}
run_hyperopt(DecisionTreeClassifier, X_train, y_train, space, mode="clf", max_evals=200)

# %% Train/score optimal Decision Tree
clf = DecisionTreeClassifier(
    criterion="gini", max_depth=35, max_features=0.82, min_samples_split=56, splitter="random", random_state=0
)
clf.fit(X_train, y_train)
plot_confusion_matrix(clf, X_test, y_test)
plot_roc_curve(clf, X_test, y_test)
clf.score(X_test, y_test)


# %%
with open("./models/dec_tree_clf.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("./models/clf_X_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

#%% Classifier: gradient boosting
space = {
    "loss": hyperopt.hp.choice("loss", ["deviance", "exponential"]),
    "learning_rate": hyperopt.hp.uniform("learning_rate", 0.0001, 0.5),
    "n_estimators": hyperopt.hp.uniformint("n_estimators", 5, 200),
    "subsample": hyperopt.hp.uniform("subsample", 0, 1),
    "criterion": hyperopt.hp.choice("criterion", ["friedman_mse", "mse"]),
    "min_samples_split": hyperopt.hp.uniform("min_samples_split", 0, 1),
    "min_samples_leaf": hyperopt.hp.uniform("min_samples_leaf", 0, 0.5),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 50),
    "max_features": hyperopt.hp.uniform("max_features", 0, 0.9999),
    "warm_start": hyperopt.hp.choice("warm_start", [False, True]),
    "validation_fraction": 0.2,
    "n_iter_no_change": 10,
}
run_hyperopt(GradientBoostingClassifier, X_train, y_train, space, mode="clf", max_evals=20)
#%%
clf = GradientBoostingClassifier(
    criterion="friedman_mse",
    learning_rate=0.3,
    loss="deviance",
    max_depth=17,
    max_features=0.65,
    min_samples_leaf=0.012,
    min_samples_split=0.14,
    n_estimators=85,
    subsample=0.93,
    warm_start=False,
)
clf.fit(X_train, y_train)
plot_confusion_matrix(clf, X_test, y_test)
plot_roc_curve(clf, X_test, y_test)
clf.score(X_test, y_test)
#%% Classifier: random forest
space = {
    "n_estimators": hyperopt.hp.uniformint("n_estimators", 5, 100),
    "criterion": hyperopt.hp.choice("criterion", ["gini", "entropy"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 50),
    "min_samples_split": hyperopt.hp.uniform("min_samples_split", 0, 0.9999),
    "min_weight_fraction_leaf": hyperopt.hp.uniform("min_weight_fraction_leaf", 0, 0.5),
    "max_features": hyperopt.hp.uniform("max_features", 0, 0.9999),
    "bootstrap": hyperopt.hp.choice("bootstrap", [False, True]),
    # "oob_score": hyperopt.hp.choice("oob_score", [False, True]),
    "n_jobs": -1,
}
run_hyperopt(RandomForestClassifier, X_train, y_train, space, mode="clf", max_evals=200)

#%%
clf = RandomForestClassifier(
    bootstrap=True,
    criterion="entropy",
    max_depth=41,
    max_features=0.5,
    min_samples_split=0.015,
    min_weight_fraction_leaf=0.004,
    n_estimators=100,
)
clf.fit(X_train, y_train)
plot_confusion_matrix(clf, X_test, y_test)
plot_roc_curve(clf, X_test, y_test)
clf.score(X_test, y_test)
# %% Regression for 'lot.revenue': train/test/validation split
space = {
    "criterion": "friedman_mse",
    "splitter": hyperopt.hp.choice("splitter", ["best", "random"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 50),
    "min_samples_leaf": hyperopt.hp.uniformint("min_samples_leaf", 1, 1000),
    "max_features": hyperopt.hp.uniform("max_features", 0.3, 0.99),
    "random_state": 0,
}

train_data_reg = pd.get_dummies(df[TRAIN_COLS_REVENUE].dropna())
y = train_data_reg["lot.revenue"]
X = train_data_reg.drop("lot.revenue", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
# %%
run_hyperopt(DecisionTreeRegressor, X_train, y_train, space, mode="reg", max_evals=200)
# %%
reg = DecisionTreeRegressor(
    criterion="friedman_mse",
    splitter="best",
    max_depth=46,
    max_features=0.91,
    min_samples_leaf=3,
    random_state=0,
)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)
r2_score(y_test, reg.predict(X_test)), mape(y_test, reg.predict(X_test))

# %%
with open("./models/dec_tree_reg.pkl", "wb") as f:
    pickle.dump(reg, f)
with open("./models/reg_X_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# %%
space = {
    "loss": hyperopt.hp.choice("loss", ["ls", "lad", "huber", "quantile"]),
    "learning_rate": hyperopt.hp.uniform("learning_rate", 0.001, 0.2),
    "n_estimators": hyperopt.hp.uniformint("n_estimators", 50, 150),
    "subsample": hyperopt.hp.uniform("subsample", 0.5, 1),
    "criterion": hyperopt.hp.choice("criterion", ["friedman_mse", "mse"]),
    "min_samples_split": hyperopt.hp.uniformint("min_samples_split", 2, 100),
    "max_features": hyperopt.hp.uniform("max_features", 0.2, 1),
}
run_hyperopt(GradientBoostingRegressor, X_train, y_train, space, mode="reg", max_evals=200)

reg = GradientBoostingRegressor(
    loss="ls",
    learning_rate=0.0082,
    n_estimators=100,
    subsample=0.942,
    criterion="friedman_mse",
    min_samples_split=2,
    max_features=0.85,
)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)
r2_score(y_test, reg.predict(X_test)), mape(y_test, reg.predict(X_test))


# %%
space = {
    "n_estimators": hyperopt.hp.uniformint("n_estimators", 10, 150),
    "bootstrap": hyperopt.hp.choice("bootstrap", [False, True]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 2, 50),
    "min_samples_split": hyperopt.hp.uniformint("min_samples_split", 2, 100),
    "min_samples_leaf": hyperopt.hp.uniformint("min_samples_leaf", 2, 100),
    "max_features": hyperopt.hp.uniform("max_features", 0.2, 1),
    "n_jobs": -1,
}
run_hyperopt(RandomForestRegressor, X_train, y_train, space, mode="reg", max_evals=200)
reg = RandomForestRegressor(
    n_estimators=100,
    bootstrap=True,
    max_depth=40,
    min_samples_split=20,
    min_samples_leaf=2,
    max_features=0.76,
    n_jobs=-1,
)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)
r2_score(y_test, reg.predict(X_test)), mape(y_test, reg.predict(X_test))

# %%
