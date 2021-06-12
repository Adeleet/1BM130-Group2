# %%
from locale import normalize
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    plot_roc_curve,
    roc_auc_score,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    mean_absolute_percentage_error as MAPE,
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

# %%
df.shape
# %% Add feature 'lot.rel_nr': lot position on website corrected for total #nr lots in auction
df.sort_values(["auction.id", "lot.number"], inplace=True)
auction_size = df.groupby("auction.id").size().reset_index(name="auction.num_lots")
df = pd.merge(df, auction_size)
df["lot.rel_nr"] = (df.groupby("auction.id").cumcount() + 1) / df["auction.num_lots"]
df.shape
# %%
# %% Add feature 'lot.revenue': revenue per lot
df["lot.revenue"] = df["bid.amount"] * df["bid.is_latest"] * df["bid.is_valid"]
lot_revenue = df.groupby("lot.id")["lot.revenue"].max().reset_index()
lot_revenue = lot_revenue.replace(0, np.nan)
df = pd.merge(df, lot_revenue, how='outer')
# %%
df.sort_values(by='lot.closingdate', inplace=True)
# %%
df.columns

# %%
df.drop_duplicates("lot.id", inplace=True)

# %% Add feature 'lot.closingdate_day': lot closing date without time
# df["lot.closingdate_day"] = pd.to_datetime(df["lot.closingdate"].dt.date)
# %% Sort first by auction, then by lot, then by bid date
df = df.sort_values(by=["auction.id", "lot.id", "bid.date"]).reset_index(drop=True)
# %% Add feature 'lot.days_open': number of days a lot is open
df["lot.days_open"] = (df["lot.closingdate_day"] - df["lot.startingdate"]).dt.days


# %% Remove outlier with 384 days (mean=7.04, median=7)
df = df[df['lot.days_open'] < 50]

# %%
df[['lot.id', 'lot.startingdate', 'lot.closingdate']]
# pd.read_csv("data/raw_data/fact_lots.csv.gz").drop_duplicates("lot_id")["closingdate"].isna().value_counts()
# %%
df[['lot.closingdate', 'lot.closingdate_day']]
# %% Add feature 'lot.category_count_in_auction': number of lots of same category in auction
auction_cat_counts = df.groupby(["auction.id", "lot.category"]).size().reset_index()
auction_cat_counts.rename(columns={0: "lot.category_count_in_auction"}, inplace=True)
df = pd.merge(df, auction_cat_counts)
# %% Add feature 'lot.category_closing_count': number of lots of same category closing on same day
df_lots_scarcity = (df.groupby(["lot.category", "lot.closingdate_day"])["lot.id"].nunique()).reset_index(
    name="lot.category_closing_count"
)
df = pd.merge(df_lots_scarcity, df, how="outer")
df[["lot.category", "lot.closingdate_day", "lot.id", "lot.category_closing_count"]]
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
    "max_features": hyperopt.hp.uniform("max_features", 0.5, 0.99),
}
run_hyperopt(DecisionTreeClassifier, X_train, y_train, space, mode="clf")
# %% Train/score optimal Decision Tree
clf = DecisionTreeClassifier(
    criterion="gini", max_depth=45, max_features=0.92, min_samples_split=65, splitter="random"
)
clf.fit(X_train, y_train)
plot_confusion_matrix(clf, X_test, y_test)
plot_roc_curve(clf, X_test, y_test)
clf.score(X_test, y_test)

# %% Save linear constraints from optimal decision tree
# with open("predictive_dec_tree_constraints.txt", "w") as f:
#     constraints = tree.export_text(clf, feature_names=list(X.columns), show_weights=True)
#     f.write(constraints)

# %% Regression for 'lot.revenue': train/test/validation split
space = {
    "criterion": "friedman_mse",
    "splitter": hyperopt.hp.choice("splitter", ["best", "random"]),
    "max_depth": hyperopt.hp.uniformint("max_depth", 1, 30),
    "min_samples_leaf": hyperopt.hp.uniformint("min_samples_leaf", 1, 1000),
    "max_features": hyperopt.hp.uniform("max_features", 0.3, 0.99),
}
train_data_reg = pd.get_dummies(df[TRAIN_COLS_REVENUE].dropna())
y = train_data_reg["lot.revenue"]
X = train_data_reg.drop("lot.revenue", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
run_hyperopt(DecisionTreeRegressor, X_train, y_train, space, mode="reg", max_evals=250)
# %%
scores = []
x_vals = list(range(2, 30))
for num_leaves in x_vals:
    reg = DecisionTreeRegressor(max_leaf_nodes=num_leaves, criterion='friedman_mse')
    reg.fit(X_train, y_train)
    df_ada = pd.DataFrame({"pred": reg.predict(X_train)})
    df_ada['y'] = y_train.values

    probs = df_ada['pred'].value_counts(normalize=True).reset_index()
    probs.columns = ["pred", "class_pr"]

    df_ada = df_ada.merge(probs)
    df_ada['err'] = (df_ada['y'] - df_ada['pred'])**2
    mse = df_ada['err'].mean()**(1 / 2)
    df_ada['class_prK'] = df_ada['class_pr'] * (1 - df_ada['class_pr'])
    df_ada['pred'].value_counts()
    s1 = (df_ada['err'] * df_ada['class_prK']).mean()
    # print(s1 * num_leaves, mse)
    scores.append(mse / (s1 * num_leaves))

# %%
plt.plot(x_vals, scores)
# %%
df_ada['pred'].value_counts()
# %%
reg = DecisionTreeRegressor(
    criterion="mse", splitter="best", max_depth=16, max_features=0.92, min_samples_split=4
)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)

# %%
with open("predictive_reg_tree_constraints.txt", "w") as f:
    constraints = tree.export_text(reg, feature_names=list(X_reg.columns), show_weights=True)
    f.write(constraints)


# %%
pd.Series(reg.predict(X_reg)).nunique()
# %%
