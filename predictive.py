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

from imblearn.under_sampling import RandomUnderSampler 

import hyperopt
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
df.drop_duplicates('lot.id', inplace=True)

# %%
df = df.sort_values(
    by=["auction.id", "lot.id", "bid.date"]).reset_index(drop=True)

# %%
df = df[[
    'auction.country_isocode', 'auction.is_homedelivery', 'auction.sourcing_company',
    'auction.is_public', 'lot.auction_lot_number', 'lot.newprice', 'lot.taxrate', 
    'lot.subcategory', 'lot.category', 'lot.starting_at_1EUR', 'lot.brand', 
    'lot.condition', 'lot.category_code', 'project.business_line', 'project.business_unit',
    'project.is_homedelivery', 'project.is_public', 'lot.start_amount', 'lot.is_sold']]

    ### new price dubbel?

train_data = df.dropna()
y = train_data['lot.is_sold']
X = train_data.drop('lot.is_sold', axis=1).select_dtypes("float")
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

rus = RandomUnderSampler(random_state=0)
X_res, y_res = rus.fit_resample(X_train, y_train)

# %%
criterion = ["gini", "entropy"]
min_sample_leaf = [1, 3, 6, 9, 12]

space = {
    "max_depth": hyperopt.hp.quniform("max_depth", 1, 15, 1),
    "criterion": hyperopt.hp.choice("criterion", criterion),
    "min_samples_leaf": hyperopt.hp.choice("min_samples_leaf", min_sample_leaf)
}

def hyperopt_objective_tuner(params):
    clf = DecisionTreeClassifier(**params)
    clf.fit(X_res, y_res)
    accuracy = clf.score(X_val, y_val)
    # accuracy = cross_val_score(model, X_test, y_test, scoring="accuracy").mean()

    # Hyperopic minimizes the function. Therefore, a negative sign in the accuracy
    return {"loss": -accuracy, "status": hyperopt.STATUS_OK}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    fn=hyperopt_objective_tuner,
    space=space,
    algo=hyperopt.tpe.suggest,
    max_evals=10,
    trials=trials
)
print(f"Best setting: {best}")

clf = DecisionTreeClassifier(criterion=criterion[best['criterion']], max_depth=best['max_depth'], min_samples_leaf=min_sample_leaf[best['min_samples_leaf']])
clf.fit(X_res, y_res)
print(f'Test results: {clf.score(X_test, y_test)}')
