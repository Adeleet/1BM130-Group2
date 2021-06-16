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
import matplotlib.pyplot as plt

# %% Read dataset
df = pd.read_csv("data/data_cleaned.csv.gz")


# %%
USED_COLS = list(set(TRAIN_COLS_IS_SOLD).union(TRAIN_COLS_REVENUE))


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

#%%
plot_confusion_matrix(clf, X_test, y_test, values_format="2d")
plt.savefig("./figures/predictive_dec_tree_confusion_matrix.svg")

#%%
plot_roc_curve(clf, X_test, y_test)
plt.savefig("./figures/predictive_dec_tree_roc_auc_curve.svg")

clf.score(X_test, y_test)
# 0.8262788099161033


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

# %%
plot_confusion_matrix(clf, X_test, y_test, values_format="2d")
plt.savefig("./figures/predictive_gbclassifier_confusion_matrix.svg")

#%%
plot_roc_curve(clf, X_test, y_test)
plt.savefig("./figures/predictive_gbclassifier_roc_auc_curve.svg")
#%%
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

#%%
pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).plot.bar()
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
