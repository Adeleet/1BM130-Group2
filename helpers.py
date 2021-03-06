import hyperopt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_percentage_error as mape, r2_score
import numpy as np


def run_hyperopt(algorithm, X, y, space, mode, max_evals=100):
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

    def hyperopt_objective_tuner(params):
        model = algorithm(**params)
        model.fit(X_train, y_train)
        if mode == "clf":
            loss = model.score(X_val, y_val)
            # Hyperopic minimizes the function. Therefore, a negative sign in the accuracy
            return {"loss": -loss, "status": hyperopt.STATUS_OK}
            # loss = roc_auc_score(y_val, y_pred)
        elif mode == "reg":
            y_pred = model.predict(X_val)
            loss = mape(y_val, y_pred)
            return {"loss": loss, "status": hyperopt.STATUS_OK}
        else:
            raise ValueError(f"mode must be 'clf/classification' or 'reg/regression', got '{mode}")

    best = hyperopt.fmin(
        fn=hyperopt_objective_tuner,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        trials=hyperopt.Trials(),
    )
    print(f"Best setting: {best}")
    return best


# def train_test_split_undersampling(dataset, stratified_label):
#     unique_vals = list(y_train.value_counts().index)
#     min_freq = y_train.value_counts().min()
#     samples = []
#     for uniq_value in unique_vals:
#         sample = dataset[dataset[stratified_label] == uniq_value].sample(min_freq)
#         samples.append(sample)
#     return pd.concat(samples)
