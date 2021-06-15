#%%
import pickle
from ast import literal_eval

import gurobipy as grb
import pandas as pd
from tqdm import tqdm

from lpclasslibrary import Leafnode, Lot, Node

# %% Define Auction id of sample that we want to run
SAMPLE_ID = 45453

# %% Load the list with the column names of the X data for model training
reg_train_columns = pickle.load(open("models/reg_X_columns.pkl", 'rb'))
clf_train_columns = pickle.load(open("models/clf_X_columns.pkl", 'rb'))

ALL_MODEL_CATEGORIES_CLF = set([feature for feature in clf_train_columns
                                if ("lot.category_" in feature)
                                and ("lot.category_count" not in feature)]) 
ALL_MODEL_CATEGORIES_REG = set([feature for feature in reg_train_columns
                                if ("lot.category_" in feature)
                                and ("lot.category_count" not in feature)])
ALL_MODEL_CATEGORIES = ALL_MODEL_CATEGORIES_CLF.union(ALL_MODEL_CATEGORIES_REG)

ALL_MODEL_SUBCATEGORIES_CLF = set([feature for feature in clf_train_columns
                                if ("lot.subcategory_" in feature)
                                and ("lot.subcategory_count" not in feature)]) 
ALL_MODEL_SUBCATEGORIES_REG = set([feature for feature in reg_train_columns
                                if ("lot.subcategory_" in feature)
                                and ("lot.subcategory_count" not in feature)])
ALL_MODEL_SUBCATEGORIES = ALL_MODEL_SUBCATEGORIES_CLF.union(ALL_MODEL_SUBCATEGORIES_REG)

# %% Load sample dataset
samples = pd.read_csv("Data/sample_auctions_25.csv.gz")
#Select the data for the chosen sample
sample_data = samples[samples['auction.id'] == SAMPLE_ID].reset_index(drop=True)
# Remove the features that will be variables
sample_data = sample_data.drop(["lot.rel_nr",
                                "lot.num_closing_timeslot",
                                "lot.num_closing_timeslot_category",
                                "lot.num_closing_timeslot_subcategory"],
                               axis=1)
# Remove columns that are not relevant
sample_data = sample_data.drop([column for column in sample_data.columns
                                if column not in reg_train_columns],
                               axis=1)
# %% Load the lots into a dictionary
Lots = {}
for lot_nr, features in sample_data.iterrows():
    feature_dict = features.to_dict()
    non_present_features_from_model_training = (
        i for i in clf_train_columns if (i not in feature_dict) 
        and (i not in ["lot.rel_nr","lot.num_closing_timeslot",
                       "lot.num_closing_timeslot_category",
                       "lot.num_closing_timeslot_subcategory"]))
    # Add the non present features to the features_dict
    for feature in non_present_features_from_model_training:
        feature_dict[feature] = 0
    Lots[lot_nr] = Lot(lot_nr, features=feature_dict)
    

#%% Load decision tree regressor for price
clf_price = pickle.load(open("models\dec_tree_reg.pkl", 'rb'))

children_left_price = clf_price.tree_.children_left
children_right_price = clf_price.tree_.children_right
feature_price = clf_price.tree_.feature
threshold_price = clf_price.tree_.threshold
value_price = clf_price.tree_.value

# %% Load decision tree classifier for is_sold
clf_sold = pickle.load(open("models\dec_tree_clf.pkl", 'rb'))

children_left_sold = clf_sold.tree_.children_left
children_right_sold = clf_sold.tree_.children_right
feature_sold = clf_sold.tree_.feature
threshold_sold = clf_sold.tree_.threshold
value_sold = clf_sold.tree_.value

# %% Load auction timeslot info
auction_time_df = pd.read_csv("Data/auction_timeslot_info.csv")
auction_time_df['auction.num_closing_timeslot_category_other_auctions'] = (
    auction_time_df['auction.num_closing_timeslot_category_other_auctions'].apply(
        lambda x: literal_eval(str(x))))
auction_time_df['auction.num_closing_timeslot_subcategory_other_auctions'] = (
    auction_time_df['auction.num_closing_timeslot_subcategory_other_auctions'].apply(
        lambda x: literal_eval(str(x))))

# Select only the info for this auction
sample_auction_time_df = auction_time_df[
    auction_time_df["auction.id"] == SAMPLE_ID]

# %% Retrieve all categories that are in the sample
# ALL_CATEGORIES = [column for column in sample_data.columns if (
#     "lot.category_" in column) and ("lot.category_count" not in column)]
# ALL_SUBCATEGORIES = [column for column in sample_data.columns if (
#     "lot.subcategory" in column) and ("lot.subcategory_count" not in column)]
# %% Create the auction parameters
big_M = 100_000_000
tau_min = int(min(sample_auction_time_df["auction.closing_timeslot"]))
tau_max = int(max(sample_auction_time_df["auction.closing_timeslot"]))
N = sample_data.shape[0]
thetadict = {}
bigthetadict = {}
Odict = {}
for tau in range(tau_min, tau_max + 1):
    # For each timeslot, find theta
    thetadict[tau] = int(
        sample_auction_time_df[
            sample_auction_time_df["auction.closing_timeslot"] == tau][
                "auction.num_closing_timeslot_other_auctions"])
    
    # Retrieve the big theta value for this timeslot and cats in data
    bigthetadict[tau] = sample_auction_time_df[
        sample_auction_time_df["auction.closing_timeslot"] == tau][
            "auction.num_closing_timeslot_category_other_auctions"].values[0]
    # Next 2 for loops to rename the keys
    to_delete = []
    new_keys = []
    for key in bigthetadict[tau].keys():
        to_delete.append(key)
        new_keys.append(f"lot.category_{key}")
    for i in range(len(to_delete)):
        bigthetadict[tau][new_keys[i]] = bigthetadict[tau][to_delete[i]]
        del bigthetadict[tau][to_delete[i]]
        
    categories_not_yet_in_dict = (
        key for key in ALL_MODEL_CATEGORIES if key not in bigthetadict[tau].keys())
    # Set the counts to 0 , for those that were not in the data
    for category in categories_not_yet_in_dict:
        bigthetadict[tau][category] = 0
        
    # Retrieve the \mathcal{O} value for this timeslot and subcats in data
    Odict[tau] = sample_auction_time_df[
        sample_auction_time_df["auction.closing_timeslot"] == tau][
            "auction.num_closing_timeslot_subcategory_other_auctions"].values[0]
    # Next 2 for loops to rename the keys
    to_delete = []
    new_keys = []
    for key in Odict[tau].keys():
        to_delete.append(key)
        new_keys.append(f"lot.subcategory_{key}")
    for i in range(len(to_delete)):
        Odict[tau][new_keys[i]] = Odict[tau][to_delete[i]]
        del Odict[tau][to_delete[i]]
    subcategories_not_yet_in_dict = (
        key for key in ALL_MODEL_SUBCATEGORIES if key not in Odict[tau].keys())
    # Set the counts to 0 , for those that were not in the data
    for subcategory in subcategories_not_yet_in_dict:
        Odict[tau][subcategory] = 0

# %% Calculate minimum and maximum feature values
feature_maxima = {}
for feature in Lots[0].features:     
    feature_maxima[feature] = max(l.features[feature] for l in Lots.values())
feature_maxima["lot.rel_nr"] = 1
feature_maxima["lot.num_closing_timeslot"] = 100000
feature_maxima["lot.num_closing_timeslot_category"] = 100000
feature_maxima["lot.num_closing_timeslot_subcategory"] = 100000

feature_minima = {}
for feature in Lots[0].features:
    feature_minima[feature] = min(l.features[feature] for l in Lots.values())
feature_minima["lot.rel_nr"] = 0
feature_minima["lot.num_closing_timeslot"] = 0
feature_minima["lot.num_closing_timeslot_category"] = 0
feature_minima["lot.num_closing_timeslot_subcategory"] = 0
# %%
#Create classes for the nodes and load in the leaves in the subtrees of each node
Nodes_price = {}
Leafnodes_price = {}
for node in range(len(value_price)):
    if children_left_price[node] != children_right_price[node]:
        Nodes_price[node] = Node(node, children_left_price, children_right_price,
                                 feature_price, threshold_price)
    else:
        Leafnodes_price[node] = Leafnode(node, value_price)
for node in Nodes_price:
    Nodes_price[node].find_subtree_leaves(Nodes_price, Leafnodes_price)

Nodes_sold = {}
Leafnodes_sold = {}
for node in range(len(value_sold)):
    if children_left_sold[node] != children_right_sold[node]:
        Nodes_sold[node] = Node(node, children_left_sold, children_right_sold,
                                feature_sold, threshold_sold)
    else:
        Leafnodes_sold[node] = Leafnode(node, value_sold)
for node in Nodes_sold:
    Nodes_sold[node].find_subtree_leaves(Nodes_sold, Leafnodes_sold)

#%%
lpmodel = grb.Model("1BM130 prescriptive analytics")
lpmodel.modelSense = grb.GRB.MAXIMIZE

#%%create the z^p_l_r variables
for leafnode in Leafnodes_price:
    myvars = {}
    for lot in Lots:
        myvar = lpmodel.addVar(vtype = grb.GRB.BINARY, name = f"z^p_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes_price[leafnode].set_z_vars(myvars)

#%%create the z^x_l_r variables
for leafnode in Leafnodes_sold:
    myvars = {}
    for lot in Lots:
        myvar = lpmodel.addVar(vtype = grb.GRB.BINARY, name = f"z^x_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes_sold[leafnode].set_z_vars(myvars)

#%%create the p, x, s, o, LotNrRel an ClosingCount variables
for lot in Lots:
    my_p_var = lpmodel.addVar(vtype = grb.GRB.CONTINUOUS, name = f"p_{lot}")
    my_x_var = lpmodel.addVar(vtype = grb.GRB.BINARY, name = f"x_{lot}")
    my_s_var = lpmodel.addVar(vtype = grb.GRB.CONTINUOUS, lb=1, name = f"s_{lot}")
    my_o_var = lpmodel.addVar(vtype = grb.GRB.CONTINUOUS, name = f"o_{lot}")
    my_LotNrRel_var = lpmodel.addVar(vtype = grb.GRB.CONTINUOUS, name = f"LotNrRel_{lot}")
    my_ClosingCount_var = lpmodel.addVar(vtype = grb.GRB.CONTINUOUS, name = f"ClosingCount_{lot}")
    my_ClosingCountCat_var = lpmodel.addVar(vtype = grb.GRB.CONTINUOUS, name = f"ClosingCountCat_{lot}")
    my_ClosingCountSub_var = lpmodel.addVar(vtype = grb.GRB.CONTINUOUS, name = f"ClosingCountSub_{lot}")
    Lots[lot].set_p_var(my_p_var)
    Lots[lot].set_x_var(my_x_var)
    Lots[lot].set_s_var(my_s_var)
    Lots[lot].set_o_var(my_o_var)
    Lots[lot].set_LotNrRel_var(my_LotNrRel_var)
    Lots[lot].set_ClosingCount_var(my_ClosingCount_var)
    Lots[lot].set_ClosingCountCat_var(my_ClosingCountCat_var)
    Lots[lot].set_ClosingCountSub_var(my_ClosingCountSub_var)

#%%Create y variables
for lot in Lots:
    my_y_vars = {}
    for i in range(1,N+1):
        my_y_var = lpmodel.addVar(vtype = grb.GRB.BINARY, name = f"y_{i},{lot}")
        my_y_vars[i] = my_y_var
    Lots[lot].set_y_vars(my_y_vars)

#%%Create q variables
for lot in Lots:
    my_q_vars = {}
    for tau in range(tau_min, tau_max +1):
        my_q_var = lpmodel.addVar(vtype = grb.GRB.BINARY, name = f"y_{tau},{lot}")
        my_q_vars[tau] = my_q_var
    Lots[lot].set_q_vars(my_q_vars)

#%%Create a variables
for lot in Lots:
    my_a_varss = {}
    for lot2 in Lots:
        my_a_vars = {}
        for tau in range(tau_min, tau_max+1):
            my_a_var = lpmodel.addVar(vtype= grb.GRB.BINARY, name = f"a_{lot},{lot2},{tau}")
            my_a_vars[tau] = my_a_var
        my_a_varss[lot2] = my_a_vars
    Lots[lot].set_a_vars(my_a_varss)

#%%Set the objective function (1)
lpmodel.setObjective(sum(Lots[lot].get_o_var() for lot in Lots))

#%%Set constraint (2)
for lot in Lots:
    for node in Nodes_sold:
        FV = Lots[lot].features[clf_train_columns[Nodes_sold[node].feature]]
        M_f = feature_maxima[clf_train_columns[Nodes_sold[node].feature]]
        sum_z_vars = sum(l.get_z_vars()[lot] 
                         for l in Nodes_sold[node].leaves_left_subtree)
        lpmodel.addConstr(
            FV + (M_f - Nodes_sold[node].threshold) * sum_z_vars <= M_f,
            name=f"Constraint (2) for lot {lot} and node {node} in the classification model"
            )
for lot in Lots:
    for node in Nodes_price:
        FV = Lots[lot].features[clf_train_columns[Nodes_price[node].feature]]
        M_f = feature_maxima[clf_train_columns[Nodes_price[node].feature]]
        sum_z_vars = sum(l.get_z_vars()[lot]
                         for l in Nodes_price[node].leaves_left_subtree)
        lpmodel.addConstr(
            FV + (M_f - Nodes_price[node].threshold) * sum_z_vars <= M_f,
            name=f"Constraint (2) for lot {lot} and node {node} in the regression model"
            )

#%%Set constraint (3)
for lot in Lots:
    for node in Nodes_sold:
        FV = Lots[lot].features[clf_train_columns[Nodes_sold[node].feature]]
        m_f = feature_minima[clf_train_columns[Nodes_sold[node].feature]]
        sum_z_vars = sum(l.get_z_vars()[lot]
                         for l in Nodes_sold[node].leaves_right_subtree)
        lpmodel.addConstr(
            FV + (m_f - Nodes_sold[node].threshold) * sum_z_vars >= m_f, 
            name=f"Constraint (3) for lot {lot} and node {node} in the classification model"
            )
for lot in Lots:
    for node in Nodes_price:
        FV = Lots[lot].features[clf_train_columns[Nodes_price[node].feature]]
        m_f = m_f = feature_minima[clf_train_columns[Nodes_price[node].feature]]
        sum_z_vars = sum(l.get_z_vars()[lot]
                         for l in Nodes_price[node].leaves_right_subtree)
        lpmodel.addConstr(
            FV + (m_f - Nodes_price[node].threshold) * sum_z_vars >= m_f, 
            name=f"Constraint (3) for lot {lot} and node {node} in the regression model"
            )

#%%Set constraint (4)
for lot in Lots:
    lpmodel.addConstr(
        sum(l.get_z_vars()[lot] for l in Leafnodes_sold.values()) == 1,
        name=f"Constraint (4) for lot {lot} and the classification model"
        )
    lpmodel.addConstr(
        sum(l.get_z_vars()[lot] for l in Leafnodes_price.values()) == 1,
        name=f"Constraint (4) for lot {lot} and the regression model"
        )

#%%Set constraint (5)
for lot in Lots:
    lpmodel.addConstr(
        Lots[lot].get_p_var() == sum(l.leaf_value * l.get_z_vars()[lot]
                                     for l in Leafnodes_price.values()),
        name=f"Constraint (5) for lot {lot}"
        )

#%%Set contraint (6)
for lot in Lots:
    lpmodel.addConstr(
        Lots[lot].get_x_var() == sum(l.leaf_value * l.get_z_vars()[lot]
                                     for l in Leafnodes_sold.values()),
        name=f"Constraint (6) for lot {lot}"
        )

#%%Set constraint (7)
for lot in Lots:
    lpmodel.addConstr(
        Lots[lot].get_o_var() <= big_M * Lots[lot].get_x_var(),
        name=f"Constraint (7) for lot {lot}"
        )

#%%Set constraint (8)
for lot in Lots:
    lpmodel.addConstr(
        Lots[lot].get_o_var() <= Lots[lot].get_p_var(),
        name=f"Constraint (8) for lot {lot}"
        )

#%%Set constraint (9)
for lot in Lots:
    lpmodel.addConstr(
        Lots[lot].get_o_var() >= (Lots[lot].get_p_var() 
                                  - big_M * (1- Lots[lot].get_x_var())),
        name=f"Constraint (9) for lot {lot}")

#%%Set constraint (10)
for i in range(1,N+1):
    lpmodel.addConstr(
        sum(lot.get_y_vars()[i] for lot in Lots.values()) == 1,
        name=f"Constraint (10) for location {i}"
        )

#%%Set constraint (11)
for lot in Lots:
    lpmodel.addConstr(
        sum(Lots[lot].get_y_vars()[i] for i in range(1, N+1)) == 1,
        name=f"Constraint (11) for lot {lot}"
        )

#%%Set constraint (12)
for lot in Lots:
    lpmodel.addConstr(
        sum(i*Lots[lot].get_y_vars()[i]
            for i in range(1, N+1)) / N == Lots[lot].get_LotNrRel_var(),
        name=f"Constraint (12) for lot {lot}"
        )

#%%Set constraints (13), (14) and (15)
for lot in Lots:
    for lot2 in Lots:
        for tau in range(tau_min, tau_max+1):
            lpmodel.addConstr(
                Lots[lot].get_a_vars()[lot2][tau] <= Lots[lot].get_q_vars()[tau],
                name=f"Constraint (13) for lots {lot}, {lot2}, and time slot {tau}"
                )
            lpmodel.addConstr(
                Lots[lot].get_a_vars()[lot2][tau] <= Lots[lot2].get_q_vars()[tau],
                name=f"Constraint (14) for lots {lot}, {lot2}, and time slot {tau}"
                )
            lpmodel.addConstr(
                (Lots[lot].get_a_vars()[lot2][tau] >=
                 Lots[lot].get_q_vars()[tau] + Lots[lot2].get_q_vars()[tau] - 1),
                name=f"Constraint (15) for lots {lot}, {lot2}, and time slot {tau}"
                )

#%%Set constraint (16)
for lot in Lots:
    lpmodel.addConstr(
        sum(Lots[lot].get_q_vars()[tau] for tau in range(tau_min, tau_max+1)) == 1,
        name = f"Constraint (18) for lot {lot}"
    )

#%%Set constraint (17)
for lot in Lots:
    lpmodel.addConstr(
        sum(thetadict[tau]*Lots[lot].get_q_vars()[tau]
            + sum(Lots[lot].get_a_vars()[lot2][tau] for lot2 in Lots)
            for tau in range(tau_min, tau_max+1))
        == Lots[lot].get_ClosingCount_var(),
        name=f"Constraint (17) for lot {lot}"
        )

#%%Set constraint (18)
for lot in Lots:
    lpmodel.addConstr(
        sum(
            sum(bigthetadict[tau][c] * Lots[lot].kappas[c] * Lots[lot].get_q_vars()[tau] 
                + Lots[lot].kappas[c]
                * sum(Lots[lot].get_a_vars()[lot2][tau] * Lots[lot2].kappas[c]
                      for lot2 in Lots
                      ) 
                for c in ALL_MODEL_CATEGORIES
                )
            for tau in range(tau_min, tau_max+1)
            ) == Lots[lot].get_ClosingCountCat_var(),
        name=f"Constraint (18) for lot {lot}"
        )

#%%Set constraint (19)
for lot in tqdm(Lots):
    lpmodel.addConstr(
        sum(
            sum(Odict[tau][sigma] * Lots[lot].ks[sigma] * Lots[lot].get_q_vars()[tau]
                + Lots[lot].ks[sigma]
                * sum(Lots[lot].get_a_vars()[lot2][tau] * Lots[lot2].ks[sigma]
                      for lot2 in Lots
                      ) 
                for sigma in ALL_MODEL_SUBCATEGORIES
                )
            for tau in range(tau_min, tau_max+1)
            ) == Lots[lot].get_ClosingCountSub_var(),
        name = f"Constraint (19) for lot {lot}"
        )
# %% Update the model
lpmodel.update()
# %% Write the model to .lp file
lpmodel.write("1BM130.lp")
# %% Optimize the model
lpmodel.optimize()
# %% Display the optimal configuration and its results
for lot in Lots.values():
    for tau in lot.get_q_vars():
        if lot.get_q_vars()[tau].x == 1:
            if lot.get_x_var().x == 1:
                print(f"Lot {lot.id} was listed for a starting pice of {lot.get_s_var().x}, ended in timeslot {tau}, and was sold for price {lot.get_p_var().x}")
            else:
                print(f"Lot {lot.id} was listed for a starting pice of {lot.get_s_var().x}, ended in timeslot {tau}, and was not sold")

print(f"The total revenue of the auction is {lpmodel.objVal}")
# %%