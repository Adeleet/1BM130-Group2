#%%
import gurobipy as grb
from sklearn.tree  import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import pickle
#%%
# clf_price = DecisionTreeRegressor()

# children_left_price = clf_price.tree_.children_left
# children_right_price = clf_price.tree_.children_right
# feature_price = clf_price.tree_.feature
# threshold_price = clf_price.tree_.threshold
# value_price = clf_price.tree_.value

clf_sold = pickle.load(open("models\dec_tree_clf.pkl", 'rb'))

children_left_sold = clf_sold.tree_.children_left
children_right_sold = clf_sold.tree_.children_right
feature_sold = clf_sold.tree_.feature
threshold_sold = clf_sold.tree_.threshold
value_sold = clf_sold.tree_.value

# #Auction parameters
# tau_min = 1
# tau_max = 10
# N = 120
# thetadict = {}
# bigthetadict = {}
# Odict = {}
# for tau in range(tau_min, tau_max + 1):
#     thetadict[tau] = theta
#     bigthetadict[tau] = {}
#     for c in C:
#         bigthetadict[tau][c] = Theta
#     Odict[tau] = {}
#     for sigma in S:
#         Odict[tau][sigma] = O
#%%
class Lot:
    def __init__(self, lot_id, features:dict):
        self.id = lot_id
        self.features = features
        kappas = {}
        ks = {}
        
    def set_end_time(self, end_time):
        self.end_time = end_time

    def get_end_time(self):
        return self.end_time

    def set_starting_price(self, starting_price):
        self.starting_price = starting_price

    def get_starting_price(self):
        return self.starting_price

    def get_id(self):
        return self.id

    def set_p_var(self, p_var):
        self.p_var = p_var

    def get_p_var(self):
        return self.p_var

    def set_x_var(self, x_var):
        self.x_var = x_var

    def get_x_var(self):
        return self.x_var

    def set_s_var(self, s_var):
        self.s_var = s_var

    def get_s_var(self):
        return self.s_var

    def set_y_vars(self, y_vars):
        self.y_vars = y_vars

    def get_y_vars(self):
        return self.y_vars

    def set_q_vars(self, q_vars):
        self.q_vars = q_vars

    def get_q_vars(self):
        return self.q_vars   

    def set_LotNrRel_var(self, LotNrRel_var):
        self.LotNrRel_var = LotNrRel_var
        #self.features[featurenumber of LotNrRel] = LotNrRel_var

    def get_LotNrRel_var(self):
        return self.LotNrRel_var
        #return self.features[featurenumber of LotNrRel]

    def set_ClosingCount_var(self, ClosingCount_var):
        self.ClosingCount_var = ClosingCount_var

    def get_ClosingCount_var(self):
        return self.ClosingCount_var

    def set_ClosingCountCat_var(self, ClosingCountCat_var):
        self.ClosingCountCat_var = ClosingCountCat_var

    def get_ClosingCountCat_var(self):
        return self.ClosingCountCat_var

    def set_ClosingCountSub_var(self, ClosingCountSub_var):
        self.ClosingCountSub_var = ClosingCountSub_var

    def get_ClosingCountSub_var(self):
        return self.ClosingCountSub_var

    def set_a_vars(self, a_vars):
        self.a_vars = a_vars

    def get_a_vars(self):
        return self.a_vars

    def set_o_var(self, o_var):
        self.o_var = o_var

    def get_o_var(self):
        return self.o_var

class Node:
    def __init__(self, id, children_left, children_right, feature, threshold):
        self.id = id
        self.feature = feature[self.id]
        self.leftchild = children_left[self.id]
        self.rightchild = children_right[self.id]
        self.threshold = threshold[self.id]
        
    def find_subtree_leaves(self, nodes:dict, leafnodes:dict):
        '''
        create the list of leaf nodes that are part of the left and right subtree
        '''
        #Start by initializing the nodes in the left subtree by taking the left child of the node
        try:
            leftsubtreenodes = [nodes[self.leftchild]]
            leftsubtreeleaves = []
        except KeyError:
            leftsubtreenodes = []
            leftsubtreeleaves = [leafnodes[self.leftchild]]
        #Iterate over all split nodes in the left subtree
        for i in leftsubtreenodes:
            #Try to add the children of the split node as nodes to the queue, if this is not possible, it are leaves, so add them to the list of leaves
            try:
                leftsubtreenodes.append(nodes[i.leftchild])
            except KeyError:
                leftsubtreeleaves.append(leafnodes[i.leftchild])
            try:
                leftsubtreenodes.append(nodes[i.rightchild])
            except KeyError:
                leftsubtreeleaves.append(leafnodes[i.rightchild])
        self.leaves_left_subtree = leftsubtreeleaves

        #Repeat for the right subtree
        try:
            rightsubtreenodes = [nodes[self.rightchild]]
            rightsubtreeleaves = []
        except KeyError:
            rightsubtreenodes = []
            rightsubtreeleaves = [leafnodes[self.rightchild]]
        for i in rightsubtreenodes:
            try:
                rightsubtreenodes.append(nodes[i.leftchild])
            except KeyError:
                rightsubtreeleaves.append(leafnodes[i.leftchild])
            try:
                rightsubtreenodes.append(nodes[i.rightchild])
            except KeyError:
                rightsubtreeleaves.append(leafnodes[i.rightchild])
        self.leaves_right_subtree = rightsubtreeleaves

    def get_threshold(self):
        return self.threshold

    def get_left_subtree_leaves(self):
        return self.leaves_left_subtree

    def get_right_subtree_leaves(self):
        return self.leaves_right_subtree

class Leafnode:
    def __init__(self, id, value):
        self.id = id
        self.leaf_value = value[self.id]
        self.z_vars = {}
    
    def set_z_vars(self, lot_id, z_vars):
        self.z_vars[lot_id] = z_vars

    def get_z_vars(self):
        return self.z_vars

    def get_leaf_value(self):
        return self.leaf_value

# #Create classes for the nodes and load in the leaves in the subtrees of each node
# Nodes_price = {}
# Leafnodes_price = {}
# for node in range(len(value_price)):
#     if children_left_price[node] != children_right_price[node]:
#         Nodes_price[node] = Node(node, children_left_price, children_right_price, feature_price, threshold_price)
#     else:
#         Leafnodes_price[node] = Leafnode(node, value_price)

Nodes_sold = {}
Leafnodes_sold = {}
for node in range(len(value_sold)):
    if children_left_sold[node] != children_right_sold[node]:
        Nodes_sold[node] = Node(node, children_left_sold, children_right_sold, feature_sold, threshold_sold)
    else:
        Leafnodes_sold[node] = Leafnode(node, value_sold)
for node in Nodes_sold:
    Nodes_sold[node].find_subtree_leaves(Nodes_sold, Leafnodes_sold)

#%%
lpmodel = grb.Model("1BM130 prescriptive analytics")
lpmodel.modelSense = grb.GRB.MAXIMIZE

#create the z^p_l_r variables
for leafnode in Leafnodes_price:
    myvars = {}
    for lot in Lots:
        myvar = lpmodel.addVar(vtype = grb.GRB.binary, name = f"z^p_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes_price[leafnode].set_z_vars(myvars)

#create the z^x_l_r variables
for leafnode in Leafnodes_sold:
    myvars = {}
    for lot in Lots:
        myvar = lpmodel.addVar(vtype = grb.GRB.binary, name = f"z^x_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes_sold[leafnode].set_z_vars(myvars)

#create the p, x, s, o, LotNrRel an ClosingCount variables
for lot in Lots:
    my_p_var = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"p_{lot}")
    my_x_var = lpmodel.addVar(vtype = grb.GRB.binary, name = f"x_{lot}")
    my_s_var = lpmodel.addVar(vtype = grb.GRB.continuous, lb=0, name = f"s_{lot}")
    my_o_var = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"o_{lot}")
    my_LotNrRel_var = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"LotNrRel_{lot}")
    my_ClosingCount_var = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"ClosingCount_{lot}")
    my_ClosingCountCat_var = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"ClosingCountCat_{lot}")
    my_ClosingCountRel_var = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"ClosingCountRel_{lot}")
    Lots[lot].set_p_var(my_p_var)
    Lots[lot].set_x_var(my_x_var)
    Lots[lot].set_s_var(my_s_var)
    Lots[lot].set_o_var(my_o_var)
    Lots[lot].set_LotNrRel_var(my_LotNrRel_var)
    Lots[lot].set_ClosingCount_var(my_ClosingCount_var)
    Lots[lot].set_ClosingCountCat_var(my_ClosingCountCat_var)
    Lots[lot].set_ClosingCountRel_var(my_ClosingCountRel_var)

#Create y variables
for lot in Lots:
    my_y_vars = {}
    for i in range(1,N+1):
        my_y_var = lpmodel.addVar(vtype = grb.GRB.binary, name = f"y_{i},{lot}")
        my_y_vars[1] = my_y_var
    Lots[lot].set_y_vars(my_y_vars)

#Create q variables
for lot in Lots:
    my_q_vars = {}
    for tau in range(tau_min, tau_max +1):
        my_q_var = lpmodel.addVar(vtype = grb.GRB.binary, name = f"y_{tau},{lot}")
        my_q_vars[tau] = my_q_var
    Lots[lot].set_q_vars(my_q_vars)

#Create a variables
for lot in Lots:
    my_a_varss = {}
    for lot2 in Lots:
        my_a_vars = {}
        for tau in range(tau_min, tau_max+1):
            my_a_var = lpmodel.addVar(vtype= grb.GRB.binary, name = f"a_{lot},{lot2},{tau}")
            my_a_vars[tau] = my_a_var
        my_a_varss[lot2] = my_a_vars
    lot.set_a_vars(my_a_varss)

#Set the objective function (1)
lpmodel.setObjective(sum(lot.get_o_var for lot in Lots))

#Set constraint (2)
for lot in Lots:
    for node in Nodes_sold:
        lpmodel.addConstr(Lots[lot].features[Nodes_sold[node].feature] + (max(l.features[Nodes_sold[node].feature] for l in Lots.values())
                             - Nodes_sold[node].threshold) * sum(l.get_z_vars()[lot] for l in Nodes_sold[node].leaves_left_subtree) <= 
                             max(l.features[Nodes_sold[node].feature] for l in Lots.values()), 
                             name = f"Constraint (2) for lot {lot} and node {node} in the classification model")
for lot in Lots:
    for node in Nodes_price:
        lpmodel.addConstr(Lots[lot].features[Nodes_price[node].feature] + (max(l.features[Nodes_price[node].feature] for l in Lots.values())
                             - Nodes_price[node].threshold) * sum(l.get_z_vars()[lot] for l in Nodes_price[node].leaves_left_subtree) <= 
                             max(l.features[Nodes_price[node].feature] for l in Lots.values()), 
                             name = f"Constraint (2) for lot {lot} and node {node} in the regression model")                             

#Set constraint (3)
for lot in Lots:
    for node in Nodes_sold:
        lpmodel.addConstr(Lots[lot].features[Nodes_sold[node].feature] + (min(l.features[Nodes_sold[node].feature] for l in Lots.values())
                             - Nodes_sold[node].threshold) * sum(l.get_z_vars()[lot] for l in Nodes_sold[node].leaves_right_subtree) >= 
                             min(l.features[Nodes_sold[node].feature] for l in Lots.values()), 
                             name = f"Constraint (3) for lot {lot} and node {node} in the classification model")
for lot in Lots:
    for node in Nodes_price:
        lpmodel.addConstr(Lots[lot].features[Nodes_price[node].feature] + (min(l.features[Nodes_price[node].feature] for l in Lots.values())
                             - Nodes_price[node].threshold) * sum(l.get_z_vars()[lot] for l in Nodes_price[node].leaves_right_subtree) >= 
                             min(l.features[Nodes_price[node].feature] for l in Lots.values()), 
                             name = f"Constraint (3) for lot {lot} and node {node} in the regression model")

#Set constraint (4)
for lot in Lots:
    lpmodel.addConstr(sum(l.get_z_vars()[lot] for l in Leafnodes_sold.values()) == 1,
                         name = f"Constraint (4) for lot {lot} and the classification model")
    lpmodel.addConstr(sum(l.get_z_vars()[lot] for l in Leafnodes_price.values()) == 1,
                         name = f"Constraint (4) for lot {lot} and the regression model")

#Set constraint (5)
for lot in Lots:
    lpmodel.addConstr(Lots[lot].get_p_var() == sum(l.leaf_value * l.get_z_vars()[lot] for l in Leafnodes_price.values()),
                         name = f"Constraint (5) for lot {lot}")

#Set contraint (6)
for lot in Lots:
    lpmodel.addConstr(Lots[lot].get_x_var() == sum(l.leaf_value * l.get_z_vars()[lot] for l in Leafnodes_sold.values()),
                         name = f"Constraint (6) for lot {lot}")

#Set constraint (7)
for i in range(1,N+1):
    lpmodel.addConstr(sum(lot.get_y_vars()[i] for lot in Lots.values()) == 1,
                         name = f"Constraint (7) for location {i}")

#Set constraint (8)
for lot in Lots:
    lpmodel.addConstr(sum(Lots[lot].get_y_vars()[i] for i in range(1, N+1)) == 1,
                         name = f"Constraint (8) for lot {lot}")

#Set constraint (9)
for lot in Lots:
    lpmodel.addConstr(sum(i*Lots[lot].get_y_vars()[i] for i in range(1, N+1))/N == Lots[lot].get_LotNrRel_var(),
                         name = f"Constraint (9) for lot {lot}")

#Set constraint (10)
for lot in Lots:
    lpmodel.addConstr(sum(thetadict[tau]*Lots[lot].get_q_vars()[tau] + 
                            Lots[lot].get_q_vars()[tau]*sum(r.get_q_vars()[tau] for r in Lots.values()) for tau in range(tau_min,tau_max+1)
                            == Lots[lot].get_ClosingCount()),
                          name = f"Constraint (10) for lot {lot}")

#Set constraint (11)
for lot in Lots:
    lpmodel.addConstr(sum(sum(bigthetadict[tau][c] * Lots[lot].kappas[c] * Lots[lot].get_q_vars()[tau] +
                                 Lots[lot].kappas[c] * Lots[lot].get_q_vars()[tau] * sum(r.get_q_vars()[tau]*r.kappas[c] for r in Lots.values())
                                 for c in C) for tau in range(tau_min, tau_max+1)) == Lots[lot].get_ClosingCountCat_var(),
                                 name = f"Constraint (11) for lot {lot}")

#Set constraint (12)
for lot in Lots:
    lpmodel.addConstr(sum(sum(bigthetadict[tau][sigma] * Lots[lot].ks[sigma] * Lots[lot].get_q_vars()[tau] +
                                 Lots[lot].ks[sigma] * Lots[lot].get_q_vars()[tau] * sum(r.get_q_vars()[tau]*r.ks[sigma] for r in Lots.values())
                                 for sigma in S) for tau in range(tau_min, tau_max+1)) == Lots[lot].get_ClosingCountSub_var(),
                                 name = f"Constraint (12) for lot {lot}")

lpmodel.update()
lpmodel.write("1BM130.lp")
lpmodel.optimize()