#%%
import gurobipy as grb
from sklearn.tree  import DecisionTreeClassifier, DecisionTreeRegressor
#%%
clf_price = DecisionTreeRegressor()

children_left_price = clf_price.tree_.children_left
children_right_price = clf_price.tree_.children_right
feature_price = clf_price.tree_.feature
threshold_price = clf_price.tree_.threshold
value_price = clf_price.tree_.value

clf_sold = DecisionTreeClassifier()

children_left_sold = clf_sold.tree_.children_left
children_right_sold = clf_sold.tree_.children_right
feature_sold = clf_sold.tree_.feature
threshold_sold = clf_sold.tree_.threshold
value_sold = clf_sold.tree_.value
#%%
class Lot:
    def __init__(self, lot_id, *features):
        self.id = lot_id
        self.features = features
        
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

    def set_s_var(self, s_var):
        self.s_var = s_var

    def get_s_var(self):
        return self.s_var

class Node:
    def __init__(self, id, children_left, children_right, feature, threshold):
        self.id = id
        self.feature = feature[self.id]
        self.leftchild = children_left[self.id]
        self.rightchild = children_right[self.id]
        self.threshold = threshold[self.id]
        #create the list of leaf nodes that are part of the left and right subtree
        leftsubtreenodes = [Nodes_price[self.leftchild]]
        leftsubtreeleaves = []
        for i in leftsubtreenodes:
            if i.leftchild == i.rightchild:
                leftsubtreeleaves.append(Leafnodes_price[i])
            else:
                leftsubtreenodes.append(Nodes_price[i.leftchild])
                leftsubtreenodes.append(Nodes_price[i.rightchild])
        self.leaves_left_subtree = leftsubtreeleaves
        rightsubtreenodes = [Nodes_price[self.right_child]]
        rightsubtreeleaves = []
        for i in rightsubtreenodes:
            if i.leftchild == i.rightchild:
                rightsubtreeleaves.append(Leafnodes_price[i])
            else:
                rightsubtreenodes.append(Nodes_price[i.leftchild])
                rightsubtreenodes.append(Nodes_price[i.rightchild])
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


Nodes_price = {}
Leafnodes_price = {}
for node in range(len(value_price)):
    if children_left_price[node] != children_right_price[node]:
        Nodes_price[node] = Node(node, children_left_price, children_right_price, feature_price, threshold_price)
    else:
        Leafnodes_price[node] = Leafnode(node, value_price)

Nodes_sold = {}
Leafnodes_sold = {}
for node in range(len(value_sold)):
    if children_left_sold[node] != children_right_sold[node]:
        Nodes_sold[node] = Node(node, children_left_sold, children_right_sold, feature_sold, threshold_sold)
    else:
        Leafnodes_sold[node] = Leafnode(node, value_sold)


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

#create the z^s_l_r variables
for leafnode in Leafnodes_sold:
    myvars = {}
    for lot in Lots:
        myvar = lpmodel.addVar(vtype = grb.GRB.binary, name = f"z^s_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes_sold[leafnode].set_z_vars(myvars)

#create the p and s variables
for lot in Lots:
    my_p_var = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"p_{lot}")
    my_s_var = lpmodel.addVar(vtype = grb.GRB.binary, name = f"s_{lot}")
    Lots[lot].set_p_var(my_p_var)
    Lots[lot].set_s_var(my_s_var)

lpmodel.setObjective(sum(lot.get_p_var * lot.get_s_var for lot in Lots))

lpmodel.update()
lpmodel.write("1BM130.lp")