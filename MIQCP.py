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

    def set_x_var(self, x_var):
        self.x_var = x_var

    def get_x_var(self):
        return self.x_var

    def set_s_var(self, s_var):
        self.s_var = s_var

    def get_s_var(self):
        return self.s_var

    def set_c_var(self, c_var):
        self.c_var = c_var 

    def get_c_var(self):
        return self.c_var

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

    def get_LotNrRel_var(self):
        return self.LotNrRel_var

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
miqcpmodel = grb.Model("1BM130 prescriptive analytics")
miqcpmodel.modelSense = grb.GRB.MAXIMIZE

#create the z^p_l_r variables
for leafnode in Leafnodes_price:
    myvars = {}
    for lot in Lots:
        myvar = miqcpmodel.addVar(vtype = grb.GRB.binary, name = f"z^p_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes_price[leafnode].set_z_vars(myvars)

#create the z^x_l_r variables
for leafnode in Leafnodes_sold:
    myvars = {}
    for lot in Lots:
        myvar = miqcpmodel.addVar(vtype = grb.GRB.binary, name = f"z^x_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes_sold[leafnode].set_z_vars(myvars)

#create the p and x variables
for lot in Lots:
    my_p_var = miqcpmodel.addVar(vtype = grb.GRB.continuous, name = f"p_{lot}")
    my_s_var = miqcpmodel.addVar(vtype = grb.GRB.binary, name = f"x_{lot}")
    Lots[lot].set_p_var(my_p_var)
    Lots[lot].set_x_var(my_s_var)

miqcpmodel.setObjective(sum(lot.get_p_var * lot.get_s_var for lot in Lots))

miqcpmodel.update()
miqcpmodel.write("1BM130.lp")