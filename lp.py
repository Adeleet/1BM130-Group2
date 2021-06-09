#%%
import gurobipy as grb
from sklearn.tree  import DecisionTreeClassifier
#%%
clf = DecisionTreeClassifier()

children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold
value = clf.tree_.value
#%%
class Lot:
    def __init__(self, lot_id, closing_timeslot):
        self.id = lot_id
        self.closing_timeslot = closing_timeslot

    def get_id(self):
        return self.id

    def set_p_var(self, p_var):
        self.p_var = p_var

    def get_p_var(self, p_var):
        return self.p_var

class Node:
    def __init__(self, id, children_left, children_right, feature, threshold):
        self.id = id
        self.feature = feature[self.id]
        self.leftchild = children_left[self.id]
        self.rightchild = children_right[self.id]
        self.threshold = threshold[self.id]

        leftsubtreenodes = [Nodes[self.leftchild]]
        leftsubtreeleaves = []
        for i in leftsubtreenodes:
            if i.leftchild == i.rightchild:
                leftsubtreeleaves.append(Leafnodes[i])
            else:
                leftsubtreenodes.append(Nodes[i.leftchild])
                leftsubtreenodes.append(Nodes[i.rightchild])
        self.leaves_left_subtree = leftsubtreeleaves
        rightsubtreenodes = [Nodes[self.right_child]]
        rightsubtreeleaves = []
        for i in rightsubtreenodes:
            if i.leftchild == i.rightchild:
                rightsubtreeleaves.append(Leafnodes[i])
            else:
                rightsubtreenodes.append(Nodes[i.leftchild])
                rightsubtreenodes.append(Nodes[i.rightchild])
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

Nodes = {}
Leafnodes = {}
for node in range(len(value)):
    if children_left[node] != children_right[node]:
        Nodes[node] = Node(node, children_left, children_right, feature, threshold)
    else:
        Leafnodes[node] = Leafnode(node, value)


#%%
lpmodel = grb.Model("1BM130 prescriptive analytics")
for leafnode in Leafnodes:
    myvars = {}
    for lot in Lots:
        myvar = lpmodel.addVar(vtype = grb.GRB.binary, name = f"z_{leafnode},{lot}")
        myvars[lot] = myvar
    Leafnodes[leafnode].set_z_vars(myvars)

for lot in Lots:
    myvar = lpmodel.addVar(vtype = grb.GRB.continuous, name = f"p_{lot}")
    Lots[lot].set_p_var(myvar)

