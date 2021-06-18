"""
Module containing classes that are used in lp.py to create the linear program
"""
from numpy import argmax
class Lot:
    def __init__(self, lot_id, features:dict):
        self.id = lot_id
        self.features = features
        self.kappas = {key: value for key, value in self.features.items()
                       if ("lot.category_" in key)
                       and ("lot.category_count" not in key)}
        self.ks = {key: value for key, value in self.features.items()
                   if ("lot.subcategory_" in key)
                   and ("lot.subcategory_count" not in key)}
        
    def set_feature(self, feature, value):
        self.features[feature] = value
    
    def get_feature(self, feature):
        return self.features[feature]
        
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
        self.features["lot.start_amount"] = s_var

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
        self.features["lot.rel_nr"] = LotNrRel_var
        #self.features[featurenumber of LotNrRel] = LotNrRel_var

    def get_LotNrRel_var(self):
        return self.LotNrRel_var
        #return self.features[featurenumber of LotNrRel]

    def set_ClosingCount_var(self, ClosingCount_var):
        self.ClosingCount_var = ClosingCount_var
        self.features['lot.num_closing_timeslot'] = ClosingCount_var
        
    def get_ClosingCount_var(self):
        return self.ClosingCount_var

    def set_ClosingCountCat_var(self, ClosingCountCat_var):
        self.ClosingCountCat_var = ClosingCountCat_var
        self.features["lot.num_closing_timeslot_category"] = ClosingCountCat_var

    def get_ClosingCountCat_var(self):
        return self.ClosingCountCat_var

    def set_ClosingCountSub_var(self, ClosingCountSub_var):
        self.ClosingCountSub_var = ClosingCountSub_var
        self.features["lot.num_closing_timeslot_subcategory"] = ClosingCountSub_var

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
        value_this_node = value[self.id]
        # If value from price tree, value is the single value that is stored
        if value_this_node.shape == (1, 1):
            self.leaf_value = value_this_node[0,0]
        # If value from is_sold tree, value is value with the most items in it
        elif value_this_node.shape == (1,2):
            self.leaf_value = argmax(value_this_node)
        self.z_vars = {}
    
    def set_z_vars(self, z_vars):
        self.z_vars = z_vars

    def get_z_vars(self):
        return self.z_vars

    def get_leaf_value(self):
        return self.leaf_value
