import numpy as np
import random
import copy
import math

class Node:
    """
    A Node class for our DecisionTree.

    @attrs:
        left: The left child of the node, a node
        right: The right child of the node, a node
        depth: The depth of the node within the tree, a int
        index_split_on: The index that will be splited on of current node,
        split_value: The split value of the split index, a float 
        isleaf: True if the Node is a leaf, otherwise False, a boolean
        label: The label of the node,  
        info: A dictionary that can be used for saving useful information for debugging
    """
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, split_value=0.0, isleaf=False, label='', info=None):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.split_value = split_value
        self.isleaf = isleaf
        self.label = label
        self.info ={} if info is None else info

def empirical_distribution(dataset):
    """
    Computes the empirical distribution of a dataset with n classes.
    p[i] = P(y = label_i). Mathematically, p[i] = (# of y=label_i)/(# total examples)

    :param dataset: A non-empty dataset as a Python list of lists.
    :return: An array indicating the probability of each class.
    """
    labels = [data[0] for data in dataset]
    unique_labels = list(set(labels))
    p = [0.0] * len(unique_labels)

    for i, label in enumerate(unique_labels):
        p[i] = float(labels.count(label))/len(dataset)
    return p

def entropy(dataset):
    """
    Calculates the entropy of the dataset
    This function is used to calculate the entropy of a dataset with multi-classes.
    entropy = sum(-(p_i)*log(p_i)) i=1,2,...,n, p_i is the empirical distribution of i-th class

    :param dataset: A non-empty dataset as a Python list of lists.
    :return: The entropy of the dataset
    """
    # TODO: Compute the entropy of the given dataset. This should be different
    # from your function in hw1 since we are now generalizing to multi-class classification.
    pass

def gini_index(dataset):
    """
    Calculate the gini index of the dataset
    gini_index = sum((p_i) * (p_j)), i,j=1,2,...,n, i!=j

    :param dataset: A non-empty dataset as a Python list of lists.
    :return: The gini_index of the dataset
    """
    # TODO: Compute the gini index of the given dataset. This should be different
    # from your function in hw1 since we are now generalizing to multi-class classification.
    pass

def calc_info_gain(data, split_index, split_value, gain_function):
    """
    Calculate the infomation gain of the proposed splitting
    gain = C(data) - (P[x_i>=value] * C(sub_dataset with x_i>=value) + P[x_i<value] * C(sub_dataset with x_i<value))
    C is the gain function

    :param data: A non-empty dataset as a Python list of lists.
    :param split_index: The index to split the dataset
    :param split_value: The value of the index to split the dataset, a float
    :param gain_function: The function that will be used to evaluate the chaos of a dataset
    :return: The information gain of the proposed splitting
    """
    left_data = []
    right_data = []
    for row in data:
        if row[split_index] < split_value:
            left_data.append(row)
        else:
            right_data.append(row)

    total_gain = gain_function(data)
    left_gain = 0.0
    if len(left_data) != 0:
        left_gain = gain_function(left_data)

    right_gain = 0.0
    if len(right_data) != 0:
        right_gain = gain_function(right_data)

    return  total_gain-(float(len(left_data))/len(data) * left_gain + float(len(right_data))/len(data) * right_gain)

def is_terminal(node, data, max_depth):
    """
    Check whether this should be a terminal node (ie: a leaf node)

    :param node: The Node
    :param data: The data at the Node, as a Python list of lists
    :param max_depth: The maximum depth allowed for any Node in the tree, a int
    :return:
        - A boolean, True indicating the current node should be a leaf.
        - A label, indicating the label of the Node.

    Stop the recursion when:
        1. The dataset is empty.
        2. All the instances in this dataset belong to the same class
        3. The depth of the nodex exceede the maximum depth.
    """

    if len(data) == 0:
        return True, None

    count_label = {}
    max_count = 0
    max_label = None
    labels = [row[0] for row in data]

    #get the most appear label
    for item in labels:
        if item in count_label:
            count_label[item]+=1
        else:
            count_label[item] = 1
        if count_label[item] > max_count:
            max_label = item

    if len(count_label) == 1 or node.depth == max_depth:
        return True, max_label
    else:
        return False, max_label

def split_recurs(node, data, indices,  gain_function, max_depth):
    """
    Recursively split the node to greedily construct a DecisionTree.

    :param node: A Node that will either be split or become a leaf.
    :param data: The data at this Node as a Python list of lists
    :param indices: The indices(attributes) that can be used to split, a Python list 
    :param gain_function: The function that will be used to evaluate the gain of a split.
    :param max_depth: The maximum depth allowed for any Node in the tree, a int
    :return: None
    """
    done, label = is_terminal(node, data, max_depth)
    node.label = label
    node.info['data_size'] = len(data)

    if done:
        node.isleaf = True
        return

    split_index = None
    split_value = None

    #  TODO: Go through all the attributes that you can still split on
    #  and for each attribute, try all possible values that you can split on to
    #  find the split that maximizes the gain_function.

    #  If you have correctly implemented the TODO above, split_index
    #  and split_value should not be None at this point.
    if split_index is None or split_value is None:
        print("ERROR: split_index or split_value was not set.")
        exit(1)

    node.index_split_on = split_index
    node.split_value = split_value

    left_data, right_data = [], []
    for row in data:
        if row[split_index] < split_value:
            left_data.append(row)
        else:
            right_data.append(row)

    node.left = Node(depth=node.depth + 1)
    node.right = Node(depth=node.depth + 1)

    split_recurs(node.left, left_data, indices, gain_function, max_depth)
    split_recurs(node.right, right_data, indices, gain_function, max_depth)

def predict_recurs(node, row):
    """
    Recursively visit the tree and predict the label of given data

    :param node: A Node of the decision tree
    :param row: The data that need to be predicted, a Python list
    :return: The predicted label of the data
    """
    # Check if a terminal node is met
    if node.isleaf == False:
        split_index = node.index_split_on

        if row[split_index] < node.split_value:
            return predict_recurs(node.left, row)
        else:
            return predict_recurs(node.right, row)
    else:
        # Return terminal node's label as a prediction
        return node.label


def print_tree(root):
    """
    Helper methods for tree visualization.
    You DON'T need to touch these
    """
    temp = []
    output = []
    print('---START PRINT TREE---')
    def print_subtree(node, indent=''):
        if node is None:
            return str("None")
        if node.isleaf:
            return "label is %s, sample size is %d" % (node.label, node.info.get('data_size', 'No data_size set'))
        else:
            decision = 'split attribute = %d;  sample size = %d' % (node.index_split_on,  node.info.get('data_size', 'No data_size set'))
        left = indent + 'T -> '+ print_subtree(node.left, indent + '\t\t')
        right = indent + 'F -> '+ print_subtree(node.right, indent + '\t\t')
        return (decision + '\n' + left + '\n' + right)

    print(print_subtree(root))
    print('----END PRINT TREE---')
