import numpy as np
import random
import copy
import math


def train_error(dataset):
    '''
        TODO:
        Calculate the train error of the subdataset and return it.
        For a dataset with two classes:
        C(p) = min{p, 1-p}
    '''
    p = sum([row[0] for row in dataset]) / len(dataset)
    return min([p, 1 - p])


def entropy(dataset):
    '''
        TODO:
        Calculate the entropy of the subdataset and return it.
        This function is used to calculate the entropy for a dataset with 2 classes.
        Mathematically, this function return:
        C(p) = -p*log(p) - (1-p)log(1-p)
    '''
    p = sum([row[0] for row in dataset]) / len(dataset)
    if p == 1:
        return -p * math.log(p)
    elif p == 0:
        return - (1 - p) * math.log(1 - p)
    else:
        return -p * math.log(p) - (1 - p) * math.log(1 - p)


def gini_index(dataset):
    '''
        TODO:
        Calculate the gini index of the subdataset and return it.
        For dataset with 2 classes:
        C(p) = 2*p*(1-p)
    '''
    p = sum([row[0] for row in dataset]) / len(dataset)
    return 2 * p * (1 - p)


class Node:
    '''
    Helper to construct the tree structure.
    '''

    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1, info={}):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = info


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)

    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)

    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt / len(data)

    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)

    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        '''
        loss_validation = self.loss(validation_data)
        if not node.isleaf and not node.left.isleaf and not node.right.isleaf:
            data_left = np.array(validation_data)[np.array(validation_data)[:, node.index_split_on] is True, :].tolist()
            data_right = np.array(validation_data)[np.array(validation_data)[:, node.index_split_on] is False, :].tolist()
            if self.loss(data_left) + self.loss(data_right) > loss_validation:
                node.isleaf, node.label = True, -1
                node.right, node.left = None, None
            self._predict_recurs(node.left, data_left)
            self._predict_recurs(node.right, data_right)

    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the nodex exceede the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf.
            - A label, indicating the label of the leaf (-1 if False)
        '''
        if len(data) == 0 or len(indices) == 0 or node.depth > self.max_depth:
            return True, -1
        else:
            return False, node.label

    def _split_recurs(self, node, rows, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.
        First use _is_terminal() to check if the node needs to be splitted.
        Then select the column that has the maximum infomation gain to split on.
        Also store the label predicted for this node.
        Then split the data based on whether satisfying the selected column.
        The node should not store data, but the data is recursively passed to the children.
        '''
        if len(indices) is not 0 and len(rows) is not 0:
            gain_cols = []
            for _, v in enumerate(indices):
                gain_cols.append(self._calc_gain(rows, v, self.gain_function))
            node.index_split_on = indices[np.argmax(gain_cols)]
            data_left = [row for row in rows if row[node.index_split_on] is True]
            data_right = [row for row in rows if row[node.index_split_on] is False]
            if len(data_left) != 0:
                node.left = Node(depth=node.depth + 1, label=1)
            if len(data_right) != 0:
                node.right = Node(depth=node.depth + 1, label=0)
            node.label = self.predict([row[node.index_split_on] for row in rows])
            node.info['cost'] = self.loss(rows)
            node.info['data_size'] = len(rows)

            indices.remove(node.index_split_on)
            node.left.isleaf, node.left.label = self._is_terminal(node.left, data_left, indices)
            node.right.isleaf, node.right.label = self._is_terminal(node.right, data_right, indices)
            if not node.left and not node.left.isleaf:
                self._split_recurs(node.left, data_left, indices)
            if not node.right and not node.right.isleaf:
                self._split_recurs(node.right, data_right, indices)
            

    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - (P[x_i=True] * C(P[y=1|x_i=True]) * P[x_i=False]C(P[y=1|x_i=False)])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        x_i = [row[split_index] for row in data]
        data_true = [row for i, row in enumerate(data) if x_i[i] is True]
        data_false = [row for i, row in enumerate(data) if x_i[i] is False]
        p_true = sum([x is True for x in x_i]) / len(x_i)
        try:
            gain = gain_function(data) - (p_true * gain_function(data_true) + (1 - p_true) * gain_function(data_false))
        except ZeroDivisionError:
            if len(data_true) == 0:
                gain = gain_function(data) - ((1 - p_true) * gain_function(data_false))
            elif len(data_false) == 0:
                gain = gain_function(data) - (p_true * gain_function(data_true))
        finally:
            gain = gain_function(data)
        return gain

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')

        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = %d; cost = %f; sample size = %d' % (node.index_split_on, node.info['cost'], node.info['data_size'])
            left = indent + 'T -> ' + print_subtree(node.left, indent + '\t\t')
            right = indent + 'F -> ' + print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')

    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)

        return 1 - np.array(loss_vec) / len(data)

    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left is not None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right is not None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
