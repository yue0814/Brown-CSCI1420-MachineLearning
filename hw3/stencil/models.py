"""
    This is the class file you will have to fill in.

    You will have to define three classifiers: Nearest Neighbors, K-means and Decision Tree using helper functions
    defined in kneighbors.py, kmeans.py and dtree.py files.

"""

from kmeans import assign_step, kmeans
from kneighbors import get_neighbors_indices, get_response
from dtree import *

class KNeighborsClassifier(object):
    """
    Classifier implementing the k-nearest neighbors vote.

    @attrs:
        k: The number of neighbors to use, an int
        train_inputs: inputs of training data used to train the model, a 2D Python list
        train_labels: labels of training data used to train the model, a Python list
        n_labels_ : number of labels in the training data, an int,
                    this attribute is used in plot_KNN() to produce classification plot
    """
    def __init__(self, k):
        """
        Initiate K Nearest Neighbors Classifier with some parameters

        :param n_neighbors: number of neighhbors to use, an int
        """
        self.k = k
        self.train_inputs = None
        self.train_labels = None
        self.n_labels_ = None

    def train(self, X, y):
        """
        train the data (X and y) to model, calculate the number of unique labels and store it in self.n_labels_

        :param X: inputs of data, a 2D Python list
        :param y: labels of data, a Python list
        :return: None
        """
        self.train_inputs = X
        self.train_labels = y
        self.n_labels_ = len(np.unique(y))

    def predict(self, X):
        """
        Compute predictions of input X

        :param X: inputs of data, a 2D Python list
        :return: a Numpy array of predictions
        """

        # For each data point in X:
        # 1. Compute the k nearest neighbors (indices)
        # 2. Compute the highest label response given the k nearest neighbors
        # Use the helper methods!

        # TODO
        pass

    def accuracy(self, data):
        """
        Compute accuracy of the model when applied to data

        :param data: a namedtuple including inputs and labels
        :return: a float number indicating accuracy
        """
        # TODO: Compute the portion of data with correctly predicted labels
        pass

class KmeansClassifier(object):
    """
    K-means Classifier via Iterative Improvement

    @attrs:
        k: The number of clusters to form as well as the number of centroids to generate (default = 3), an int
        tol: Relative tolerance with regards to inertia to declare convergence, a float number,
                the default value is set to 0.0001
        max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int,
                  the default value is set to 500 iterations
        cluster_centers_: a Python dictionary where each (key, value) pair is
                            (a class label[an int], k cluster centers of that class label[a Numpy array])

    K-means is not a classification algorithm, it is an unsupervised learning algorithm. You will be creating K
    cluster centers for EACH label (k * #labels total). The label of the closest center is then used to classify data.

    """

    def __init__(self, n_clusters = 3, max_iter = 500, threshold = 1e-4):
        """
        Initiate K-means with some parameters
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers_ = dict()

    def train(self, X, y):
        """
        Compute K-means clustering over data with each class label and store your result in self.cluster_centers_
        You should use kmeans helper function from kmeans.py

        :param X: inputs of training data, a 2D Python list
        :param y: labels of training data, a Python list
        :return: None
        """
        # TODO
        pass

    def predict(self, X):
        """
        Predict the label of each sample in X, which is the label of the closest cluster center each sample in X belongs to
        Be sure to identify the closest center out of all (k * #labels) centers

        :param X: inputs of data, a 2D Python list
        :return: a Python list of labels predicted by model
        """
        # TODO
        pass

    def accuracy(self, data):
        """
        Compute accuracy of the model when applied to data

        :param data: a namedtuple including inputs and labels
        :return: a float number indicating accuracy
        """
        # TODO: Compute the portion of data with correctly predicted labels
        pass

class DecisionTree:
    """
    A DecisionTree with ranges. Can handle (multi-class) classification tasks
    with continious inputs.

    @attrs:
        data: The data that will be used to construct the decision tree, as a python list of lists.
        gain_function: The gain_function specified by the user.
        max_depth: The maximum depth of the tree, a int.
    """

    def __init__(self, data, validation_data=None,  gain_function='entropy', max_depth=40):
        """
        Initiate the deccision tree with some parameters
        """
        self.max_depth = max_depth
        self.root = Node()

        if gain_function=='entropy':
            self.gain_function = entropy
        elif gain_function=='gini_index':
            self.gain_function = gini_index
        else:
            print("ERROR: GAIN FUNCTION NOT IMPLEMENTED")

        indices = list(range(1, len(data[0])))
        split_recurs(self.root, data, indices, self.gain_function, self.max_depth)

        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def predict(self, X):
        """
        Predicts the label of each sample in X

        :param X: A dataset as a python list of lists
        :return: A list of labels predicted by the trained decision tree
        """
        labels = [predict_recurs(self.root, data) for data in X]
        return labels

    def accuracy(self, data):
        """
        Computes accuracy of the model when applied to data

        :param data: dataset with the first column as the label, a python list of lists.
        :return: A float indicating accuracy (between 0 and 1)
        """
        cnt = 0.0
        test_Y = [row[0] for row in data]
        pred =  self.predict(data)
        for i in range(0, len(test_Y)):
            if test_Y[i] == pred[i]:
                cnt+=1
        return float(cnt/len(data))

    def print_tree(self):
        """
        Visualize the decision tree
        """
        print_tree(self.root)
