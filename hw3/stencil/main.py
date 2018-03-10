"""
    This file contains the main program to read data, run classifiers and print results.
    Please DO NOT modify any functions in this file other than the main function.

    To run the main.py file from command line, simply navigate to the directory where main.py resides, and type:
        python main.py PATH_TO_DATASET

    The two datasets are available in the /course/cs1420/data/hw3 folder, named digits.csv and fishiris.csv.
    Copy these two files to your hw3 folder, PATH_TO_DATASET should be the relative path to the csv files.

"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import namedtuple
import random
from sklearn.model_selection import train_test_split

from models import KmeansClassifier, KNeighborsClassifier, DecisionTree


def plot_KNN(data, N_NEIGHBORS=6, h=0.02):
    """
        This is a helper function that trains the Iris data to a KNN classifier and produce a 2D plot of the
        classification graph showing decision boundaries.

        The x and y axis of the graph is the first two features of inputs in order to produce a 2D plot.
        In the iris dataset, "Sepal length" and "Sepal width" are the first two features.

        The other features are discarded in both training and prediction.

    :param data: a namedtuple including inputs and labels, used in both training and ploting
    :param N_NEIGHBORS: number of neighbors to use for knn classifier
    :param h: step size of x and y axis in meshgrid
    :return: None
    """
    # Check if data has at least two features
    if len(data.inputs[0]) < 2:
        print("Number of features is less than 2!")
        return
    # Check if NUM_NEIGHBORS is a positive integer
    if isinstance(N_NEIGHBORS, int) == False:
        print("Invalid input! Number of neighbors must be an integer!")
        return
    elif N_NEIGHBORS <= 0:
        print("Invalid input! Number of neighbors must be greater than zero!")
        return

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])  # used to show decision boundaries
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])  # used to mark true labels of data points

    X = data.inputs[:, :2]
    y = data.labels

    # calculate min, max and limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    model = KNeighborsClassifier(N_NEIGHBORS)
    model.train(X, y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(str(model.n_labels_) + "-Class classification (k = %i) on Iris data" % (N_NEIGHBORS))
    plt.show()

def plot_Kmeans(data, NUM_CLUSTERS = 3):
    """
        This is a helper function that trains 8 x 8 handwritten digits data to a K-means classifier
        and visualize k cluster centers for each digit class (0 ~ 9).

        Note: this function is designed only for digits.csv data set.

    :param data:  a namedtuple including inputs and labels, used to train the K-means classifier
    :param NUM_CLUSTERS: number of clusters used in K-Means classifier
    :return: None
    """
    # Check if input data is valid
    if len(data.inputs[0]) != 64:
        print("Invalid input! Input data must be 8 x 8 hand-written digits!")
        return
    # Check if NUM_CLUSTERS is a positive integer
    if isinstance(NUM_CLUSTERS, int) == False:
        print("Invalid input! Number of clusters must be an integer!")
        return
    elif NUM_CLUSTERS <= 0:
        print("Invalid input! Number of clusters must be greater than zero!")
        return

    # Run K-means Classifier
    model = KmeansClassifier(NUM_CLUSTERS)
    model.train(data.inputs, data.labels)
    cluster_centers = model.cluster_centers_

    fig, ax = plt.subplots(NUM_CLUSTERS, len(cluster_centers.keys()), figsize=(8, 3))
    unflattened_centers = np.array(list(cluster_centers.values())).reshape(len(cluster_centers.keys()), NUM_CLUSTERS, 8,
                                                                           8)
    for i in range(len(ax[0])):
        for j in range(len(ax)):
            axi = ax[j][i]
            center = unflattened_centers[i, j, :, :]
            axi.set(xticks=[], yticks=[])
            axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    plt.show()


def test_KNN(train_data, test_data, N_NEIGHBORS = 6):
    """
        Create a K-Nearest Neighbor classifier, train it with train_data and print the accuracy of model on test_data

    :param train_data: a namedtuple including training inputs and training labels
    :param test_data: a namedtuple including test inputs and test labels
    :param N_NEIGHBORS: number of neighbors used in KNN classifier
    :return: None
    """
    # Check if NUM_NEIGHBORS is a positive integer
    if isinstance(N_NEIGHBORS, int) == False:
        print("Invalid input! Number of neighbors must be an integer!")
        return
    elif N_NEIGHBORS <= 0:
        print("Invalid input! Number of neighbors must be greater than zero!")
        return

    # Run K Nearest Neighbors Classifier
    model = KNeighborsClassifier(N_NEIGHBORS)
    model.train(train_data.inputs, train_data.labels)
    accuracy = model.accuracy(test_data)
    print("Testing on K Nearest Neighbor Classifier (K = " + str(N_NEIGHBORS) + "), the accuracy is {:.2f}%".format(accuracy * 100))

def test_Kmeans(train_data, test_data, NUM_CLUSTERS = 3):
    """
        Create a K-Means classifier, train it with train_data and print the accuracy of model on test_data

    :param train_data: a namedtuple including training inputs and training labels
    :param test_data: a namedtuple including test inputs and test labels
    :param NUM_CLUSTERS: number of clusters used in K-Means classifier
    :return: None
    """
    # Check if NUM_CLUSTERS is a positive integer
    if isinstance(NUM_CLUSTERS, int) == False:
        print("Invalid input! Number of clusters must be an integer!")
        return
    elif NUM_CLUSTERS <= 0:
        print("Invalid input! Number of clusters must be greater than zero!")
        return

    # Run K-means Classifier
    model = KmeansClassifier(NUM_CLUSTERS)
    model.train(train_data.inputs, train_data.labels)
    accuracy = model.accuracy(test_data)
    print("Testing on K-Means Classifier (K = " + str(NUM_CLUSTERS) + "), the accuracy is {:.2f}%".format(accuracy * 100))


def test_Dtree(data):
    """
        Create a Decision Tree classifier, using part of the data constructing the tree

    :param data: a panda DataFrame object
    :return: None
    """
    # In order to construct the tree, the data will be parsed to a form of 2-d array,
    # with data[:,0] as the label, and data[:,1:] as the values for each feature
    data = data.values.tolist()
    random.shuffle(data)
    ratio = 0.66
    num_train = int(np.ceil(len(data) * ratio))
    train_data = data[:num_train]
    test_data = data[num_train:]

    # Construct the decision tree
    decision_tree =  DecisionTree(train_data, gain_function='entropy')
    decision_tree.visualize_tree()
    print("\nExploring dataset with entropy...")
    print("Training size: ",len(train_data) )
    print("Test size: ",len(test_data) )
    print("Training data accuracy", decision_tree.accuracy(train_data))
    print("Test data accuracy", decision_tree.accuracy(test_data))

    decision_tree =  DecisionTree(train_data, gain_function='gini_index')
    #decision_tree.visualize_tree()
    print("\nExploring dataset with gini index...")
    print("Training size: ",len(train_data) )
    print("Test size: ",len(test_data) )
    print("Training data accuracy", decision_tree.accuracy(train_data))
    print("Test data accuracy", decision_tree.accuracy(test_data))

def main():

    random.seed(0)
    np.random.seed(0)
    if len(sys.argv) != 2:
        print('Incorrect number of argments. Usage: python main.py <path_to_dataset>')
        exit()

    script, filename = sys.argv

    Dataset = namedtuple('Dataset', ['inputs', 'labels'])

    # Read data
    data = pd.read_csv(filename, header = 0)

    # We assume labels are in the first column of the dataset
    labels = data.values[:, 0]

    # If labels are of type string, convert class names to numeric values
    if isinstance(labels[0], str):
        classes = np.unique(labels)
        class_mapping = dict(zip(classes, range(0, len(classes))))
        labels = np.vectorize(class_mapping.get)(labels)

    # Features columns are indexed from 1 to the end, make sure that dtype = float32
    inputs = data.values[:, 1:].astype("float32")

    # Split data into training set and test set with a ratio of 2:1
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size = 0.33)

    all_data = Dataset(inputs, labels)
    train_data = Dataset(train_inputs, train_labels)
    test_data = Dataset(test_inputs, test_labels)
    print("Shape of training data inputs: ", train_data.inputs.shape)
    print("Shape of test data inputs:", test_data.inputs.shape)

    # DO NOT MODIFY ABOVE THIS LINE!
    # TODO: call test_KNN(), test_Kmeans() and test_Dtree() to test your implementation
    # TODO: try out plot_KNN() on the iris data and plot_Kmeans() on the digits data

if __name__ == '__main__':
    main()
