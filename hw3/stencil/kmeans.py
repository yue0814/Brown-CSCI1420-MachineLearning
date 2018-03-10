"""
    This is a file you will have to fill in.

    It contains helper functions required by K-means method via iterative improvement

"""

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids

    :param k: number of cluster centroids, an int
    :param inputs: a 2D Python list, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    pass


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance

    :param inputs: inputs of data, a 2D Python list
    :param centroids: a Numpy array of k current centroids
    :return: a Python list of centroid indices, one for each row of the inputs
    """
    # TODO
    pass


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster - the average of all data points in the cluster

    :param inputs: inputs of data, a 2D Python list
    :param indices: a Python list of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    pass


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    Use init_centroids, assign_step, and update_step!
    The only computation that should occur within this function is checking 
    for convergence - everything else should be handled by helpers

    :param inputs: inputs of data, a 2D Python list
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    pass
