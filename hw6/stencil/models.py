import numpy as np


class EMNaiveBayesClassifier:

    def __init__(self, num_hidden):
        '''
        @attrs:
            num_hidden  The number of hidden states per class. (An integer)
            priors The estimated prior distribution over the classes, P(Y) (A Numpy Array)
            parameters The estimated parameters of the model. A Python dictionary from class to the parameters
                conditional on that class. More specifically, the dictionary at parameters[Y] should store
                - bjy: b^jy = P(h^j | Y) for each j = 1 ... k
                - bij: b^ij = P(x_i | h^j, Y)) for each i, for each j = 1 ... k
        '''
        self.num_hidden = num_hidden
        self.priors = None
        self.parameters = None
        pass

    def train(self, X, Y, max_iters=10, eps=1e-4):
        '''
            Trains the model using X, Y. More specifically, it learns the parameters of
            the model. It must learn:
                - b^y = P(y) (via MLE)
                - b^jy = P(h^j | Y)  (via EM algorithm)
                - b^ij = P(x_i | h^j, Y) (via EM algorithm)

            Before running the EM algorithm, you should partition the dataset based on the labels Y. Then
            run the EM algorithm on each of the subsets of data.

            :param X 2D Numpy array where each row, X[i, :] contains a sample. Each sample is a binary feature vector.
            :param Y 1D Numpy array where each element Y[i] is the corresponding label for X[i, :]
            :max_iters The maxmium number of iterations to run the EM-algorithm. One
                iteration consists of both the E-step and the M-step.
            :eps Used for convergence test. If the maximum change in any of the parameters is
                eps, then we should stop running the algorithm.
            :return None
        '''
        # TODO

        pass

    def _em_algorithm(self, X, num_hidden, max_iters, eps):
        '''
            EM Algorithm to learn parameters of a Naive Bayes model.

            :param X A 2D Numpy array containing the inputs.
            :max_iters The maxmium number of iterations to run the EM-algorithm. One
                iteration consists of both the E-step and the M-step.
            :eps Used for convergence test. If the maximum change in any of the parameters is
                eps, then we should stop running the algorithm.
            :return the learned parameters as a tuple (b^ij,b^jy)
        '''
        # TODO

        pass

    def _e_step(self, X, num_hidden, bjy, bij):
        '''
            The E-step of the EM algorithm. Returns Q(t+1) = P(h^j | x, y, theta)
            See the handout for details.

            :param X The inputs to the EM-algorthm. A 2D Numpy array.
            :param num_hidden The number of latent states per class (k in the handout)
            :param bjy at the current iteration (b^jy = P(h^j | y))
            :param bij at the current iteration (b^ij = P(x_i | h^j, y))
            :return Q(t+1)
        '''
        # TODO

        pass

    def _m_step(self, X, num_hidden, probs):
        '''
            The M-step of the EM algorithm. Returns the next update to the parameters,
            theta(t+1).

            :param X The inputs to the EM-algorthm. A 2D Numpy array.
            :param num_hidden The number of latent states per class (k in the handout)
            :param probs (Q(t))
            :return theta(t+1) as a tuple (b^ij,b^jy)
        '''
        # TODO

        pass

    def predict(self, X):
        '''
        Returns predictions for the vectors X. For some input vector x,
        the classifier should output y such that y = argmax P(y | x),
        where P(y | x) is approximated using the learned parameters of the model.

        :param X 2D Numpy array. Each row contains an example.
        :return A 1D Numpy array where each element contains a prediction 0 or 1.
        '''
        # TODO

        pass

    def accuracy(self, X, Y):
        '''
            Computes the accuracy of classifier on some data (X, Y). The model
            should already be trained before calling accuracy.

            :param X 2D Numpy array where each row, X[i, :] contains a sample. Each sample is a binary feature vector.
            :param Y 1D Numpy array where each element Y[i] is the corresponding label for X[i, :]
            :return A float between 0-1 indicating the fraction of correct predictions.
        '''
        return np.mean(self.predict(X) == Y)
