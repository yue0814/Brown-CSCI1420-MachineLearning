#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains two classifiers: Naive Bayes and Logistic Regression

   Brown CS142, Spring 2018
"""
import random

import numpy as np


class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        # You are free to add more fields here.

    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        # TODO

    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        #TODO

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        #TODO


class LogisticRegression(object):
    """ Multinomial Linear Regression

    @attrs:
        weights: a parameter of the model
        alpha: the step size in gradient descent
        n_features: the number of features
        n_classes: the number of classes
    """
    def __init__(self, n_features, n_classes):
        """ Initializes a LogisticRegression classifer. """
        self.alpha = 0.2  # tune this parameter
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.zeros((n_features, n_classes))

    def train(self, data):
        """ Trains the model, using stochastic gradient descent

        @params:
            data: a namedtuple including training data and its information
        @return:
            None
        """
        #TODO

    def predict(self, inputs):
        """ Compute predictions based on the learned parameters

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        #TODO

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        #TODO

    def _softmax(self, x):
        """ apply softmax to an array

        @params:
            x: the original array
        @return:
            an array with softmax applied elementwise.
        """
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
