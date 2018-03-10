#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run classifiers
   and print results to stdout.

   You do not need to change this file. You can add debugging code or code to
   help produce your report, but this code should not be run by default in
   your final submission.

   Brown CS142, Spring 2018
"""

from collections import namedtuple
import gzip
import numpy as np
from models import NaiveBayes, LogisticRegression


def main():
    Dataset = namedtuple('Dataset', ['inputs', 'labels'])

    # Reading in data. You do not need to touch this.
    with open("data/train-images-idx3-ubyte.gz", 'rb') as f1, open("data/train-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 60000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 60000)
        inputs = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(60000, 28 * 28)
        inputs = np.where(inputs > 99, 1, 0)
        labels = np.frombuffer(buf2, dtype='uint8', offset=8)
        data_train = Dataset(inputs, labels)
    with open("data/t10k-images-idx3-ubyte.gz", 'rb') as f1, open("data/t10k-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 10000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 10000)
        inputs = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(10000, 28 * 28)
        inputs = np.where(inputs > 99, 1, 0)
        labels = np.frombuffer(buf2, dtype='uint8', offset=8)
        data_test = Dataset(inputs, labels)

    # run naive bayes
    model = NaiveBayes(10)
    model.train(data_train)
    print("{:.1f}%".format(model.accuracy(data_test) * 100))

    # run logistic regression
    model = LogisticRegression(784, 10)
    model.train(data_train)
    print("{:.1f}%".format(model.accuracy(data_test) * 100))


if __name__ == "__main__":
    main()
