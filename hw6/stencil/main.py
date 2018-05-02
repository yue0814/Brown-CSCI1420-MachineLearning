
from collections import namedtuple
import gzip
import numpy as np
from models import EMNaiveBayesClassifier
import matplotlib.pyplot as plt


def generate_data(num_datapoints=10000):
    '''
        Generates data to test your EM-algorithm implementation. You should
        use the data generated to run EM with 2 hidden states.

        :return A tuple with inputs X, hidden states hj, and labels y. For this dataset,
        all the examples are labeled 0 since we are only using this to test the EM-algorithm.
    '''

    # Generative parameters (You can change these if you want)
    # Your EM-algorithm should get close to recovering these parameters.
    parameters = {0: {'bjy': np.array([0.5, 0.5]), 'bij': np.array([[0.9, 0.9], [0.1, 0.2]])}}
    hj = np.random.random(num_datapoints) <= parameters[0]['bjy'][0] # 0.5
    qy1_1 = np.random.random(num_datapoints) <= parameters[0]['bij'][0, 0] # 0.9
    qy1_2 = np.random.random(num_datapoints) <= parameters[0]['bij'][0, 1] # 0.9
    qy1 = np.hstack((qy1_1[:, np.newaxis], qy1_2[:, np.newaxis]))

    qy0_1 = np.random.random(num_datapoints) <= parameters[0]['bij'][1, 0] # 0.1
    qy0_2 = np.random.random(num_datapoints) <= parameters[0]['bij'][1, 1] # 0.2
    qy0 = np.hstack((qy0_1[:, np.newaxis], qy0_2[:, np.newaxis]))

    to_take = np.hstack(((hj == 1)[:, np.newaxis], (hj == 1)[:, np.newaxis]))
    X = np.where(to_take == 1, qy1, qy0)
    y = np.zeros(X.shape[0], dtype=np.uint8)
    return X, hj, y


def fake_dataset_set():
    '''
        Tests the EM algorithm on a fake dataset.

    '''
    # Generate fake dataset
    X, _, y = generate_data()

    # All ys have the same label, so we should run the EM-algorithm once on all
    # the X data. This should recover parameters *similar* to the ones used to
    # generate the data.
    model = EMNaiveBayesClassifier(2)
    model.train(X, y, max_iters=100)

    print(model.parameters)

def plot_parameters(model):
    num_classes = 2
    img_shape = (28, 28)
    for i in range(num_classes):
        for j in range(model.num_hidden):
            plt.subplot(num_classes, model.num_hidden,
                        i * model.num_hidden + j + 1)
            plt.imshow(model.parameters[i]['bij'][j].reshape(img_shape))
    plt.show()


def main():
    Dataset = namedtuple('Dataset', ['inputs', 'labels'])

    # Reading in data. You do not need to touch this.
    with open("data/train-images-idx3-ubyte.gz", 'rb') as f1, open("data/train-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 60000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 60000)
        inputs = np.frombuffer(buf1, dtype='uint8',
                               offset=16).reshape(60000, 28 * 28)
        inputs = np.where(inputs > 99, 1, 0)
        labels = np.frombuffer(buf2, dtype='uint8', offset=8)
        # Change labels to even/odd
        labels = (np.mod(labels, 2) == 0).astype(np.uint8)
        data_train = Dataset(inputs, labels)

    with open("data/t10k-images-idx3-ubyte.gz", 'rb') as f1, open("data/t10k-labels-idx1-ubyte.gz", 'rb') as f2:
        buf1 = gzip.GzipFile(fileobj=f1).read(16 + 10000 * 28 * 28)
        buf2 = gzip.GzipFile(fileobj=f2).read(8 + 10000)
        inputs = np.frombuffer(buf1, dtype='uint8',
                               offset=16).reshape(10000, 28 * 28)
        inputs = np.where(inputs > 99, 1, 0)
        labels = np.frombuffer(buf2, dtype='uint8', offset=8)
        # Change labels to even/odd
        labels = (np.mod(labels, 2) == 0).astype(np.uint8)
        data_test = Dataset(inputs, labels)


    ##### Fake Data Test ########
    # TODO: Uncomment this to test your EM-algorithm.
    # fake_dataset_set()


    ### Run on MNIST #####
    # TODO: Uncomment this to run on MNIST
    # model = EMNaiveBayesClassifier(5)
    # model.train(data_train.inputs, data_train.labels, max_iters=10)
    # print('Training Accuracy:', model.accuracy(data_train.inputs, data_train.labels))
    # print('Testing Accuracy:', model.accuracy(data_test.inputs, data_test.labels))

    # TODO: Uncomment to plot the parameters
    # plot_parameters(model)

main()
