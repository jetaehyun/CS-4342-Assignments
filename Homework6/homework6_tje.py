from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient


# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which))
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))

    return images, labels


# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    W1 = w[:NUM_INPUT*NUM_HIDDEN].reshape(40, 784)
    b1 = w[NUM_INPUT*NUM_HIDDEN: NUM_INPUT*NUM_HIDDEN+NUM_HIDDEN].reshape(40)
    W2 = w[NUM_INPUT*NUM_HIDDEN+NUM_HIDDEN: NUM_INPUT*NUM_HIDDEN+NUM_HIDDEN+NUM_HIDDEN*NUM_OUTPUT].reshape(10, 40)
    b2 = w[NUM_INPUT*NUM_HIDDEN+NUM_HIDDEN+NUM_HIDDEN*NUM_OUTPUT:].reshape(10)

    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return np.concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))


def plotGraph(x1, y1, title1, x2, y2, title2, plot_title):
    plt.plot(x1, y1, label=title1)
    plt.plot(x2, y2, label=title2)
    plt.title(plot_title)
    plt.legend()
    plt.show()


def one_hot_label(label):
    shape = (label.size, label.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(label.size)
    one_hot[rows, label] = 1

    return one_hot


def shuffleData(X, Y):
    shuffler = np.random.permutation(X.shape[0])

    return X[shuffler], Y[shuffler]


def getValidationSet(X, Y, percentage=0.2):
    return train_test_split(X, Y, test_size=percentage, random_state=42)


def accuracy(y, y_hat):
    return np.mean(np.argmax(y.T, axis=0) == np.argmax(y_hat, axis=0))


def ReLU(z):
    z[z <= 0] = 0

    return z


def ReLU_prime(z):
    z[z <= 0] = 0
    z[z > 0] = 1

    return z


def softmax(z):
    z = z.T
    preActivScores = np.exp(z)

    return np.divide(preActivScores.T, np.sum(preActivScores, axis=1))


def forw_prop(W1, b1, W2, b2, X):

    z1 = W1.dot(X) + b1[:,None]
    h1 = ReLU(z1)
    z2 = W2.dot(h1) + b2[:,None]
    y_hat = softmax(z2)

    return z1, h1, z2, y_hat


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    X = X.T
    Y = Y.T
    W1, b1, W2, b2 = unpack(w)

    net_outputs = forw_prop(W1, b1, W2, b2, X)
    y_hat = np.log(net_outputs[3])

    cost = -np.mean(np.sum(Y * y_hat, axis=0))

    return cost


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w, b=0.0001):
    W1, b1, W2, b2 = unpack(w)
    Y = Y.T
    X = X.T

    net_outputs = forw_prop(W1, b1, W2, b2, X)
    z1 = net_outputs[0]
    h1 = net_outputs[1]
    y_hat = net_outputs[3]

    _w2 = ((y_hat - Y).dot(h1.T) + b*np.sign(W2)) / X.shape[1]
    _b2 = np.mean(y_hat - Y, axis=1)

    g = (((y_hat - Y).T.dot(W2)) * ReLU_prime(z1.T)).T

    _w1 = (g.dot(X.T) + b*np.sign(W1)) / X.shape[1]
    _b1 = np.mean(g, axis=1)

    return pack(_w1, _b1, _w2, _b2)


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w, EPSILON=0.1, batchSize=100, numEpoch=50, b=0.0001):

    rounds = int(trainX.shape[0] / batchSize)
    CELoss = []
    accuracies = []

    for e in range(numEpoch):

        for m in range(rounds):
            start = m * batchSize
            end = start + batchSize

            x, y = trainX[start:end:], trainY[start:end:]

            grad = gradCE(x, y, w, b)

            w -= (EPSILON * grad)

        if e >= numEpoch - 20:
            w1, b1, w2, b2 = unpack(w)
            y_hat = forw_prop(w1, b1, w2, b2, testX.T)[3]
            acc = accuracy(testY, y_hat)
            cost = fCE(testX, testY, w)

            CELoss.append(cost)
            accuracies.append(acc)
            print(f'Accuracy: {acc}, Cost: {cost}')

    # _x = list(range(20))
    # plotGraph(_x, CELoss, "fCE", _x, accuracies, "accuracy", "fCE and Accuracy, First 20 Epochs")

    w1, b1, w2, b2 = unpack(w)
    y_hat = forw_prop(w1, b1, w2, b2, testX.T)[3]

    acc = accuracy(testY, y_hat)
    cost = fCE(testX, testY, w)
    print(f'test set: Accuracy: {acc}, Cost: {cost}')

    return acc, cost, w


def findBestHyperparameters(X, Y, learning_rates, batchSizes, epochs, w, b):
    Xtr, Xte, ytr, yte = getValidationSet(X, Y)
    accuracy = 0.0
    cost = 100
    params = []
    param_best = None

    for i in range(10):
        rate = np.random.choice(learning_rates)
        e = np.random.choice(epochs)
        batch = np.random.choice(batchSizes)
        beta = np.random.choice(b)

        param_set = (rate, batch, e, beta)

        if param_set in params:
            i -= 1
            continue


        W = np.copy(w)
        print(f'params: Learning Rate={rate}, batchSize={batch}, epochs={e}, beta={beta}')
        a, c, _w = train(Xtr, ytr, Xte, yte, W, rate, batch, e, beta)

        if a > accuracy and c < cost:
            accuracy = a
            cost = c
            param_best = param_set

        params.append(param_set)

    print(f'best param: {param_best}')




if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    trainX = np.divide(trainX, 255)
    testX = np.divide(testX, 255)
    # testY = np.divide(testY, 255)
    # trainY = np.divide(trainY, 255)


    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)
    trainY = one_hot_label(trainY)
    testY = one_hot_label(testY)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK] # (5,)

    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))
    # best param: (0.1, 128, 50, 0.001)
    # Train the network using SGD.

    trainX, trainY = shuffleData(trainX, trainY)
    # train(trainX, trainY, testX, testY, w, EPSILON=0.1, batchSize=128, numEpoch=50, b=0.001)

    # learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    # minibatch_size = [16, 32, 64, 128, 256]
    # epochs = [10, 20, 30, 40, 50]
    # beta = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # findBestHyperparameters(trainX, trainY, learning_rate, minibatch_size, epochs, w, beta)
