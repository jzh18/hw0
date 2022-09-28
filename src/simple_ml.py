import struct
import numpy as np
import gzip

try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    byte_order = 'big'
    X = []
    with gzip.open(image_filename) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2051:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_rows = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_cols = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            img_one_dim = np.frombuffer(f.read(num_of_rows * num_of_cols), dtype=np.uint8).astype(np.float32)
            X.append(img_one_dim)
    X = np.array(X)
    min = np.min(X)
    max = np.max(X)
    X = (X - min) / (max - min)
    y = []
    with gzip.open(label_filename) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2049:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            label = np.frombuffer(f.read(1), dtype=np.uint8)
            y.append(label)

    return X, np.array(y).squeeze()

    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    maxes = np.max(Z, axis=1).reshape(len(Z), 1)
    shiftz = np.subtract(Z, maxes)
    exps = np.exp(shiftz)
    indices = np.stack((np.arange(len(y)), y), axis=1)
    exps_up = exps[indices[:, 0], indices[:, 1]]
    exps_down = np.sum(exps, axis=1)
    return -1 * np.mean(np.log(exps_up / exps_down))
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    start_indices = np.arange(0, len(X), batch)
    for i in start_indices:
        if i + batch > len(X):
            break
        batch_X = X[i:i + batch]
        batch_y = y[i:i + batch]

        num_examples = batch_X.shape[0]
        num_classes = theta.shape[1]

        exp_theta_x = np.exp(np.matmul(batch_X, theta))  # (num_classes, num_classes)
        normalize_z = _normalize(exp_theta_x)  # (num_examples, num_classes)


        one_hot_labels = _one_hot_labels(batch_y, num_classes)

        grad = np.matmul(np.transpose(batch_X), normalize_z - one_hot_labels) / batch  # (input_dim, num_classes)

        theta -= lr * grad


    ### END YOUR CODE


def _one_hot_labels(data, num_classes):
    """

    :param data: (num,) ndarray
    :return: (num, num_classes)
    """
    num_examples = len(data)
    one_hot_labels = np.zeros((num_examples, num_classes))  # (num_examples, num_classes)
    indices = np.stack((np.arange(len(data)), data), axis=1)
    one_hot_labels[indices[:, 0], indices[:, 1]] = 1
    return one_hot_labels


def _normalize(data):
    """

    :param data: (m,n) ndarray
    :return:
    """
    return data / np.sum(data, axis=1).reshape(len(data), 1)  # (m,n)


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    start_indices = np.arange(0, len(X), batch)
    num_classes = W2.shape[1]
    for i in start_indices:
        if i + batch > len(X):
            break
        batch_X = X[i:i + batch]  # (batch_size, input_dim)
        batch_y = y[i:i + batch]  # (batch_size,)

        # calculate Z_1
        tmp = np.matmul(batch_X, W1)
        tmp[tmp < 0] = 0
        z_1 = tmp  # (batch_size, hidden_dim)

        # calculate G_2
        G_2 = _normalize(np.exp(np.matmul(z_1, W2))) - _one_hot_labels(batch_y,
                                                                       num_classes)  # (batch_size, num_classes)

        # calculate G_1
        tmp = np.copy(z_1) # do we really need to copy it?
        tmp[tmp > 0] = 1
        z_1_large_0 = tmp  # (batch_size, hidden_dim)

        tmp = np.matmul(G_2, np.transpose(W2))  # (batch_size,hidden_dim)
        G_1 = np.multiply(z_1_large_0, tmp)  # (batch_size,hidden_dim)

        grad_w1 = np.matmul(np.transpose(batch_X), G_1) / batch
        grad_w2 = np.matmul(np.transpose(z_1), G_2) / batch

        W1 -= lr * grad_w1
        W2 -= lr * grad_w2
    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
