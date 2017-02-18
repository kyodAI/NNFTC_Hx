import numpy as np
from scipy.stats import bernoulli
import argparse
import copy


#############################
### Core functions
#############################

def initNetwork(nn_arch, act_func_name):
    """
        Initialize the neural network weights, activation function and return the number of parameters
  
        :param nn_arch: the number of units per hidden layer 
        :param act_func_name: the activation function name (sigmoid, tanh or relu)
        :type nn_arch: list of int
        :type act_func_name: str
        :return W: a list of weights for each hidden layer
        :return B: a list of bias for each hidden layer
        :return act_func: the activation function
        :return nb_params: the number of parameters 
        :rtype W: list of ndarray
        :rtype B: list of ndarray
        :rtype act_func: function
        :rtype n_params: number of parameters
    """

    W, B = [], []
    sigma = 1.0
    act_func = globals()[act_func_name]  # Cast the string to a function
    nb_params = 0

    if act_func_name == 'sigmoid':
        sigma = 4.0

    for i in range(np.size(nn_arch) - 1):
        w = np.random.normal(loc=0.0, scale=sigma / np.sqrt(nn_arch[i]), size=(nn_arch[i + 1], nn_arch[i]))
        W.append(w)
        b = np.zeros((w.shape[0], 1))
        if act_func_name == 'sigmoid':
            b = np.sum(w, 1).reshape(-1, 1) / -2.0
        B.append(b)
        nb_params += nn_arch[i + 1] * nn_arch[i] + nn_arch[i + 1]

    return W, B, act_func, nb_params


def forward(act_func, W, B, X, drop_func=None):
    """
        Perform the forward propagation

        :param act_func: the activation function 
        :param W: the weights
        :param B: the bias
        :param X: the input
        :param drop_func: the dropout function
        :type act_func: function
        :type W: list of ndarray
        :type B: list of ndarray
        :type X: ndarray
        :type drop_func: function
        :return H: a list of activation values
        :return Hp: a list of the derivatives w.r.t. the pre-activation of the activation values
        :rtype H: list of ndarray
        :rtype Hp: list of ndarray
    """

    if drop_func:
        mask, X, _ = drop_func(0.8, X)  # Hard-coded probability
    H, Hp = [np.transpose(X)], []
    for k in range(len(W) - 1):
        h, hp = act_func(W[k].dot(H[k]) + B[k])
        if drop_func:
            mask, h, hp = drop_func(0.5, h, hp)  # hard-coded probability
        H.append(h)
        Hp.append(hp)
    z = W[-1].dot(H[-1]) + B[-1]
    H.append(z)
    return H, Hp


def backward(error, W, Hp):
    """
        Perform the backward propagation

        :param error: the gradient w.r.t. to the last layer 
        :param W: the weights
        :param Hp: the derivatives w.r.t. the pre-activation of the activation functions
        :type error: ndarray
        :type W: list of ndarray
        :type Hp: list of ndarray
        :return gradb: a list of gradient w.r.t. 
        :rtype gradB: list of ndarray
    """

    gradB = [error]
    for k in reversed(range(1, len(W))):
        grad_b = W[k].T.dot(gradB[0]) * Hp[k - 1]
        gradB.insert(0, grad_b)
    return gradB


def update(eta, batch_size, W, B, gradB, H, regularizer, my_lambda):
    """
        Perform the update of the parameters

        :param eta: the step-size of the gradient descent 
        :param batch_size: number of examples in the batch (for normalizing)
        :param W: the weights
        :param B: the bias
        :param gradB: the gradient of the activations w.r.t. to the loss
        :param H: the activation values
        :param regularizer: the regularizater name
        :param my_lambda: the amplitude of regularization
        :type eta: float
        :type batch_size: int
        :type W: list of ndarray
        :type B: list of ndarray
        :type gradB: list of ndarray
        :type H: list of ndarray
        :type regularizer: str
        :type my_lambda: float
        :return W: the weights updated 
        :return B: the bias updated 
        :rtype W: list of ndarray
        :rtype B: list of ndarray
    """

    for k in range(len(gradB)):
        grad_w = gradB[k].dot(H[k].T) / batch_size
        grad_b = np.sum(gradB[k], 1).reshape(-1, 1) / batch_size

        W[k] = updateParams(W[k], grad_w, eta, regularizer, my_lambda)
        B[k] = updateParams(B[k], grad_b, eta, regularizer, my_lambda)

    return W, B


#############################
#############################

#############################
### Activation functions
#############################
def sigmoid(z, grad_flag=True):
    """
        Perform the sigmoid transformation to the pre-activation values

        :param z: the pre-activation values
        :param grad_flag: flag for computing the derivatives w.r.t. z
        :type z: ndarray
        :type grad_flag: boolean
        :return h: the activation values
        :return hp: the derivatives w.r.t. z
        :rtype h: ndarray
        :rtype hp: ndarray
    """

    try:
        h = 1. / (1. + np.exp(-z))
    except Exception:
        np.clip(z, -30., 30., z)
        h = 1. / (1. + np.exp(-z))
    if grad_flag:
        hp = np.multiply(h, (1 - h))
        return h, hp
    else:
        return h


def tanh(z, grad_flag=True):
    """
        Perform the tanh transformation to the pre-activation values

        :param z: the pre-activation values
        :param grad_flag: flag for computing the derivatives w.r.t. z
        :type z: ndarray
        :type grad_flag: boolean
        :return h: the activation values
        :return hp: the derivatives w.r.t. z
        :rtype h: ndarray
        :rtype hp: ndarray
    """

    try:
        h = np.tanh(z)
    except Exception:
        np.clip(z, -30., 30., z)
        h = np.tanh(z)
    if grad_flag:
        hp = 1.0 - np.square(h)
        return h, hp
    else:
        return h


def relu(z, grad_flag=True):
    """
        Perform the relu transformation to the pre-activation values

        :param z: the pre-activation values
        :param grad_flag: flag for computing the derivatives w.r.t. z
        :type z: ndarray
        :type grad_flag: boolean
        :return h: the activation values
        :return hp: the derivatives w.r.t. z
        :rtype h: ndarray
        :rtype hp: ndarray
    """

    try:
        h = np.maximum(np.zeros(z.shape), z)
    except Exception:
        np.clip(z, -30., 30., z)
        h = np.maximum(np.zeros(z.shape), z)
    if grad_flag:
        hp = np.zeros(z.shape)
        hp[z > 0.0] = 1.0
        return h, hp
    else:
        return h


def softmax(z):
    """
        Perform the softmax transformation to the pre-activation values

        :param z: the pre-activation values
        :type z: ndarray
        :return: the activation values
        :rtype: ndarray
    """
    return np.exp(z - np.max(z, 0)) / np.sum(np.exp(z - np.max(z, 0)), 0.)


#############################

#############################
## Regularization
#############################
def dropout(p, h, hp=None):
    """
        Perform the dropout transformation to the activation values

        :param p: the probability of dropout
        :param h: the activation values
        :param hp: the derivatives w.r.t. the pre-activation values
        :type p: float
        :type h: ndarray
        :type hp: ndarray
        :return mask: the bernoulli mask
        :return h: the transformed activation values
        :return hp: the transformed derivatives w.r.t. z
        :rtype h: ndarray
        :rtype hp: ndarray
    """

    mask = bernoulli.rvs(p, size=h.shape)
    h = np.multiply(h, mask)
    if hp is not None:
        hp = np.multiply(hp, mask)
    return mask, h, hp


def updateParams(theta, dtheta, eta, regularizer=None, my_lambda=0.):
    """
        Perform the update of the parameters with the 
        possibility to do L1 or L2 regularization 

        :param theta: the network parameters
        :param dtheta: the updates of the parameters
        :param eta: the step-size of the gradient descent 
        :param regularizer: the name of the regularizer
        :param my_lambda: the value of the regularizer
        :type theta: ndarray
        :type dtheta: ndarray
        :type eta: float
        :type regularizer: str
        :type my_lambda: float
        :return: the parameters updated 
        :rtype: ndarray
    """

    if regularizer == None:
        return theta - eta * dtheta
    elif regularizer == 'L1':
        return theta - eta * my_lambda * np.sign(theta) - eta * dtheta
    elif regularizer == 'L2':
        return (1. - eta * my_lambda) * theta - eta * dtheta
    else:
        raise NotImplementedError


#############################

#############################
## Auxiliary functions 
#############################
def getMiniBatch(i, batch_size, train_set, one_hot):
    """
        Return a minibatch from the training set and the associated labels

        :param i: the identifier of the minibatch
        :param batch_size: the number of training examples
        :param train_set: the training set
        :param one_hot: the one-hot representation of the labels
        :type i: int
        :type batch_size: int
        :type train_set: ndarray
        :type ont_hot: ndarray
        :return: the minibatch of examples
        :return: the minibatch of labels
        :return: the number of examples in the minibatch
        :rtype: ndarray
        :rtype: ndarray
        :rtype: int
    """

    ### Mini-batch creation
    n_training = np.size(train_set[1])
    idx_begin = i * batch_size
    idx_end = min((i + 1) * batch_size, n_training)
    mini_batch_size = idx_end - idx_begin

    batch = train_set[0][idx_begin:idx_end]
    one_hot_batch = one_hot[:, idx_begin:idx_end]

    return np.asfortranarray(batch), one_hot_batch, mini_batch_size


def predict(W, B, batch, act_func):
    h = np.transpose(batch)
    for k in range(len(W) - 1):
        h, hp = act_func(W[k].dot(h) + B[k])
    z = W[-1].dot(h) + B[-1]

    ### Compute the softmax
    out = softmax(z)
    pred = np.argmax(out, axis=0)
    return pred


def computeLoss(W, B, batch, labels, act_func):
    """
        Compute the loss value of the current network on the full batch

        :param W: the weights
        :param B: the bias
        :param batch: the weights
        :param labels: the bias
        :param act_func: the weights
        :type W: ndarray
        :type B: ndarray
        :type batch: ndarray
        :type act_func: function
        :return loss: the negative log-likelihood
        :return accuracy: the ratio of examples that are well-classified
        :rtype: float
        :rtype: float
    """

    ### Forward propagation
    h = np.transpose(batch)
    for k in range(len(W) - 1):
        h, hp = act_func(W[k].dot(h) + B[k])
    z = W[-1].dot(h) + B[-1]

    ### Compute the softmax
    out = softmax(z)
    pred = np.argmax(out, axis=0)
    fy = out[labels, np.arange(np.size(labels))]
    try:
        loss = np.sum(-1. * np.log(fy)) / np.size(labels)
    except Exception:
        fy[fy < 1e-4] = fy[fy < 1e-4] + 1e-6
        loss = np.sum(-1. * np.log(fy)) / np.size(labels)
    accuracy = np.sum(np.equal(pred, labels)) / float(np.size(labels))

    return loss, accuracy


def updateEpoch(eta, loss, prev_loss, W, B, pW, pB, drop_func):
    """
        Perform step-size adaptation 

        :param eta: the step-size of the gradient descent 
        :param loss: the negative log-likelihood of the current epoch
        :param prev_loss: the negative log-likelihood of the previous epoch
        :param W: the weights
        :param B: the bias
        :param pW: the weights of the previous epoch
        :param pB: the bias of the previous epoch
        :param drop_func: the dropout_func
        :type eta: float
        :type loss: float
        :type W: ndarray
        :type B: ndarray
        :type pW: ndarray
        :type pB: ndarray
        :type drop_func: function
        :return eta: the updated step-size of the gradient descent 
        :return prev_loss: the updated negative log-likelihood of the previous epoch
        :return W: the updated weights
        :return B: the updated bias
        :return pW: the updated weights of the previous epoch
        :return pB: the updated bias of the previous epoch
        :rtype eta: float
        :rtype prev_loss: float
        :rtype W: ndarray
        :rtype B: ndarray
        :rtype pW: ndarray
        :rtype pB: ndarray

    """

    if not drop_func:
        if loss < prev_loss:
            eta *= 1.2
            pW = copy.deepcopy(W)
            pB = copy.deepcopy(B)
            prev_loss = loss
        else:
            eta *= 0.5
            W = copy.deepcopy(pW)
            B = copy.deepcopy(pB)

    return eta, prev_loss, W, B, pW, pB


def parseArgs():
    # Retrieve the arguments
    parser = argparse.ArgumentParser(description='MiniNN -- Minimalist code for Neural Network Learning')
    parser.add_argument('--arch', help='Architecture of the hidden layers', default=[100], nargs='+', type=int)
    parser.add_argument('--act_func', help='Activation function name', default="sigmoid", type=str)
    parser.add_argument('--batch_size', help='Minibatch size', default=500, type=int)
    parser.add_argument('--dropout_flag', help='Flag for dropout', default=False, type=bool)
    parser.add_argument('--eta', help='Step-size for the optimization algorithm', default=1.0, type=float)
    parser.add_argument('--lambda_reg', help='Value of the lambda associated to the regularizer', default=0.,
                        type=float)
    parser.add_argument('--n_epoch', help='Number of epochs', default=5000, type=int)
    parser.add_argument('--regularizer', help='Name of the regularizer', default=None, type=str)

    args = parser.parse_args()
    return args


#############################

#############################
## Printing function
#############################
def printDescription(algo_name, eta, nn_arch, act_func_name, minibatch_size, regularizer, drop_func, nb_param, lamda=0,
                     sparsity=0):
    print("Description of the experiment")
    print("----------")
    print("Learning algorithm: " + algo_name)
    print("Initial step-size: " + str(eta))
    print("Network Architecture: " + str(nn_arch))
    print("Number of parameters: " + str(nb_param))
    print("Minibatch size: " + str(minibatch_size))
    print("Activation: " + act_func_name)
    if regularizer:
        print("Regularization: " + regularizer + " with " + str(lamda))
    if drop_func:
        print("Dropout: " + drop_func.__name__)
    print("----------")

#############################
