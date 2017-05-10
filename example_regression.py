from numpy import sum, cos, pi, array
import numpy as np
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from CoreDeepLearning.layers import Linear, Sigmoid, Dropout, PlusBias, Sum, Relu, RegularizedLinear, Mul
from CoreDeepLearning.network import Seq
from CoreDeepLearning.trainers import OnlineTrainer, NeuralControl, MinibatchTrainer
from CoreDeepLearning.loss import SquaredLoss, MSE
from CoreDeepLearning.optim import MomentumSGD, AdaGrad, RMSProp
from HxCopter.hexacopter import Hexacopter
from HxCopter.hxcontroller import HexacopterController
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import cm

default_colormap = cm.jet  # the default colormap that we will use for our plots


def plot_problem_controur_keras(problem, bounds, optimum=None,
                                resolution=100., cmap=default_colormap, rstride=3, cstride=3, linewidth=0.1, alpha=1,
                                ax=None):
    'Plots a given deap benchmark problem as a countour plot'
    (minx, miny), (maxx, maxy) = bounds
    x_range = np.arange(minx, maxx, (maxx - minx) / resolution)
    y_range = np.arange(miny, maxy, (maxy - miny) / resolution)

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros((len(x_range), len(y_range)))

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            Z[i, j] = problem(np.array([[x_range[i], y_range[j]]]))

    if not ax:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
    z = Z.ravel()
    minnf, maxxf = z.min(), z.max()
    levels = np.linspace(minnf, maxxf, 80)
    cset = ax.contourf(X, Y, Z, cmap=cmap, rstride=rstride, cstride=cstride, levels=levels, linewidth=linewidth,
                       alpha=alpha)

    if optimum:
        ax.plot(optimum[0], optimum[1], 'bx', linewidth=4, markersize=15)
    return ax, cset


def plot_problem_controur(problem, bounds, optimum=None,
                          resolution=100., cmap=default_colormap, rstride=3, cstride=3, linewidth=0.1, alpha=1,
                          ax=None):
    'Plots a given deap benchmark problem as a countour plot'
    (minx, miny), (maxx, maxy) = bounds
    x_range = np.arange(minx, maxx, (maxx - minx) / resolution)
    y_range = np.arange(miny, maxy, (maxy - miny) / resolution)

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros((len(x_range), len(y_range)))

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            Z[i, j] = problem(np.array([x_range[i], y_range[j]]))

    if not ax:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
    z = Z.ravel()
    minnf, maxxf = z.min(), z.max()
    levels = np.linspace(minnf, maxxf, 80)
    cset = ax.contourf(X, Y, Z, cmap=cmap, rstride=rstride, cstride=cstride, levels=levels, linewidth=linewidth,
                       alpha=alpha)

    if optimum:
        ax.plot(optimum[0], optimum[1], 'bx', linewidth=4, markersize=15)
    return ax, cset


def sphere(phenome):
    """The bare-bones sphere function."""
    return sum(x ** 2 for x in phenome)


def rastrigin(x):
    x = array(x)
    return 10.0 * len(x) + sum(x ** 2.0 - 10.0 * cos(2.0 * pi * x))


X = np.random.uniform(-1, 1, (1000, 2))
np.random.shuffle(X)
Y = [rastrigin(i) for i in X]

fig = plt.figure()
ax = fig.add_subplot(122)
# ax, cs = plot_problem_controur(rastrigin, ((-1, -1), (1, 1)), ax=ax)
# ax.set_aspect('equal')


def neuralnet_coredeeplearning():
    model = Seq(
        Linear(2, 128),
        PlusBias(128),
        Sigmoid(),
        # Linear(256, 256), PlusBias(256),
        # Sigmoid(),
        # Linear(256, 256), PlusBias(256), Relu(),
        # Linear(128, 128), PlusBias(128),# Sigmoid(),
        Linear(128,1),
        # Sum(),
    )
    # neural_controller = NeuralControl(model=model, loss=SquaredLoss(),
    #                                   optimizer= RMSProp(learning_rate=0.1,decay_rate=0.9)
    #                                   # AdaGrad(learning_rate=0.1),
    #                                   # MomentumSGD(learning_rate=0.1, momentum=0.9)
    #                                   )
    trainset = list(zip(X, Y))
    np.random.shuffle(trainset)
    MinibatchTrainer().train_minibatches(model=model,
                                         train_set=trainset,
                                         batch_size=32, loss=MSE(),
                                         optimizer= RMSProp(learning_rate=0.1,decay_rate=0.8),
                                         # optimizer=AdaGrad(learning_rate=0.1),
                                         epochs=200,
                                         show_progress=True)

    print model.forward(np.array([0.5,0.5]))
    return model.forward


def neuralnet_keras():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation

    # y = np.reshape( y, (-1,1))
    # print y
    # exit()
    global Y
    Y = np.reshape(Y, newshape=(-1, 1))
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=256, init='normal', activation='relu'))
    model.add(Dense(input_dim=256, output_dim=256, init='normal', activation='sigmoid'))
    model.add(Dense(input_dim=256, output_dim=128, init='normal', activation='relu'))
    # model.add(Dense(input_dim=128, output_dim=128, init='normal', activation='sigmoid'))
    # model.add(Dense(input_dim=128, output_dim=128, init='normal', activation='relu'))
    # model.add(Dense(input_dim=2, output_dim=128, init='normal', activation='sigmoid'))

    model.add(Dense(input_dim=128, output_dim=1))
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')

    model.fit(X, Y, nb_epoch=50, batch_size=16)
    return model.predict


model = neuralnet_coredeeplearning()
# model = neuralnet_keras()

lb, ub = [-1, 1]
ax2 = fig.add_subplot(121)
ax2, _ = plot_problem_controur(model, ((lb, lb), (ub, ub)), ax=ax2)
ax2.set_aspect('equal')

plt.savefig('rastrigin.pdf')
