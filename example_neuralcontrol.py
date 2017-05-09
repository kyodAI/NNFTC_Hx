import numpy as np

from CoreDeepLearning.layers import Linear, Sigmoid, Dropout,PlusBias,Sum,Relu
from CoreDeepLearning.network import Seq
from CoreDeepLearning.trainers import OnlineTrainer,NeuralControl
from CoreDeepLearning.loss import SquaredLoss
from CoreDeepLearning.optim import MomentumSGD
from HxCopter.hexacopter import Hexacopter
from HxCopter.hxcontroller import HexacopterController
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# from keras.wrappers.scikit_learn import KerasRegressor



def buildneuralnetwork(input_size, output_size):
    """

    :return:
    """

    model = Seq([
        Linear(input_size, 32, initialize='random'),
        PlusBias(32),
        Sigmoid(),
        Dropout(0.9),
        # Linear(128,32),
        # PlusBias(32),
        # Relu(),
        # Dropout(0.9),
        Linear(32, out_size=output_size, initialize='random'),

        # Sum()
    ])
    neural_controller =  NeuralControl(model=model,loss=SquaredLoss(),optimizer=MomentumSGD(learning_rate=0.1, momentum=0.9))

    return neural_controller


# simulation setting
simulation_time = 10.
dt = 0.01
num_step = int(simulation_time / dt)

# command setting
cmds = np.zeros(4)

# data log
log_data = np.zeros(num_step)
log_cmd = np.zeros(num_step)
log_time = np.arange(0., simulation_time, dt)


def cmd_pt1(t):
    return 0.


def cmd_pt2(t):
    if t < 2. / dt:
        return 0.
    return 1.


def cmd_pt3(t):
    if t >= 2. / dt and t < 3. / dt:
        return 1.
    if t >= 5. / dt and t < 6. / dt:
        return -1.
    return 0.


def run(useneuralnet=True):

    if useneuralnet:
        neuralcontrol = [buildneuralnetwork(input_size=4, output_size=1) for i in range(6)]


    # define initial rotor speed
    o0 = 380
    # generate hexacopter model
    hx = Hexacopter(dt, o0)
    # generate PID for HxC
    hc = HexacopterController(dt)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # PID gain setting
    #
    hc.set_gain_w(0.25, 0.3, 0.0012)
    hc.set_gain_p(.02, 0., 0.001)
    hc.set_gain_q(.02, 0., 0.001)
    hc.set_gain_r(.015, 0.01, 0.01)
    # hc.set_gain_w(3., 0.3, 0.2)

    comp = hc.YAW

    for t in range(num_step):
        # command change
        cmds[comp] = cmd_pt3(t)

        ss = np.zeros(4)
        vs = hx.get_velosity()
        fbo = hc.update(cmds * hc.DIG_TO_RAD, ss, vs)
        if useneuralnet:

            for n,y in zip(neuralcontrol,fbo):
                n.fit(vs,y)

            fpred = np.array([n.predict(vs)[0] for n in neuralcontrol])



        aaa = np.array(np.array(fpred))
        hx.set_rotor_speed(aaa)
        hx.update()
        log_data[t] = hx.get_velosity()[comp] * hc.RAD_TO_DIG
        log_cmd[t] = cmds[comp]

    plt.figure(figsize=(5, 2))
    plt.xlim(0., simulation_time)
    plt.ylim(-1.5, 1.5)

    # plot
    print list(log_cmd)
    print list(log_data)
    plt.plot(log_time, log_cmd,c='red')
    plt.plot(log_time, log_data,c='blue')

    plt.savefig('ttt.pdf')



if __name__ == '__main__':
    run()
