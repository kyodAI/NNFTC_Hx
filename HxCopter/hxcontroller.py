#coding: utf-8

import numpy as np
#from pid import PIDs
from HxCopter.pid import PIDs

class HexacopterController:
    '''4 dimension PID controller for the attitude of the Hexacopter.

    Controller is divided into 2 part, velocities and attitudes controller.    
    '''

    # state index
    VERTICAL = 0
    ROLL     = 1
    PITCH    = 2
    YAW      = 3

    DIG_TO_RAD = np.pi / 180.
    RAD_TO_DIG = 180. / np.pi

    # hexacopter parameters
    g = 9.81 # gravity coefficient [ m/s^2]
    m = 2.1 # mass [kg]
    L = 0.34 # Distance of rotor from CG [m]
    I = (0.061, 0.060, 0.12) # moments of inertia (Ixx, Iyy, Izz) [kg*m^2]
    kT = 2.3 * 10 ** (-5) # propeller thrust [rad^2/sec^2]
    kQ = 7 * 10 ** (-7) # torque coefficients [rad^2/sec^2]

    # inv-E
    ew = - m / (6 * kT)
    ep = 1.73 * I[0] / (6 * L * kT)
    eq = I[1] / (3 * L * kT)
    er = I[2] / (6 * kQ)
    e1 = np.array([ew,    0,   eq,     - er])
    e2 = np.array([ew, - ep,   eq / 2,   er])
    e3 = np.array([ew, - ep, - eq / 2, - er])
    e4 = np.array([ew,    0, - eq,       er])
    e5 = np.array([ew,   ep, - eq / 2, - er])
    e6 = np.array([ew,   ep,   eq / 2,   er])
    Einv = np.array([e1, e2, e3, e4, e5, e6])

    def __init__(self, dt):
        self.dt = dt
        self.pid_v = PIDs(dt, 4)

    def set_gain_w(self, kp, ki, kd):
        self.pid_v.set_gain(self.VERTICAL, kp=kp, ki=ki, kd=kd)
    def set_gain_p(self, kp, ki, kd):
        self.pid_v.set_gain(self.ROLL, kp=kp, ki=ki, kd=kd)
    def set_gain_q(self, kp, ki, kd):
        self.pid_v.set_gain(self.PITCH, kp=kp, ki=ki, kd=kd)
    def set_gain_r(self, kp, ki, kd):
        self.pid_v.set_gain(self.YAW, kp=kp, ki=ki, kd=kd)

    def update(self, setpoints, state, velocity):
        velocity_command = setpoints
        feedback_omegas = self.Einv.dot(self.pid_v.update(velocity_command, velocity))
        return feedback_omegas

if __name__ == '__main__':
    ctrl = HexacopterController(0.01)
    ctrl.set_gain_w(1., 2., 3.)
    ctrl.set_gain_p(2., 1., 0.1)
    print(ctrl.pid_v.kp)
    print(ctrl.pid_v.ki)
    print(ctrl.pid_v.kd)