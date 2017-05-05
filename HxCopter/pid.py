#coding: utf-8

import numpy as np

class PID:
    """Basic PID unit for the simulation."""

    def __init__(self, dt, Kp=0.2, Ki=0.0, Kd=0.0):
        """Generates PID controller unit.

        Parameters
        ----------
        dt : float
            The simulation time step width [sec]

        Kp : float
            Proportional gain

        Ki : float
            Integral gain

        Kd : float
            Differential gain

        """

        self.dt = dt
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Integral of error
        self.Ei = 0.0
        # Last error
        self.Ep = 0.0

    def update(self, setpoint, feedback):
        """Returns PID output.

        Parameters
        ----------
        setpoint : float
            Setpoint

        feedback : float
            Feedback value
        """

        error = setpoint - feedback

        # integration
        self.Ei += error * self.dt
        # differentiation
        Ed = error - self.Ep

        # output
        u = self.Kp * error + self.Ki * self.Ei + self.Kd * Ed

        # update past error
        self.Ep = error

        return u


class PIDs:
    def __init__(self, dt, dim):
        '''Generates the multi dimensional, N input N output, PID controller unit.
        
        Initially all PID gains are set into 0.0, and should be set with the method.
        PID gains can be set for each command component as bellow: [Kp, Ki, Kd].
        
        Parameters
        ----------
        dt : float
            The simulation time step width [sec]
            
        dim : int
            The dimension of command input
            
        '''

        self.dt  = dt
        self.dim = dim

        # initialize PID gains by zero vectors.
        self.kp = np.zeros(dim)
        self.ki = np.zeros(dim)
        self.kd = np.zeros(dim)

        # initialize error
        self.Eis = np.zeros(dim)
        self.E_pasts = np.zeros(dim)



    def set_gain(self, comp, kp=0., ki=0., kd=0. ):
        '''Sets PID gain for 1 component.
        
        Unnecessary gains are optimal. 
        When a gain is omitted, the gain is set to 0.0 by default.
        
        Parameters
        ----------
        comp : int
            The index of the component to be set
            
        kp : float | int
            Proportional gain

        ki : float | int
            Integral gain

        kd : float | int
            Differential gain
        
        '''

        # dimension check
        if comp < 0 or comp > self.dim:
            raise ValueError('Out of the dimension range component is requested.')

        # type check for argument of PID gain
        if not isinstance(kp, (int, float)):
            raise ValueError('PID gain Kp must be float or integer.')
        if not isinstance(ki, (int, float)):
            raise ValueError('PID gain Ki must be float or integer.')
        if not isinstance(kd, (int, float)):
            raise ValueError('PID gain Kd must be float or integer.')

        # type conversion to float
        kpf = float(kp)
        kif = float(ki)
        kdf = float(kd)

        # set gains
        self.kp[comp] = kpf
        self.ki[comp] = kif
        self.kd[comp] = kdf


    def update(self, setpoints, states):
        '''Calculates PID command vector.
        
        The negative feedback is used to make error.
        
        Parameters
        ----------
        setpoints : [float ... float] | length == self.dim
            Command vector
        
        states : [float ... float] | length == self.dim
            Feedback state vector
        
        '''
        # dimension check
        if not len(setpoints) == self.dim:
            raise ValueError('The length of setpoints does\'t match to PID dimension.')
        if not len(states) == self.dim:
            raise ValueError('The length of states does\'t match to PID dimension.')

        # calculate error
        Es = setpoints - states
        self.Eis += Es * self.dt
        self.Eds  = (Es - self.E_pasts) / self.dt

        # PID command
        Us = self.kp * Es + self.ki * self.Eis + self.kd * self.Eds

        # update past states
        self.E_pasts = Es

        return Us

if __name__ == '__main__':
    pids = PIDs(0.01, 4)
    pids.set_gain(2, kp=2, ki=1)
    print(pids.kp)
    print(pids.update(np.ones(4), np.zeros(4)))
    print(pids.update(np.ones(4), np.zeros(4)))