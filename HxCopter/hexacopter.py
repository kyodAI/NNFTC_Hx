import numpy as np

class Hxcopter:
    g = 9.81 # gravity coefficient [ m/s^2]
    m = 2.1 # mass [kg]
    L = 0.34 # Distance of rotor from CG [m]
    I = (0.061, 0.060, 0.12) # moments of inertia (Ixx, Iyy, Izz) [kg*m^2]
    kT = 2.3 * 10 ** (-5) # propeller thrust [rad^2/sec^2]
    kQ = 7 * 10 ** (-7) # torque coefficients [rad^2/sec^2]

    def __init__(self, dt, omega0):
        ```Initiates with generating matrix of EOM.```
        self.dt = dt # simulation time step [sec]
        self.rotors = [Rotor(omega0, dt),
                        Rotor(omega0, dt),
                        Rotor(omega0, dt),
                        Rotor(omega0, dt),
                        Rotor(omega0, dt),
                        Rotor(omega0, dt)]
        self.state = np.array([0.0, 0.0, 0.0, 0.0]) # altitude [m], roll, pitch, yaw [rad]
        self.dstate = np.array([0.0, 0.0, 0.0, 0.0]) # vertical speed [m/s], roll, pitch, yaw speed [rad/sec]

        # Matrix of EOM
        Ew = (- self.kT / self.m) * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        Ep = (1.73 * self.L * self.kT / 2 / self.I[0]) * np.array([0, -1.0, -1.0, 0, 1.0, 1.0])
        Eq = (self.L * self.kT / self.I[1]) * np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
        Er = (self.kQ / self.I[2]) * np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, ])
        self.E = np.array([Ew, Ep, Eq, Er])

    def get_state(self):
        ```Returns state vector as below:
        s = [z phi theta psi]
        , where z is the vertical position with downward direction and, phi, theta and psi are body angles [rad] with positive clockwise around the x, y, z axis respectively.```
        return self.state

    def get_velosity(self):
        ```Returns velocity vector as below:
        ds/dt = [w p q r]
        where w is the vertical velocity [m/s] with downward direction, and p, q, and q are the body angle speed [rad/s] with positive clockwise around the x, y and z axis respectively.```
        return self.dstate

    def func(self, x, y):
        ```Equation of Motion of hexacopter.```
        o = np.array([0, 0, 0, 0, 0, 0])
        for i in range(6):
            o[i] = self.rotors[i].o
        dx = y
        dy = self.E.dot(o * o) + np.array([self.g, 0.0, 0.0, 0.0])
        return dx, dy

    def update(self):
        for r in self.rotors:
            r.update()
        kx1, ky1= self.func(self.state, self.dstate)
        kx2, ky2 = self.func(self.state + kx1 * self.dt / 2, self.dstate + ky1 * self.dt / 2)
        kx3, ky3 = self.func(self.state + kx2 * self.dt / 2, self.dstate + ky2 * self.dt / 2)
        kx4, ky4 = self.func(self.state + kx3 * self.dt, self.dstate + ky3 * self.dt)
        self.dstate += (ky1 + 2 * ky2 + 2 * ky3 + ky4) * self.dt / 6
        self.state += (kx1 + 2 * kx2 + 2 * kx3 + kx4) * self.dt / 6
        return self.state

    def set_rotor_speed(self, cmds):
        for i in range(6):
            self.rotors[i].set_speed(cmds[i])

    class Rotor:
    tau = 0.15 # motor time constant

    def __init__(self, omega0, dt):
        self.o = omega0 # omega = rotor speed
        self.cmd = omega0 # speed commond
        self.dt = dt # simulation time step width

    def func(self, o):
        return (self.cmd - o) / self.tau

    def domega(self):
        k1 = self.func(self.o)
        k2 = self.func(self.o + k1 * self.dt / 2)
        k3 = self.func(self.o + k2 * self.dt / 2)
        k4 = self.func(self.o + k3 * self.dt)
        return (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

    def update(self):
        self.o += self.domega()
        return self.o

    def set_speed(self, cmd):
        self.cmd = cmd
