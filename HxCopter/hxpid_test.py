#coding: utf-8

#from hexacopter import Hexacopter
#from hxcontroller import HexacopterController

from HxCopter.hexacopter import Hexacopter
from HxCopter.hxcontroller import HexacopterController

import numpy as np
import matplotlib.pyplot as plt


# Display plots inline and change default figure size
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)


# simulation setting
simulation_time = 10.
dt = 0.01
num_step = int(simulation_time / dt)


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
#hc.set_gain_w(3., 0.3, 0.2)

# command setting
cmds = np.zeros(4)

# data log
log_data = np.zeros(num_step)
log_cmd  = np.zeros(num_step)
log_time = np.arange(0., simulation_time, dt)


def cmd_pt1(t):
    return 0.

def cmd_pt2(t):
    if t < 2. /dt:
        return 0.
    return 1.

def cmd_pt3(t):
    if t >= 2. / dt and t < 3. / dt:
        return 1.
    if t >= 5. / dt and t < 6./ dt:
        return -1.
    return 0.


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# simulation loop
#

# control component
comp = hc.YAW

for t in range(num_step):
    # command change
    cmds[comp] = cmd_pt3(t)

    ss = np.zeros(4)
    vs = hx.get_velosity()
    fbo = hc.update(cmds * hc.DIG_TO_RAD, ss, vs)
    hx.set_rotor_speed(fbo)
    hx.update()
    log_data[t] = hx.get_velosity()[comp] * hc.RAD_TO_DIG
    log_cmd[t]  = cmds[comp]

# graph setting
plt.figure(figsize=(5, 2))
plt.xlim(0., simulation_time)
plt.ylim(-1.5, 1.5)

# plot
plt.plot(log_time, log_cmd)
plt.plot(log_time, log_data)
plt.show()
