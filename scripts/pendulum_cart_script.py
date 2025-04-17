import numpy as np
import matplotlib.pyplot as plt

import sys
from os import fspath
from pathlib import Path
sys.path.insert(0, fspath(Path(__file__).parents[1]))

from systems.pendulum_cart_dynamics import f_pendulum_cart_discrete
from regulators.ilqr_discrete import iLQR_discrete
from simulation.pendulum_cart_sim import PendulumCartSim


def run_scenario(x1_0=0., x1_f=0., th_0=0., th_f=np.pi, v1_0=0., v1_f=0., dt=.01, t_final=20, homogeneous=False, Q=np.array([]), R=None, Qf=None, max_iter=50, u_val=0.):

    x0 = np.array([x1_0, v1_0, th_0, 0.])
    x_goal = np.array([x1_f, v1_f, th_f, 0.])

    N = int(t_final / dt) + 1
    t_span = np.linspace(0, t_final, N)
    u_init = [np.full(1, u_val) for _ in range(N)]

    if homogeneous:
        # HOMOGENEOUS
        Q = np.diag([0., 0., 0., 0.])
        R = np.diag([(1 / 2.)**2]) * 10000
        Qf = np.diag([0., 0., 0., 0.])
    elif Q.size > 0:
        Q = np.diag([5.**2., .5**2, 5.**2, .5**2])
        R = np.diag([(1 / 2.)**2])
        Qf = np.diag([100**2, 20**2, 100**2, 20**2])

    fitzgerald = iLQR_discrete(f_pendulum_cart_discrete)
    x_traj, u_traj = fitzgerald.ilqr(x0, u_init, x_goal, Q, R, Qf, max_iter=max_iter, tol=1e-8)
    x1_traj = x_traj[:, 0]
    v1_traj = x_traj[:, 1]
    th_traj = np.rad2deg(x_traj[:, 2])

    sylvie = PendulumCartSim(x0, x_hist=x_traj, dt=dt, t_end=t_final)
    sylvie.animate()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t_span, x1_traj, c='red')
    ax[0].plot(t_span, v1_traj, c='orange')
    ax[0].set_title('Cart Position')
    ax[1].plot(t_span, th_traj, c='orange')
    ax[1].set_title('Angle (Degrees)')
    ax[2].plot(t_span, u_traj.flatten(), c='blue')
    ax[2].set_title('Input Force')
    plt.show()


##### SCENARIO 0 #####
''' DROPPING PENDULUM
x1_0 = -1.5
v1_0 = 1.
th_0 = np.pi/4
run_scenario(x1_0=x1_0, v1_0=v1_0, th_0=th_0, homogeneous=True)
# '''

##### SCENARIO 1 #####
''' CONTROLLING PENDULUM TO 0
x1_0 = 0.
x1_f = 0.
th_0 = 0.
th_f = np.pi
run_scenario(x1_0, x1_f, th_0, th_f)
# '''

##### SCENARIO 2 #####
''' MOVING CART
x1_0 = -1.5
x1_f = 1.5
th_0 = 0.
th_f = 0.
run_scenario(x1_0, x1_f, th_0, th_f)
# '''

##### SCENARIO 3 #####
''' JUGGLER
x1_0 = -2.
x1_f = 6.
th_0 = np.pi/4
th_f = np.pi/4
Q = np.diag([0., 0., 1000.**2, 0.])
R = np.diag([.0001])
Qf = np.diag([0., 0., 1000.**2, 0.])
u_init = 40
run_scenario(x1_0, x1_f, th_0, th_f, Q=Q, R=R, Qf=Qf, max_iter=50, t_final=6, dt=.005)
# '''

##### SCENARIO 4 #####
# ''' REVERSI
x1_0 = 0.
x1_f = 0.
th_0 = np.pi
th_f = 0.
Q = np.diag([0., 0., 1000.**2, 0.])
R = np.diag([.0001])
Qf = np.diag([0., 0., 1000.**2, 0.])
u_init = 40
run_scenario(x1_0, x1_f, th_0, th_f, Q=Q, R=R, Qf=Qf, max_iter=50, t_final=10, dt=.005)
# '''
