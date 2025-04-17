import numpy as np
import matplotlib.pyplot as plt

import sys
from os import fspath
from pathlib import Path
sys.path.insert(0, fspath(Path(__file__).parents[1]))

from systems.pendulum_dynamics import f_pendulum_discrete
from regulators.ilqr_discrete import iLQR_discrete

jim = iLQR_discrete(f_pendulum_discrete)

th_0 = np.deg2rad(15)
th_f = np.deg2rad(85)

dt = .01
x0 = np.array([th_0, 0.])
x_goal = np.array([[th_f, 0.]])
t_final = 5
N = int(t_final / dt) + 1
t_span = np.linspace(0, t_final, N)

u_n = 1

u_init = [np.zeros(u_n) for _ in range(N)]

Q = np.diag([10**2., 1.**2])
R = np.diag([(1 / 2.)**2])
Qf = np.diag([50**2, 10**2])

x_traj, u_traj = jim.ilqr(x0, u_init, x_goal, Q, R, Qf, max_iter=50, tol=1e-8)
plt.plot(t_span, np.rad2deg(x_traj[:, 0]))
plt.plot(t_span, u_traj.flatten())
plt.show()