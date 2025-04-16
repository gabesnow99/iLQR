import numpy as np

m, g, l, b = 1., 9.81, .1, .1

def f_pendulum_continuous(x, u):

    th = x.item(0)
    th_dot = x.item(1)
    th_ddot = (u.item(0) - m*g*l*np.sin(th) - b*th_dot) / (m * l**2)

    to_return = np.array([th_dot, th_ddot]).reshape(np.shape(x))
    return to_return

def f_pendulum_discrete(x, u, dt=.001):
    x_dot = f_pendulum_continuous(x, u)
    return x + x_dot * dt

if __name__=='__main__':
    import matplotlib.pyplot as plt

    dt = .001
    x_0 = np.array([np.pi/4, 0.])
    t_final = 10
    n = int(t_final / dt) + 1
    t_span = np.linspace(0, t_final, n)

    x_new = np.copy(x_0)
    theta_hist = []
    for ii in t_span:
        x_new = f_pendulum_discrete(x_new, np.array([0.]), dt)
        theta_hist.append(x_new.item(0))
    plt.plot(t_span, theta_hist)
    plt.show()
