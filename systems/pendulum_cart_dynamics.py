import numpy as np

from IPython.core.debugger import set_trace

def f_pendulum_cart_continuous(x, u, M=1, m=.1, l=.25, g=9.81, b_th=.1):

    x1 = x.item(0)
    x1_dot = x.item(1)
    th = x.item(2)
    th_dot = x.item(3)

    F = u.item(0)

    try:
        th_dot**2
    except:
        set_trace()

    den = M + m - m*np.cos(th)**2
    x1_ddot = (F + m*l*th_dot**2*np.sin(th) - m*g*np.sin(th)*np.cos(th)) / den
    th_ddot = (-F*np.cos(th) -b_th*th_dot - m*l*th_dot**2*np.sin(th)*np.cos(th) + (M + m)*g*np.sin(th)) / (l * den)

    # den = M + m * np.sin(th)**2
    # x1_ddot = (F + m * l * th_dot**2 * np.sin(th) + m * g * np.sin(th) * np.cos(th)) / den
    # th_ddot = (-g * np.sin(th) - b_th*th_dot - np.cos(th) * (F + m * l * th_dot**2 * np.sin(th))) / (l * (4/3 * m + M))

    to_return = np.array([x1_dot, x1_ddot, th_dot, th_ddot]).reshape(np.shape(x))
    return to_return

def f_pendulum_cart_discrete(x, u, dt):
    x_dot = f_pendulum_cart_continuous(x, u)
    return x + x_dot * dt

if __name__=='__main__':
    pass