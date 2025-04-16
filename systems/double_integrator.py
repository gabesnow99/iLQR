import numpy as np

def double_integrator(x, u, dt=0.01):
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0 ],
                  [0, 0, 0, 1 ]])
    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    return A @ x + B @ u
