import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

from IPython.core.debugger import set_trace

class PendulumCartSim:
    def __init__(self, initial_conditions, x_hist, f_pendulum=None, M=1., m=.1, l=1, g=9.81, t_end=10., dt=.01):
        """
        Initialize the system with the given parameters.

        :param M: Mass of the cart (kg)
        :param m: Mass of the pendulum (kg)
        :param l: Length of the pendulum (m)
        :param g: Gravitational acceleration (m/s^2)
        :param initial_conditions: Initial conditions for the system [x, x_dot, theta, theta_dot]
        :param f: Dynamics function. If none, will use self.dynamics
        """
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.initial_conditions = initial_conditions
        self.t_end = t_end
        self.dt = dt

        # if f_pendulum == None:
        #     f_pendulum = self.dynamics

        self.state = x_hist

    def dynamics(self, state, t):
        """
        Default dynamics function. You can override this to customize the system's behavior.

        :param state: Current state [x, x_dot, theta, theta_dot]
        :param t: Time step
        :return: Derivatives of the state [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x, x_dot, theta, theta_dot = state
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Equations of motion for the system (pendulum on a cart)
        delta = self.m * self.l * (theta_dot**2 * sin_theta - self.g * sin_theta)
        denom = self.M + self.m - self.m * cos_theta**2

        # Acceleration of the cart
        x_ddot = (self.m * self.l * (theta_dot**2 * sin_theta - self.g * sin_theta * cos_theta) - self.m * self.l * theta_dot**2 * sin_theta) / denom
        # Angular acceleration of the pendulum
        theta_ddot = (self.g * sin_theta - cos_theta * (x_ddot + self.l * theta_dot**2 * sin_theta)) / self.l

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def animate(self):
        """
        Animate the pendulum on cart system.
        """

        # Extract the solution
        x_hist = self.state[:, 0]
        theta_hist = self.state[:, 2]

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title("Pendulum on a Cart")

        # Create lines for the cart and pendulum
        cart, = ax.plot([], [], 's', markersize=50, color='blue')  # Cart
        rod, = ax.plot([], [], '-', lw=4, color='black')  # Pendulum rod
        bob, = ax.plot([], [], 'o', markersize=20, color='red')  # Pendulum bob

        # Initialization function for the animation
        def init():
            cart.set_data([], [])
            rod.set_data([], [])
            bob.set_data([], [])
            return cart, rod, bob

        # Animation function
        def animate(i):
            # Get the current position and angle
            x_pos = x_hist[i]
            theta_pos = theta_hist[i]
            cart.set_data([x_pos], [0])  # wrap x_pos in a list
            x_pendulum = x_pos + self.l * np.sin(theta_pos)
            y_pendulum = self.l * np.cos(theta_pos)
            rod.set_data([x_pos, x_pendulum], [0, y_pendulum])
            bob.set_data([x_pendulum], [y_pendulum])

            return cart, rod, bob

        # Create the animation
        ani = FuncAnimation(fig, animate, frames=len(x_hist), init_func=init, interval=.001/self.dt, blit=True)

        # Show the animation
        plt.show()



if __name__=="__main__":
    pass