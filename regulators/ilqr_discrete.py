import numpy as np
from IPython.core.debugger import set_trace

class iLQR_discrete:
    def __init__(self, f, dt=.001):
        self.f = f
        self.dt = dt
        pass

    def cost(self, x, u, x_goal, Q, R):
        return (x - x_goal).T @ Q @ (x - x_goal) + u.T @ R @ u

    def finite_diff_jacobian(self, x, u, eps=1e-5):
        n = len(x)
        m = len(u)
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        fx = self.f(x, u, self.dt)
        for i in range(n):
            dx = np.zeros_like(x)
            dx[i] = eps
            A[:, i] = (self.f(x + dx, u, self.dt) - fx) / eps
        for i in range(m):
            du = np.zeros_like(u)
            du[i] = eps
            B[:, i] = (self.f(x, u + du, self.dt) - fx) / eps
        return A, B

    def ilqr(self, x0, u_init, x_goal, Q, R, Qf, max_iter=100, tol=1e-6):
        x_goal = x_goal.flatten()
        N = len(u_init)
        n = x0.shape[0]
        m = u_init[0].shape[0]

        u = u_init.copy()
        x = [x0.flatten()] + [np.zeros(n) for _ in range(N)]

        for iter in range(max_iter):

            # Forward rollout
            for t in range(N):
                x[t+1] = self.f(x[t], u[t], self.dt)

            # Cost-to-go and derivatives initialization
            Vx = Qf @ (x[-1] - x_goal)
            Vxx = Qf
            k = [np.zeros(m) for _ in range(N)]
            K = [np.zeros((m, n)) for _ in range(N)]

            # Backward pass
            for t in reversed(range(N)):
                A, B = self.finite_diff_jacobian(x[t], u[t])
                xt = x[t]
                ut = u[t]

                Qx = Q @ (xt - x_goal) + A.T @ Vx
                Qu = R @ ut + B.T @ Vx
                Qxx = Q + A.T @ Vxx @ A
                Quu = R + B.T @ Vxx @ B
                Qux = B.T @ Vxx @ A

                # Regularization (ensure Quu is PD)
                Quu_inv = np.linalg.inv(Quu + 1e-6 * np.eye(m))

                k[t] = -Quu_inv @ Qu
                K[t] = -Quu_inv @ Qux

                Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
                Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]

            # Forward pass with line search
            alpha = 1.0
            cost_prev = sum(self.cost(x[t], u[t], x_goal, Q, R) for t in range(N))
            for _ in range(10):
                x_new = [x0]
                u_new = []
                for t in range(N):
                    dx = x_new[t] - x[t]
                    ut = u[t] + alpha * k[t] + K[t] @ dx
                    xt = self.f(x_new[t], ut, self.dt)
                    u_new.append(ut)
                    x_new.append(xt)
                cost_new = sum(self.cost(x_new[t], u_new[t], x_goal, Q, R) for t in range(N))
                if cost_new < cost_prev:
                    x = x_new
                    u = u_new
                    break
                alpha *= 0.5

            # Convergence check
            if np.abs(cost_prev - cost_new) < tol:
                print(f'Converged in {iter+1} iterations.')
                break

        return np.array(x[:][:-1]), np.array(u)
