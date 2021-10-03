import numpy as np
"""
Optimization - Programming Assignment
June 2020
Yotam Leibovitz
"""
# Part I
class QuadraticFunction:
    def __init__(self, Q, b):
        self.Q = np.copy(Q)
        self.b = np.copy(b)
        self.n = np.shape(Q)[0]
        # check input correctness
        if self.n != np.shape(b)[0]:
            print("ERROR: Q and b dimensions disagree!")

    def __call__(self, x):
        return 0.5*(x.T@self.Q@x)+self.b@x

    def grad(self, x):
        return 0.5*(self.Q + self.Q.T)@x + self.b

    def hessian(self, x):
        return 0.5*(self.Q + self.Q.T)


class NewtonOptimizer:
    def __init__(self, func, alpha, init):
        self.f = func
        self.alpha = alpha
        self.x_0 = init
        self.x = self.x_0 # x = current x
        self.x_prev = 0 # x_prev = previous x

    def step(self):
        g = self.f.grad(self.x)
        H = self.f.hessian(self.x)
        try: # if H is not invertible, perform regular gradient descent (H_inv = I)
            H_inv = np.linalg.inv(self.f.hessian(self.x))
        except:
            H_inv = np.eye(self.x_0.shape[0])

        self.x = self.x - self.alpha * H_inv @ g
        return self.x, g, H

    def optimize(self, threshold, max_iters):
        for i in range(int(max_iters)):
            self.x_prev = self.x
            self.x, g, H = self.step()
            if np.linalg.norm(self.x - self.x_prev) < threshold:
                break
        return self.f(self.x), self.x, i+1


class ConjGradOptimizer:
    def __init__(self, func, init):
        self.f = func
        self.x_0 = init
        self.x = self.x_0  # x = current x
        self.x_prev = 0  # x_prev = previous x
        self.Q = self.f.hessian(self.x_0) # assume f is quadratic with symmetric Q matrix
        self.g = self.f.grad(self.x_0) # initial g0
        self.g_prev = 0 # previous g
        self.beta = 0
        self.alpha = 0
        self.d = -self.g # current direction, initial direction d0 = -g0
        self.d_prev = 0 # previous direction


    def step(self):
        # at this point we have the current x_k, d_k, g_k
        # compute alpha - we have d_k, g_k
        self.update_alpha()
        # compute x_k+1 - we have x_k, d_k, alpha_k
        self.update_x()
        # compute g_k+1 - we have x_k+1
        self.update_grad()
        # compute d_k+1 - we have g_k+1 and g_k
        self.update_dir(self.g_prev)
        # compute alpha_k+1 - to return it as output
        alpha_next = -self.g @ self.d / (self.d.T @ self.Q @ self.d)
        return self.x, self.g, self.d, alpha_next


    def optimize(self):
        # Conjugate Gradients converges in n steps ==> stop after n steps
        n = self.x_0.shape[0]
        for i in range(n):
            self.step()
        return self.f(self.x), self.x, i+1


    def update_grad(self):
        self.g_prev = self.g
        self.g = self.f.grad(self.x)
        return self.g, self.g_prev

    def update_dir(self, prev_grad):
        self.d_prev = self.d
        self.beta = np.square(np.linalg.norm(self.g)) / np.square(np.linalg.norm(prev_grad))
        self.d = -self.g + self.beta * self.d
        return self.d

    def update_alpha(self):
        self.alpha = -self.g @ self.d / (self.d.T @ self.Q @ self.d)
        return self.alpha

    def update_x(self):
        self.x_prev = self.x
        self.x = self.x + self.alpha * self.d
        return self.x


# Part II
class FastRoute:
    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):
        self.x_s = start_x
        self.y_s = start_y
        self.x_f = finish_x
        self.y_f = finish_y
        self.v = velocities
        self.dx = 0 # delta x or dx = [x1-x_s, x2-x1, x3-x2, ... , x_f - x_n-1] size = (N-1) X 1
        self.n = self.v.shape[0]
        self.d = (self.y_f - self.y_s) / self.n # d = y_i - y_i-1 (the spacing between the y_i's)

    def __call__(self, x):
        # compute dx = [x1-x_s, x2-x1, x3-x2, ... , x_f - x_n-1]
        self.dx = np.concatenate((x, np.array([self.x_f]))) - np.concatenate((np.array([self.x_s]), x))
        total_time = np.sum(np.sqrt(np.square(self.dx) + np.square(self.d)) / self.v)
        return total_time

    def grad(self, x):
        # compute dx = [x1-x_s, x2-x1, x3-x2, ... , x_f - x_n-1]
        self.dx = np.concatenate((x, np.array([self.x_f]))) - np.concatenate((np.array([self.x_s]), x))
        g = self.dx[0:-1] / (self.v[0:-1] * (np.sqrt(np.square(self.dx[0:-1]) + np.square(self.d)))) \
            - self.dx[1:] / (self.v[1:] * (np.sqrt(np.square(self.dx[1:]) + np.square(self.d))))
        return g


    def hessian(self, x):
        # compute dx = [x1-x_s, x2-x1, x3-x2, ... , x_f - x_n-1]
        self.dx = np.concatenate((x, np.array([self.x_f]))) - np.concatenate((np.array([self.x_s]), x))

        H_x = np.zeros([self.n-1, self.n-1])
        # compute the diagonal of the hessian size = (N-1) X 1
        diag = np.square(self.d) / (self.v[0:-1] * np.power(np.square(self.dx[0:-1]) + np.square(self.d), 3/2)) \
               + np.square(self.d) / (self.v[1:] * np.power(np.square(self.dx[1:]) + np.square(self.d), 3/2))

        # compute the sub-diagonals of the hessian above and below the main diagonal :
        # main diagonal == indexes [(1,1), (2,2), (3,3), ... , (n-1,n-1)]  ==> size = n-1
        # upper sub-diagonal == indexes [(1,2), (2,3), (3,4), ... , (n-2,n-1)] ==> size = n-2
        # lower sub-diagonal == indexes [(2,1), (3,2), (4,3), ... , (n-1,n-2)] ==> size = n-2
        # hessian is symmetric ==> upper sub-diagonal = lower sub-diagonal
        sub_diag = -1 * np.square(self.d) / (self.v[1:-1] * np.power(np.square(self.dx[1:-1]) + np.square(self.d), 3/2))

        # plug diagonal and sub-diagonals into H_x
        H_x[np.arange(0, self.n-1, 1), np.arange(0, self.n-1, 1)] = diag
        H_x[np.arange(0, self.n-2, 1), np.arange(0, self.n-2, 1) + 1 ] = sub_diag
        H_x[np.arange(0, self.n-2, 1) + 1, np.arange(0, self.n-2, 1)] = sub_diag
        # print(H_x)
        return H_x


def find_fast_route(objective, init, alpha=1, threshold=1e-3, max_iters=1e3):
    newton_opt = NewtonOptimizer(objective, alpha, init)
    return newton_opt.optimize(threshold, max_iters)

def find_alpha(start_x, start_y, finish_x, finish_y, num_layers):
    pass



