import numpy as np
from numpy.linalg import multi_dot

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
        return 0.5*multi_dot([x.T, self.Q, x])+multi_dot([self.b, x])

    def grad(self, x):
        return 0.5*multi_dot([(self.Q + self.Q.T), x]) + self.b

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
        try: # if H is not invertible, perform regular gradient descent (H_inv = I)
            H_inv = np.linalg.inv(self.f.hessian(self.x))
        except:
            H_inv = np.eye(self.x_0.shape[0])

        self.x = self.x - self.alpha * multi_dot([H_inv, g])
        # print("<-g,d> = ", g.T@H_inv@g)
        return self.x, g, H_inv

    def optimize(self, threshold, max_iters):
        for i in range(int(max_iters)):
            self.x_prev = self.x
            self.x, g, H_inv = self.step()
            # if (self.f(self.x_prev) > self.f(self.x)):
            #     print("ERROR: f(x_k+1) > f(x_k)")

            # print(np.sqrt(np.linalg.norm(self.x - self.x_prev)))
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
        self.d = -self.g # current direction
        self.d_prev = 0 # previous direction


    def step(self):
        # at this point we have the current x_k, d_k, g_k
        # compute alpha - we have d_k, g_k
        self.update_alpha()
        # print("alpha_i=", self.alpha)
        # compute x_k+1 - we have x_k, d_k, alpha_k
        self.update_x()
        # print("x_i=", self.x)
        # compute g_k+1 - we have x_k+1
        self.update_grad()
        # compute d_k+1 - we have g_k+1 and g_k
        self.update_dir(self.g_prev)
        # print("d_i=",self.d)
        return self.x, self.g, self.d, self.alpha # TODO: check if need to return alpha_k or alpha_k+1

    def optimize(self):
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
        self.alpha = multi_dot([-self.g, self.d]) / multi_dot([self.d.T, self.Q, self.d])
        # print("-gd = ", -self.g @ self.d)
        # print("dQd=", self.d.T @ self.Q @ self.d)
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
        # print("test", self.dx[0:-1] / (self.v[0:-1] * (np.sqrt(np.square(self.dx[0:-1]) + np.square(self.d)))))
        # print("test2", self.dx[1:] / (self.v[1:] * (np.sqrt(np.square(self.dx[1:]) + np.square(self.d)))))
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
    alpha = min([abs(finish_x - start_x)/num_layers, abs(finish_y - start_y)/num_layers, 1])
    return alpha

# TODO: consider changing te @ operator to np.multi_dot()
# TODO: Bonus question


# **************************************************************
# Test
# **************************************************************
#
# A = np.array([(10, 2, 7), (2, 3, 9), (7, 9, 11)])
# b = np.array([8, 21, 47])
# f = QuadraticFunction(A, b)
# x_0 = np.array([7, 23, -6])
#
# print(f.Q)
# print(f.b)
# print(f.grad(x_0))
# print(f.hessian(x_0))
#
# alpha = 0.8
# n = NewtonOptimizer(f, alpha, x_0)
# fmin, minimizer, num_iter = n.optimize(0.001, 30)
# print(fmin, minimizer, num_iter)
#
# print(np.linalg.inv(f.hessian(x_0)) @ -b)
#
# cg = ConjGradOptimizer(f, x_0)
# print(cg.optimize())
#

# test fast route hessian
# start_x = 1.0
# start_y = 1.0
# finish_x = 10.0
# finish_y = 12.0
# velocities = np.array([1.0, 2.0, 3.0, 4.0])
# x = np.array([4.0, 5.0, 6.0])
# FR2 = FastRoute(start_x, start_y, finish_x, finish_y, velocities)
# print(FR2.hessian(x))
# print(FR2([1.60876148 ,2.92703483, 5.26925231]))
# print(FR2.grad(x), FR2.d, FR2.dx)



#
# # test fast route newton optimizer
# start_x = 3
# start_y = 2
# finish_x = 10
# finish_y = 8
# velocities = np.array([1, 5, 2, 3, 1])
# x_fast_route = np.array([4, 5, 6, 7])
#
# alpha_fast_route = 0.3
# TH_fast_route = 1e-2
# max_iters_fast_route = 200
#
# FR = FastRoute(start_x,start_y, finish_x, finish_y, velocities)
# time, min_x, num_iter = find_fast_route(FR,x_fast_route,alpha_fast_route, TH_fast_route, max_iters_fast_route)
# print("time :", time)
# print("min x:", min_x)
# print("num of iter:", num_iter)
# # print(FR2([1.60876148 ,2.92703483, 5.26925231]),FR2.grad([1.60876148 ,2.92703483, 5.26925231]) , FR2(min_x), FR2.grad(min_x))
#


# start_x = 0.4
# start_y = 0.3
# finish_x = 1
# finish_y = 1
# velocities = np.array([1.9, 0.6, 0.53, 2.3, 2.5, 1.2, 0.6])
# x_fast_route_input = np.array([1.2, 2.3, 3.4, 4.5, 4.3, 3.2])
# alpha_fast_route = 0.1
# TH_fast_route = 0.001
# max_iters_fast_route = 200
#
# start_x = 1.0
# start_y = 1.0
# finish_x = 1.2
# finish_y = 2
# velocities = np.array([1.0, 2.0, 3.0, 4.0])
# x_fast_route_input = np.array([4.0, 5.0, 6.0])
# alpha_fast_route = 0.1
# TH_fast_route = 1e-2
# max_iters_fast_route = 200
#
# alpha = find_alpha(start_x, start_y, finish_x, finish_y, x_fast_route_input.shape[0])
# print(alpha)
#
# FR = FastRoute(start_x,start_y, finish_x, finish_y, velocities)
# time, min_x, num_iter = find_fast_route(FR,x_fast_route_input,alpha, TH_fast_route, max_iters_fast_route)
# print("time :", time)
# print("min x:", min_x)
# print("num of iter:", num_iter)



# Q = np.array([[20933.82611369, 15467.25616221, 23102.57922164, 9238.60189952, 28382.55916558, 21400.15594295, 15157.10288336, 9332.09283481, 14251.75049803], [15467.25616221, 21201.93023444, 21715.03306506, 16844.79058916, 35389.31199927, 25821.42128826, 17981.98967772, 20712.2489108 , 22479.21068503], [23102.57922164, 21715.03306506, 38036.53680995, 23905.45305855, 46357.68934824, 29835.76271053, 33514.6643661 , 26352.6868082 , 29307.09934745], [ 9238.60189952, 16844.79058916, 23905.45305855, 25656.89095958, 35033.63508484, 22041.34416936, 25138.08642151, 28498.35340437, 22307.38013115], [28382.55916558, 35389.31199927, 46357.68934824, 35033.63508484, 68306.7057647 , 46585.32738368, 43224.0882427 , 41425.89827638, 42912.40606859], [21400.15594295, 25821.42128826, 29835.76271053, 22041.34416936, 46585.32738368, 36763.35008369, 24968.04289003, 26971.81800367, 25971.48058108], [15157.10288336, 17981.98967772, 33514.6643661 , 25138.08642151, 43224.0882427 , 24968.04289003, 35794.74703827, 29066.72842427, 29919.6255736 ], [ 9332.09283481, 20712.2489108 , 26352.6868082 , 28498.35340437, 41425.89827638, 26971.81800367, 29066.72842427, 34136.90066848, 28593.19179483], [14251.75049803, 22479.21068503, 29307.09934745, 22307.38013115, 42912.40606859, 25971.48058108, 29919.6255736 , 28593.19179483, 35417.87008957]])
# b = np.array([72.37100855, 11.82342115, 82.02992352, 61.02102051, 42.03369216, 92.24298478, 19.54181557, 80.486503 , 73.03177666])
# init = np.array([90.88335732, 64.41635324, 55.69806762, 3.97568517, 0.70285537, 18.34089773, 87.89181481, 36.79256356, 51.42055209])
# alpha = 1
# thresh = 0.01
# max_iters = 1000
#
# print(Q.shape, b.shape)
# print("Eigenvalues:", np.linalg.eig(Q))
#
# f = QuadraticFunction(Q,b)
# n = NewtonOptimizer(f,alpha,init)
# print(n.optimize(thresh,max_iters))
#
#
# con_g = ConjGradOptimizer(f,init)
# print(con_g.optimize())
#
#
# x_i= np.array([ -4.07281743,  28.31596288,  -4.99713975,  13.05031645 ,-27.94186412,
#   11.74804278 , 24.03437954, -18.45800829,  -0.5817149 ])
# print(f.grad(x_i))


#
# alpha = 0.1
# TH = 1e-3
# max_iters_newton = 10000
# x = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
# b = np.array([1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,1.0])
# Q= np.array([[3.2823  ,  1.0731  ,  1.6142  ,  3.0577  ,  2.0152  ,  1.2074  ,  2.2415   , 3.1684  ,  2.3762],
#     [1.0731  ,  1.3455  ,  0.9693  ,  1.1452  ,  1.1201  ,  0.6839  ,  1.1125  ,  1.1862  ,  1.1224],
#     [1.6142  ,  0.9693  ,  1.7823  ,  2.0703  ,  1.0580  ,  1.0453  ,  1.9129  ,  1.9185  ,  1.5015],
#     [3.0577  ,  1.1452  ,  2.0703  ,  4.0244  ,  2.1055  ,  1.8273  ,  2.8407  ,  3.4036  ,  2.1863],
#     [2.0152  ,  1.1201  ,  1.0580  ,  2.1055  ,  1.8685  ,  1.0564  ,  1.4065  ,  2.0696  ,  1.5821],
#     [1.2074  ,  0.6839  ,  1.0453  ,  1.8273  ,  1.0564  ,  1.1017  ,  1.2917  ,  1.3977  ,  1.0301],
#     [2.2415  ,  1.1125  ,  1.9129  ,  2.8407  ,  1.4065  ,  1.2917  ,  3.0574  ,  2.4677  ,  1.7114],
#     [3.1684  ,  1.1862  ,  1.9185  ,  3.4036  ,  2.0696  ,  1.3977  ,  2.4677  ,  3.4320  ,  2.3904],
#     [2.3762  ,  1.1224  ,  1.5015  ,  2.1863  ,  1.5821  ,  1.0301  ,  1.7114  ,  2.3904  ,  2.0324]])
#
# x = np.array([2.0])
# Q = np.array([[4.0]])
# b = np.array([-1.0])
# alpha = 0.1
# TH = 1e-5
# max_iters_newton = 150

# x = np.array([1.0,2.0, 3.0])
# Q = np.array([[11.0,11.0,7.0],[11.0, 17.0, 11.0],[7.0,11.0,11.0]])
# b = np.array([3.0, 2.0, 1.0])
# alpha = 0.1
# TH = 1e-5
# max_iters_newton = 150
# #
# x = np.array([1.0,1.0,1.0,1.0])
# b = np.array([1.0,2.0,3.0,4.0])
# Q = np.array([[0.776291,0.295617,0.159137,0.057850], [0.295617,0.790239,0.082293,0.164077], [0.159137,0.082293,0.953726,0.069922],[0.057850,0.164077,0.069922,0.927096]])
# alpha = 1
# TH = 1e-3
# max_iters_newton = 1e3

#
# # print(max(np.linalg.eigvals(Q))/min(np.linalg.eigvals(Q)))
# f = QuadraticFunction(Q, b)
# n = NewtonOptimizer(f,alpha,x)
# print(n.optimize(TH,max_iters_newton))
# con_g = ConjGradOptimizer(f,x)
# print(con_g.optimize())
#


# **************************************************************
# Test
# **************************************************************
#
# A = np.array([(10, 2, 7), (2, 3, 9), (7, 9, 11)])
# b = np.array([8, 21, 47])
# f = QuadraticFunction(A, b)
# x_0 = np.array([7, 23, -6])
#
# print(f.Q)
# print(f.b)
# print(f.grad(x_0))
# print(f.hessian(x_0))
#
# alpha = 0.8
# n = NewtonOptimizer(f, alpha, x_0)
# fmin, minimizer, num_iter = n.optimize(0.001, 30)
# print(fmin, minimizer, num_iter)
#
# print(np.linalg.inv(f.hessian(x_0)) @ -b)
#
# cg = ConjGradOptimizer(f, x_0)
# print(cg.optimize())
#

# test fast route hessian
# start_x = 1.0
# start_y = 1.0
# finish_x = 10.0
# finish_y = 12.0
# velocities = np.array([1.0, 2.0, 3.0, 4.0])
# x = np.array([4.0, 5.0, 6.0])
# FR2 = FastRoute(start_x, start_y, finish_x, finish_y, velocities)
# print(FR2.hessian(x))
# print(FR2([1.60876148 ,2.92703483, 5.26925231]))
# print(FR2.grad(x), FR2.d, FR2.dx)



#
# # test fast route newton optimizer
# start_x = 3
# start_y = 2
# finish_x = 10
# finish_y = 8
# velocities = np.array([1, 5, 2, 3, 1])
# x_fast_route = np.array([4, 5, 6, 7])
#
# alpha_fast_route = 0.3
# TH_fast_route = 1e-2
# max_iters_fast_route = 200
#
# FR = FastRoute(start_x,start_y, finish_x, finish_y, velocities)
# time, min_x, num_iter = find_fast_route(FR,x_fast_route,alpha_fast_route, TH_fast_route, max_iters_fast_route)
# print("time :", time)
# print("min x:", min_x)
# print("num of iter:", num_iter)
# # print(FR2([1.60876148 ,2.92703483, 5.26925231]),FR2.grad([1.60876148 ,2.92703483, 5.26925231]) , FR2(min_x), FR2.grad(min_x))
#


# start_x = 0.4
# start_y = 0.3
# finish_x = 1
# finish_y = 1
# velocities = np.array([1.9, 0.6, 0.53, 2.3, 2.5, 1.2, 0.6])
# x_fast_route_input = np.array([1.2, 2.3, 3.4, 4.5, 4.3, 3.2])
# alpha_fast_route = 0.1
# TH_fast_route = 0.001
# max_iters_fast_route = 200
#
# start_x = 1.0
# start_y = 1.0
# finish_x = 1.2
# finish_y = 2
# velocities = np.array([1.0, 2.0, 3.0, 4.0])
# x_fast_route_input = np.array([4.0, 5.0, 6.0])
# alpha_fast_route = 0.1
# TH_fast_route = 1e-2
# max_iters_fast_route = 200
#
# alpha = find_alpha(start_x, start_y, finish_x, finish_y, x_fast_route_input.shape[0])
# print(alpha)
#
# FR = FastRoute(start_x,start_y, finish_x, finish_y, velocities)
# time, min_x, num_iter = find_fast_route(FR,x_fast_route_input,alpha, TH_fast_route, max_iters_fast_route)
# print("time :", time)
# print("min x:", min_x)
# print("num of iter:", num_iter)



# Q = np.array([[20933.82611369, 15467.25616221, 23102.57922164, 9238.60189952, 28382.55916558, 21400.15594295, 15157.10288336, 9332.09283481, 14251.75049803], [15467.25616221, 21201.93023444, 21715.03306506, 16844.79058916, 35389.31199927, 25821.42128826, 17981.98967772, 20712.2489108 , 22479.21068503], [23102.57922164, 21715.03306506, 38036.53680995, 23905.45305855, 46357.68934824, 29835.76271053, 33514.6643661 , 26352.6868082 , 29307.09934745], [ 9238.60189952, 16844.79058916, 23905.45305855, 25656.89095958, 35033.63508484, 22041.34416936, 25138.08642151, 28498.35340437, 22307.38013115], [28382.55916558, 35389.31199927, 46357.68934824, 35033.63508484, 68306.7057647 , 46585.32738368, 43224.0882427 , 41425.89827638, 42912.40606859], [21400.15594295, 25821.42128826, 29835.76271053, 22041.34416936, 46585.32738368, 36763.35008369, 24968.04289003, 26971.81800367, 25971.48058108], [15157.10288336, 17981.98967772, 33514.6643661 , 25138.08642151, 43224.0882427 , 24968.04289003, 35794.74703827, 29066.72842427, 29919.6255736 ], [ 9332.09283481, 20712.2489108 , 26352.6868082 , 28498.35340437, 41425.89827638, 26971.81800367, 29066.72842427, 34136.90066848, 28593.19179483], [14251.75049803, 22479.21068503, 29307.09934745, 22307.38013115, 42912.40606859, 25971.48058108, 29919.6255736 , 28593.19179483, 35417.87008957]])
# b = np.array([72.37100855, 11.82342115, 82.02992352, 61.02102051, 42.03369216, 92.24298478, 19.54181557, 80.486503 , 73.03177666])
# init = np.array([90.88335732, 64.41635324, 55.69806762, 3.97568517, 0.70285537, 18.34089773, 87.89181481, 36.79256356, 51.42055209])
# alpha = 1
# thresh = 0.01
# max_iters = 1000
#
# print(Q.shape, b.shape)
# print("Eigenvalues:", np.linalg.eig(Q))
#
# f = QuadraticFunction(Q,b)
# n = NewtonOptimizer(f,alpha,init)
# print(n.optimize(thresh,max_iters))
#
#
# con_g = ConjGradOptimizer(f,init)
# print(con_g.optimize())
#
#
# x_i= np.array([ -4.07281743,  28.31596288,  -4.99713975,  13.05031645 ,-27.94186412,
#   11.74804278 , 24.03437954, -18.45800829,  -0.5817149 ])
# print(f.grad(x_i))


# alpha = 0.1
# TH = 1e-3
# max_iters_newton = 10000
# x = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
# b = np.array([1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0,1.0])
# Q= np.array([[3.2823  ,  1.0731  ,  1.6142  ,  3.0577  ,  2.0152  ,  1.2074  ,  2.2415   , 3.1684  ,  2.3762],
#     [1.0731  ,  1.3455  ,  0.9693  ,  1.1452  ,  1.1201  ,  0.6839  ,  1.1125  ,  1.1862  ,  1.1224],
#     [1.6142  ,  0.9693  ,  1.7823  ,  2.0703  ,  1.0580  ,  1.0453  ,  1.9129  ,  1.9185  ,  1.5015],
#     [3.0577  ,  1.1452  ,  2.0703  ,  4.0244  ,  2.1055  ,  1.8273  ,  2.8407  ,  3.4036  ,  2.1863],
#     [2.0152  ,  1.1201  ,  1.0580  ,  2.1055  ,  1.8685  ,  1.0564  ,  1.4065  ,  2.0696  ,  1.5821],
#     [1.2074  ,  0.6839  ,  1.0453  ,  1.8273  ,  1.0564  ,  1.1017  ,  1.2917  ,  1.3977  ,  1.0301],
#     [2.2415  ,  1.1125  ,  1.9129  ,  2.8407  ,  1.4065  ,  1.2917  ,  3.0574  ,  2.4677  ,  1.7114],
#     [3.1684  ,  1.1862  ,  1.9185  ,  3.4036  ,  2.0696  ,  1.3977  ,  2.4677  ,  3.4320  ,  2.3904],
#     [2.3762  ,  1.1224  ,  1.5015  ,  2.1863  ,  1.5821  ,  1.0301  ,  1.7114  ,  2.3904  ,  2.0324]])

# x = np.array([2.0, 1.0])
# Q = np.array([[4.0, 1.0],[1.0, 3.0]])
# b = np.array([-1.0, -2.0])
# alpha = 0.1
# TH = 1e-5
# max_iters_newton = 150

# x = np.array([1.0,2.0, 3.0])
# Q = np.array([[11.0,11.0,7.0],[11.0, 17.0, 11.0],[7.0,11.0,11.0]])
# b = np.array([3.0, 2.0, 1.0])
# alpha = 0.1
# TH = 1e-5
# max_iters_newton = 150

# x = np.array([1.0,1.0,1.0,1.0])
# b = np.array([1.0,2.0,3.0,4.0])
# Q = np.array([[0.776291,0.295617,0.159137,0.057850], [0.295617,0.790239,0.082293,0.164077], [0.159137,0.082293,0.953726,0.069922],[0.057850,0.164077,0.069922,0.927096]])
# alpha = 1
# TH = 1e-3
# max_iters_newton = 1e3
#
#
# print(max(np.linalg.eigvals(Q))/min(np.linalg.eigvals(Q)))
# f = QuadraticFunction(Q, b)
# n = NewtonOptimizer(f,alpha,x)
# print(n.optimize(TH,max_iters_newton))
# con_g = ConjGradOptimizer(f,x)
# print(con_g.optimize())