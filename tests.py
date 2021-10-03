#################################
### Tests By Yehonatan Kohavi ###
#################################

import numpy as np
from main import *

ERR = 1
OK = 0


###################################################################
###################################################################
#####################         utils:        #######################
###################################################################

def printFailMassage(test_name, curr_res, gold_res):
    str = '{0} has failed!\ncurrent results:\n{1}\ngold results:\n{2}'.format(test_name, curr_res, gold_res)
    printMsg(str)


def RunSingleTestAndCompare(test):
    tolerance = 1e-6 #1e-6
    test_handle = test[0]
    gold_results = test[1]
    current_results = test_handle()
    bug = False
    if len(gold_results) != len(current_results):
        bug = True
    else:
        for i in range(len(current_results)):
            if np.allclose(current_results[i], gold_results[i], 0, tolerance) == False:
                bug = True
                break
    if bug:
        printFailMassage(test_handle.__name__, current_results, gold_results)
        return ERR
    return OK


def printMsg(str):
    print(str)
    print('-' * 50)


def UTL_testQuadratic(Q, b, init):
    f = QuadraticFunction(Q, b)
    val = f(init)
    grad = f.grad(init)
    hessian = f.hessian(init)
    return val, grad, hessian


def UTL_testNewton(Q, b, alpha, init, threshold, max_iters):
    f = QuadraticFunction(Q, b)
    newtonOpt = NewtonOptimizer(f, alpha, init)
    fmin, minimizer, num_iters = newtonOpt.optimize(threshold, max_iters)
    return fmin, minimizer, num_iters


def UTL_testConjGrad(Q, b, init):
    f = QuadraticFunction(Q, b)
    conjOpt = ConjGradOptimizer(f, init)
    fmin, minimizer, num_iters = conjOpt.optimize()
    return fmin, minimizer, num_iters


def UTL_testFastRoute(start_x, start_y, finish_x, finish_y, velocities, init, alpha, threshold, max_iters):
    fast_rout = FastRoute(start_x, start_y, finish_x, finish_y, velocities)
    t = fast_rout(init)
    grads = fast_rout.grad(init)
    hessian = fast_rout.hessian(init)
    fmin, minimizer, num_iters = find_fast_route(fast_rout, init, alpha, threshold, max_iters)
    return t, grads, hessian, fmin, minimizer, num_iters


def UTL_testQuadNewtonAndConj(Q, b, alpha, init, threshold, max_iters):
    quad_res = UTL_testQuadratic(Q, b, init)
    newton_res = UTL_testNewton(Q, b, alpha, init, threshold, max_iters)
    conj_res = UTL_testConjGrad(Q, b, init)
    return quad_res + newton_res + conj_res


###################################################################
###################################################################
###################         The Tests:        #####################
###################################################################
def test1():
    Q = np.array([[5, 2], [2, 3]])
    b = np.array([2, -4])
    init = np.array([-1, 1])
    alpha = 2e-2
    threshold = 1e-9 #1e-9
    max_iters = 1e3
    return UTL_testQuadNewtonAndConj(Q, b, alpha, init, threshold, max_iters)


def test2():
    """Check Assymetryc Q"""
    Q = np.array([[5, 2], [4, 3]])
    b = np.array([2, -4])
    init = np.array([-1, 1])
    return UTL_testQuadratic(Q, b, init)


def test3():
    """One dimensional"""
    Q = np.array([[2]])
    b = np.array([-4])
    alpha = 2e-2
    threshold = 1e-9 #1e-9
    max_iters = 1e3
    init = np.array([1])
    return UTL_testQuadNewtonAndConj(Q, b, alpha, init, threshold, max_iters)


def test5():
    velocities = np.array([2, 3, 5])
    start_x = 1
    start_y = 2
    finish_x = 6
    finish_y = 8
    init = np.array([3, 4])
    alpha = 1
    threshold = 1e-3
    max_iters = 1e3
    return UTL_testFastRoute(start_x, start_y, finish_x, finish_y, velocities, init, alpha, threshold, max_iters)


def test4():
    """One dimensional"""
    Q = np.array([2])
    b = np.array([-4])
    alpha = 2e-2
    threshold = 1e-9
    max_iters = 1e3
    init = np.array([2])
    return UTL_testQuadNewtonAndConj(Q, b, alpha, init, threshold, max_iters)


def test6():
    Q = np.array([[4, 1, 3], [1, 3, 1], [1, 1, 1]])
    b = np.array([-4, -4, 2])
    init = np.array([0, 0, 0])
    alpha = 0.5
    threshold = 0.05
    max_iters = 50
    return UTL_testQuadNewtonAndConj(Q, b, alpha, init, threshold, max_iters)


def test7():
    velocities = np.array([3, 23])
    start_x = 1
    start_y = 2
    finish_x = 6
    finish_y = 8
    init = np.array([4])
    alpha = 1
    threshold = 1e-3
    max_iters = 1e3
    return UTL_testFastRoute(start_x, start_y, finish_x, finish_y, velocities, init, alpha, threshold, max_iters)


def test8():
    start_x = 0.4
    start_y = 0.3
    finish_x = 11.2
    finish_y = 12.3
    velocities = np.array([1.9, 0.6, 0.53, 2.3, 2.5, 1.2, 0.6])
    init = np.array([1.2, 2.3, 3.4, 4.5, 4.3, 3.2])
    alpha = 0.1
    threshold = 1e-2
    max_iters = 200
    return UTL_testFastRoute(start_x, start_y, finish_x, finish_y, velocities, init, alpha, threshold, max_iters)


def test9():
    Q = np.array([[20933.82611369, 15467.25616221, 23102.57922164, 9238.60189952, 28382.55916558, 21400.15594295,
                   15157.10288336, 9332.09283481, 14251.75049803],
                  [15467.25616221, 21201.93023444, 21715.03306506, 16844.79058916, 35389.31199927, 25821.42128826,
                   17981.98967772, 20712.2489108, 22479.21068503],
                  [23102.57922164, 21715.03306506, 38036.53680995, 23905.45305855, 46357.68934824, 29835.76271053,
                   33514.6643661, 26352.6868082, 29307.09934745],
                  [9238.60189952, 16844.79058916, 23905.45305855, 25656.89095958, 35033.63508484, 22041.34416936,
                   25138.08642151, 28498.35340437, 22307.38013115],
                  [28382.55916558, 35389.31199927, 46357.68934824, 35033.63508484, 68306.7057647, 46585.32738368,
                   43224.0882427, 41425.89827638, 42912.40606859],
                  [21400.15594295, 25821.42128826, 29835.76271053, 22041.34416936, 46585.32738368, 36763.35008369,
                   24968.04289003, 26971.81800367, 25971.48058108],
                  [15157.10288336, 17981.98967772, 33514.6643661, 25138.08642151, 43224.0882427, 24968.04289003,
                   35794.74703827, 29066.72842427, 29919.6255736],
                  [9332.09283481, 20712.2489108, 26352.6868082, 28498.35340437, 41425.89827638, 26971.81800367,
                   29066.72842427, 34136.90066848, 28593.19179483],
                  [14251.75049803, 22479.21068503, 29307.09934745, 22307.38013115, 42912.40606859, 25971.48058108,
                   29919.6255736, 28593.19179483, 35417.87008957]])
    b = np.array([72.37100855, 11.82342115, 82.02992352, 61.02102051, 42.03369216, 92.24298478, 19.54181557, 80.486503,
                  73.03177666])
    init = np.array(
        [90.88335732, 64.41635324, 55.69806762, 3.97568517, 0.70285537, 18.34089773, 87.89181481, 36.79256356,
         51.42055209])
    alpha = 1
    threshold = 1e-8
    max_iters = 1e3
    return UTL_testQuadNewtonAndConj(Q, b, alpha, init, threshold, max_iters)


tests_list = [
    (test1, (-4.0, np.array([-1., -3.]), np.array([[5., 2.],
                                                   [2., 3.]]), -5.636363636363633, np.array([-1.27272726, 2.18181813]),
             843, -5.636363636363635, np.array([-1.27272727, 2.18181818]), 2)),
    (test2, (np.array([-5.]), np.array([0., -4.]), np.array([[5., 3.], [3., 3.]]))),
    (test3, (
        np.array([-3.]), np.array([-2.]), np.array([2.]), np.array([-4]), np.array([1.99999995]), 834, np.array([-4]),
        np.array([2.]), 1)),
    (test4, (
        np.array([-4.]), np.array([0]), np.array([2.]), np.array([-4]), np.array([2.]), 1, np.array([-4]),
        np.array([2.]),
        1)),
    (test5, (2.725254979822263, np.array([0.20448219, 0.00764984]), np.array([[0.20764531, -0.11925696],
                                                                              [-0.11925696, 0.1546123]]),
             2.5747024678138146, np.array([1.71491927, 2.88478685]), 4)),
    (test6, (0.0, np.array([-4., -4., 2.]), np.array([[4., 1., 2.],
                                                      [1., 3., 1.],
                                                      [2., 1., 1.]]), 109.99997377395613,
             np.array([-21.98925781, -7.99609375, 49.97558594]), 11, 109.99999999999991, np.array([-22., -8., 50.]),
             3)),
    (test7, (
        1.5709766613063119, np.array([0.21158486]), np.array([[0.04763204]]), 1.247361143386305,
        np.array([[1.33120087]]),
        5)),
    (test8, (25.14289710693424, np.array([-0.67750944, -0.11887859, 0.78415565, 0.28115595, 0.40368809,
                                          -2.07971106]), np.array([[0.80806969, -0.57960918, 0., 0., 0.,
                                                                    0.],
                                                                   [-0.57960918, 1.23577052, -0.65616134, 0., 0.,
                                                                    0.],
                                                                   [0., -0.65616134, 0.80736374, -0.1512024, 0.,
                                                                    0.],
                                                                   [0., 0., -0.1512024, 0.37985162, -0.22864923,
                                                                    0.],
                                                                   [0., 0., 0., -0.22864923, 0.51845382,
                                                                    -0.28980459],
                                                                   [0., 0., 0., 0., -0.28980459,
                                                                    0.29874791]]), 15.313506871053335,
             np.array([2.12320499, 2.52449449, 2.8779527, 5.73931661, 9.95096549,
                       10.7933222]), 44)),
    (test9, (1941903236.6172836, np.array([7043268.4816092, 8044815.46372093, 11714159.61767682,
                                           8192166.35651509, 16012725.7684382, 10587088.22297466,
                                           10744830.2975601, 9568307.24734481, 10473799.59706791]),
             np.array([[20933.82611369, 15467.25616221, 23102.57922164, 9238.60189952,
                        28382.55916558, 21400.15594295, 15157.10288336, 9332.09283481,
                        14251.75049803],
                       [15467.25616221, 21201.93023444, 21715.03306506, 16844.79058916,
                        35389.31199927, 25821.42128826, 17981.98967772, 20712.2489108,
                        22479.21068503],
                       [23102.57922164, 21715.03306506, 38036.53680995, 23905.45305855,
                        46357.68934824, 29835.76271053, 33514.6643661, 26352.6868082,
                        29307.09934745],
                       [9238.60189952, 16844.79058916, 23905.45305855, 25656.89095958,
                        35033.63508484, 22041.34416936, 25138.08642151, 28498.35340437,
                        22307.38013115],
                       [28382.55916558, 35389.31199927, 46357.68934824, 35033.63508484,
                        68306.7057647, 46585.32738368, 43224.0882427, 41425.89827638,
                        42912.40606859],
                       [21400.15594295, 25821.42128826, 29835.76271053, 22041.34416936,
                        46585.32738368, 36763.35008369, 24968.04289003, 26971.81800367,
                        25971.48058108],
                       [15157.10288336, 17981.98967772, 33514.6643661, 25138.08642151,
                        43224.0882427, 24968.04289003, 35794.74703827, 29066.72842427,
                        29919.6255736],
                       [9332.09283481, 20712.2489108, 26352.6868082, 28498.35340437,
                        41425.89827638, 26971.81800367, 29066.72842427, 34136.90066848,
                        28593.19179483],
                       [14251.75049803, 22479.21068503, 29307.09934745, 22307.38013115,
                        42912.40606859, 25971.48058108, 29919.6255736, 28593.19179483,
                        35417.87008957]]), -658.8965514046125,
             np.array([-7.31971714, 52.09035071, -9.31712887, 24.04439716,
                       -51.55933331, 21.65012964, 44.30344942, -33.8891791,
                       -1.02511524]), 3, -522.9216217459009,
             np.array([-4.07281741, 28.31596289, -4.99713974, 13.05031644,
                       -27.9418641, 11.7480428, 24.03437953, -18.4580083,
                       -0.5817149]), 9)),

]

failed_tests_list = []

def RunAllTests():
    num_failed = 0
    num_tests = len(tests_list)
    printMsg("Running all tests")
    for i in range(num_tests):
        try:
            ret = RunSingleTestAndCompare(tests_list[i])
            if ret == 1:
                failed_tests_list.append(tests_list[i][0].__name__)
                num_failed += 1
        except Exception as e:
            printMsg("Bug in test number{0}: {1}".format(i + 1, e))
            num_failed += 1
    printMsg("Finished to run all tests")

    if num_failed != 0:
        printMsg("You have {0} failed tests out of {1}\nThe failed tests are:\n{2}".format(num_failed, num_tests, failed_tests_list))
    else:
        printMsg("All tests have run successfuly")


if __name__ == "__main__":
    RunAllTests()
