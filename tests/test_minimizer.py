import numpy as np
from scipy.optimize import minimize
from rgaspy import test_objective_function as fun

ret = minimize(fun=fun, x0=np.random.randn(2), args=(2.0))

print(ret)