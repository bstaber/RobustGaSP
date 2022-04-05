from rgaspy import log_profile_lik
import numpy as np
from scipy.spatial import distance_matrix

param = np.ndarray(shape=(1,), buffer=np.array([0.1]), dtype=float)
nugget = 1.0
nugget_est = False
X = np.random.randn(10,2)
Y = np.random.randn(10,2)
zero_mean = "Yes"
kernel_type = np.ndarray(shape=(1,), buffer=np.array([1]), dtype=int)
R0 = [distance_matrix(X,X)]
alpha = np.ndarray(shape=(1,), buffer=np.array([0.5]), dtype=float)

for _ in range(10):
    ret = log_profile_lik(param, nugget, nugget_est, R0, X, zero_mean, Y, kernel_type, alpha)
    print(ret)