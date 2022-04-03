import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import minimize

class rgasp:
    
    def __init__(self,
                 input: np.ndarray,
                 output: np.ndarray,
                 X: np.ndarray=None,
                 R0: list=None,
                 zero_mean: bool=True,
                 alpha: np.ndarray=None,
                 nugget: float=0.0,
                 nugget_est: bool=False,
                 method: str="post_mode",
                 prior_choice: str='ref_approx',
                 kernel_type: list=["matern_5_2"],
                 isotropic: bool=False):
        
                    self.input = input
                    self.output = output
                    self.X = X
                    self.zero_mean = zero_mean
                    self.nugget = nugget
                    self.nugget_est = nugget_est
                    self.method = method
                    self.prior_choice = prior_choice
                    self.isotropic = isotropic
                    
                    if len(input.shape)==1:
                        raise ValueError("Input should be a matrix")
                    if len(output.shape)!=1:
                        raise ValueError("Output should be a vector, not a matrix")
                    if input.shape[0]!=output.shape[0]:
                        raise ValueError("The number of observations is not equal to the number of experiments")
                    self.num_obs, d = input.shape
                    
                    if alpha is None:
                        alpha = np.array([1.9]*input.shape[1])
                    
                    if self.isotropic:
                        self.p = 1
                        self.alpha = alpha[0]
                    else:
                        self.p = d
                        self.alpha = alpha
                    
                    if self.zero_mean:
                        self.X = np.zeros((len(output),1))
                        self.q = 0
                    elif not self.zero_mean and self.X is None:
                        self.X = np.ones((len(output),1))
                        self.q = 1
                    elif not self.zero_mean and self.X is not None:
                        if self.X.shape[0]!=self.num_obs or self.X.shape[1]!=1:
                            raise ValueError("X's shape should be [num_obs, 1]")
                        self.q = 1
                    
                    if method not in ["post_mode", "mle", "mmle"]:
                        raise ValueError("The method should be post_mode, mle or mmle.")
                    
                    if isinstance(kernel_type, str):
                        kernel_type = [kernel_type]
                    
                    if not self.isotropic:
                        if len(kernel_type)==1:
                            self.kernel_type = kernel_type*self.p
                        elif len(kernel_type)!=self.p:
                            raise ValueError("Please specify the correct number of kernels")
                        else:
                            self.kernel_type = kernel_type
                    else:
                        self.kernel_type = kernel_type
                    
                    kernel_type_num = np.zeros(self.p, dtype=np.int32)
                    for i_p, k_t in enumerate(kernel_type):
                        if k_t == "matern_5_2":
                            kernel_type_num[i_p] = 3
                        elif k_t == "matern_3_2":
                            kernel_type_num[i_p] = 2
                        elif k_t == "pow_exp":
                            kernel_type_num[i_p] = 1
                        elif k_t == "periodic_gauus":
                            kernel_type_num[i_p] = 4
                        elif k_t == "periodic_exp":
                            kernel_type_num[i_p] = 5
                        else:
                            raise ValueError("Kernel types must be matern_5_2, matern_3_2, pow_exp, periodic_gauss, or peridioc_exp")
                            
                    if R0 is None:
                        if not self.isotropic:
                            self.R0 = []
                            for i in range(self.p):
                                pdist = distance_matrix(self.input[:,i:i+1], self.input[:,i:i+1])
                                self.R0.append(pdist)
                        else:
                            self.R0 = [distance_matrix(self.input, self.input)]
                    elif isinstance(R0, np.ndarray):
                        self.R0 = [R0]
                    elif isinstance(R0, list):
                        self.R0 = R0
                    else:
                        raise ValueError("R0 should be either a matrix or a list")

                    if len(self.R0)!=self.p:
                        raise ValueError("Number of R0 matrices should be the same as the number of range parameters in the kernel")
                    for r0 in self.R0:
                        if r0.shape[0]!=self.num_obs or r0.shape[1]!=self.num_obs:
                            raise ValueError("The dimension of R0 matrices should match the number of observations")
                    
                    self.CL = np.zeros(self.p)
                    if not self.isotropic:
                        for i in range(self.p):
                            self.CL[i] = (np.max(self.input[:,i])-np.min(self.input[:,i]))/self.num_obs**(1.0/self.p)
                    else:
                        self.CL[0] = np.max(self.R0[0])/self.num_obs
    
    def fit(self, a=0.2, b=None, optimization="lbfgs", lower_bound=True, max_eval=None, initial_values=None, num_initial_values=2):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()
    
if __name__ == "__main__":
    
    X_train = np.random.randn(100,2)
    Y_train = np.random.randn(100)
    mode = rgasp(input=X_train, output=Y_train, zero_mean=True)