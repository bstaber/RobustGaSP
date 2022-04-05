import rgaspy
import numpy as np
from functools import partial
from scipy.spatial import distance_matrix
from scipy.optimize import minimize

class RGaSPModel:
    
    def __init__(self,
                 input: np.ndarray,
                 output: np.ndarray,
                 X: np.ndarray=None,
                 R0: list=None,
                 zero_mean: bool=True,
                 prior_choice: str='ref_approx',
                 prior_params: dict={"a": 0.2, "b": None},
                 kernel_type: list=["matern_5_2"],
                 alpha: np.ndarray=None,
                 isotropic: bool=False):
        
                    self.input = input
                    self.output = output
                    self.X = X
                    self.zero_mean = zero_mean
                    self.prior_choice = prior_choice
                    self.isotropic = isotropic
                    self.prior_params = prior_params
                    
                    if len(input.shape)==1:
                        raise ValueError("Input should be a matrix")
                    if len(output.shape)!=1:
                        raise ValueError("Output should be a vector, not a matrix")
                    if input.shape[0]!=output.shape[0]:
                        raise ValueError("The number of observations is not equal to the number of experiments")
                    self.num_obs, d = input.shape
                    
                    if alpha is None:
                        alpha = np.array([1.9]*input.shape[1])
                    
                    if "a" not in prior_params or "b" not in prior_params:
                        raise ValueError("Dictionary prior_params missing keys a or b")
                    if not isinstance(prior_params["a"], float):
                        raise ValueError("Parameter a in prior_params should be a float")
                    if prior_params["b"] is None:
                        prior_params["b"] = (1.0/len(output)**d)*(prior_params["a"] + d)
                    
                    if self.isotropic:
                        self.p = 1
                        self.alpha = alpha[0]
                    else:
                        self.p = d
                        self.alpha = alpha
                    
                    if self.zero_mean:
                        self.X = np.zeros((len(output),1))
                        self.q = 0
                        self.zero_mean = "Yes"
                    elif not self.zero_mean and self.X is None:
                        self.X = np.ones((len(output),1))
                        self.q = 1
                        self.zero_mean = "No"
                    elif not self.zero_mean and self.X is not None:
                        if self.X.shape[0]!=self.num_obs or self.X.shape[1]!=1:
                            raise ValueError("X's shape should be [num_obs, 1]")
                        self.q = 1
                        self.zero_mean = "No"
                    
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
                    self.kernel_type_num = kernel_type_num
                    
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
    
    def search_lower_bounds(self):
        raise NotImplementedError()
    
    def get_lower_bounds(self, lower_bound, nugget_est):
        if lower_bound:
            lb = self.search_lower_bounds()
            if nugget_est:
                lb = np.concatenate([lb, -np.Inf])
        else:
            # lb = np.zeros(self.p)
            # for idx in range(self.p):
            #     ilb = -np.log(0.1)/((np.max(self.input[:,idx])-np.min(self.input[:,idx]))**self.p)
            #     lb[idx] = ilb
            if nugget_est:
                lb = np.array([np.Inf]*(self.p+1))
            else:
                lb = np.array([np.Inf]*self.p)
        return lb
    
    def get_initial_values(self, lb, num_initial_values):
        raise NotImplementedError()
    
    def fit(self, method: str="post_mode", nugget: float=0.0, nugget_est: bool=False, lower_bound: bool=True, optimization: str="L-BFGS-B", max_eval: int=None, initial_values=None, num_initial_values=2):
        
        if method not in ["post_mode", "mle", "mmle"]:
            raise ValueError("The method should be post_mode, mle or mmle.")
        
        if optimization not in ["L-BFGS-B"]:
            raise ValueError("Only LBFGS is supported at the moment.")
        
        lbs = self.get_lower_bounds(lower_bound, nugget_est)
        bounds = []
        for lb in lbs:
            bounds.append((lb, None))
        bounds = tuple(bounds)
                
        if initial_values is None:
            initial_values = self.get_initial_values(lb, num_initial_values)
            
        for i_ini in range(num_initial_values):
            if nugget_est:
                ini_value=initial_values[i_ini]
            else:
                ini_value=initial_values[i_ini,1:self.p]
        
            log_post=-np.Inf
            if method=="post_mode":
                if self.prior_choice=="ref_approx":
                    def neg_log_marginal_post_approx_ref(param, nugget, nugget_est, R0, X, zero_mean, output, CL, a, b, kernel_type, alpha):
                        lml=rgaspy.log_marginal_lik(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha)
                        lp=rgaspy.log_approx_ref_prior(param, nugget, nugget_est, CL, a, b)
                        return -lml-lp
                    def neg_log_marginal_post_approx_ref_deriv(param, nugget, nugget_est, R0, X, zero_mean, output, CL, a, b, kernel_type, alpha):
                        lml_dev=rgaspy.log_marginal_lik_deriv(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha)
                        lp_dev=rgaspy.log_approx_ref_prior_deriv(param, nugget, nugget_est, CL, a, b)
                        return -(lml_dev+lp_dev)*np.exp(param)
                    
                    a, b = self.prior_params["a"], self.prior_params["b"]

                    tt_all = minimize(fun=neg_log_marginal_post_approx_ref,
                                      x0=ini_value, 
                                      args=(nugget, nugget_est, self.R0, self.X, self.zero_mean, self.output, self.CL, a, b, self.kernel_type, self.alpha),
                                      method=optimization, 
                                      jac=neg_log_marginal_post_approx_ref_deriv, 
                                      bounds=bounds, 
                                      options={'maxfun': max_eval, 'maxiter': 15000, 'maxls': 20})
                    
                elif self.prior_choice in ["ref_xi", "ref_gamma"]:
                    # fun = rgaspy.neg_log_marginal_post_ref
                    # tt_all = minimize(fun=fun, x0=ini_value, method=optimization, jac=None, 
                    #                   bounds=bounds, options={'maxfun': max_eval, 'maxiter': 15000, 'maxls': 20})
                    raise NotImplementedError()
            elif method=="mle":
                # fun = rgaspy.neg_log_profile_lik
                # jac = rgaspy.neg_log_profile_lik_deriv
                # tt_all = minimize(fun=fun, x0=ini_value, method=optimization, jac=jac, 
                #                     bounds=bounds, options={'maxfun': max_eval, 'maxiter': 15000, 'maxls': 20})
                raise NotImplementedError()
            elif method=="mmle":
                # fun = rgaspy.neg_log_marginal_lik
                # jac = rgaspy.neg_log_marginal_lik_deriv
                # tt_all = minimize(fun=fun, x0=ini_value, method=optimization, jac=jac, 
                #                     bounds=bounds, options={'maxfun': max_eval, 'maxiter': 15000, 'maxls': 20})
                raise NotImplementedError()

            if tt_all.success:
                if not nugget_est:
                    nugget_par = nugget
                else:
                    nugget_par = np.exp(tt_all.x)[-1]
                    
                if -tt_all.fun > log_post:
                    log_lik=-tt_all.fun
                    log_post=-tt_all.fun
                    if nugget_est:
                        self.beta_hat = np.exp(tt_all.x)[1:self.p]
                        self.nugget_hat = np.exp(tt_all.x)[-1]
                    else:
                        self.beta_hat = np.exp(tt_all.x)
                        self.nugget_hat = nugget
        return log_post
    
    def construct_rgasp(self, beta_hat: np.array, nugget: float):
        ret = rgaspy.construct_rgasp(beta_hat, nugget, self.R0, self.X, 
                                     self.zero_mean, self.output, self.kernel_type_num, self.alpha)
        return ret
     
    def predict(self):
        raise NotImplementedError()
    
    # def __str__(self):
    #     raise NotImplementedError()
    
if __name__ == "__main__":
    
    X_train = np.random.randn(100,2)
    Y_train = np.random.randn(100)
    model = RGaSPModel(input=X_train, output=Y_train, zero_mean=True)