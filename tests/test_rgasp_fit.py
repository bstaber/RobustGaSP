import numpy as np
from interface.models import RGaSPModel

x = np.random.rand(10,1)
y = (np.cos(x) + np.sin(2.0*x)).squeeze()

m = RGaSPModel(input=x, output=y, zero_mean=True)

log_post = m.fit(method="post_mode", 
                 nugget=0.0, nugget_est=False,
                 lower_bound=True, optimization="L-BFGS-B", 
                 max_eval=100, initial_values=None, num_initial_values=2)