import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
#from statistics import NormalDist


domain = np.array([[0, 5]])



class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.x_init =domain[:, 0]+(domain[:, 1]-domain[:, 0])*np.random.rand(domain.shape[0])
        self.x_opt = np.array([])
        self.y_f = np.array([])
        self.y_v = np.array([])


        # REVIEW THE VALUES

        self.f_var = 0.15
        self.v_var = 0.0001
        self.v_mean = 1.5

        self.f_kernel = Matern(length_scale=0.5, length_scale_bounds="fixed",nu=2.5)\
                         + WhiteKernel(noise_level=self.f_var)
        self.v_kernel =(ConstantKernel(constant_value=1.5)\
                        *  Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5))\
                         + WhiteKernel(noise_level=self.v_var)

        self.i = 1
        self.v_min =1.2

        self.f_gp = gpr(kernel=self.f_kernel,
                         random_state=17)
        self.v_gp = gpr(kernel=self.v_kernel, 
                         random_state=17)

        self.f_best = -1000
        pass


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        
        #AF greedy algorithm, x_t = argmax( AF(x) ) over x
        if self.i == 1:
          x_t =  np.atleast_2d(self.x_init)
          self.i += 1
        else:
          x_t = self.optimize_acquisition_function()

        return np.atleast_2d(x_t)

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        
        #EI formula (2) from the SNOEK paper
        x = np.atleast_2d(x)
        f_pred = self.f_gp.predict(x, return_std=True)
        f_gamma = (f_pred[0] - self.f_best)/f_pred[1]

        f_ei = f_pred[1] * (f_gamma*norm.cdf(f_gamma) + norm.pdf(f_gamma))
       
        #probabilistic constraint from GELBART paper
        v_pred = self.v_gp.predict(x, return_std=True)
        v_gamma = (v_pred[0] - self.v_min)/v_pred[1]
        prob_constraint = norm.cdf(v_gamma) 

        #Threshold to be sure the 
        if(prob_constraint < 0.5):
            af = prob_constraint
        else :
            af = prob_constraint * f_ei
        return af


        


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here

        self.x_opt = np.append(self.x_opt, x)
        self.x_opt = np.reshape(self.x_opt, (-1,1))
        self.y_f = np.append(self.y_f, f)
        self.y_v = np.append(self.y_v, v)


        #update models:
        

        try:
            self.f_gp.fit(self.x_opt, self.y_f)
            self.v_gp.fit(self.x_opt, self.y_v)
            self.f_best = self.y_f.max()
        except:
            self.f_best = self.y_f.max()

        # update best values
        


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here


        try:
          speed = self.y_v >= self.v_min
          x_solution = self.x_opt[speed]
          y_f_solution = self.y_f[speed]
          index = y_f_solution.argmax()
          x_solution = x_solution[index]
        except:
          print("except")
          
          index = y_f.argmax()
          x_solution = self.x_opt[index]
          """
          self.f_gp.fit(self.x_opt, self.y_f)
          self.v_gp.fit(self.x_opt, self.y_v)
          x_solution = self.optimize_acquisition_function()
          """
        return x_solution
     


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()
        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')

if __name__ == "__main__":
    main()