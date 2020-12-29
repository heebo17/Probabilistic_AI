
  
import numpy as np

# Scikit Learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.gaussian_process.kernels import RBF, Matern,RationalQuadratic, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing

# Scipy 
from scipy.stats import norm

# Debugging
#import ipdb

# Plotting 
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array
        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:
    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)
It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():
    def __init__(self):
        self.estimator = None
        self.score = make_scorer(cost_function)
        """
        self.grid = [{"gpr__alpha": np.logspace(-5,-1,5)}]
        self.pipe = Pipeline([("gpr", GaussianProcessRegressor(kernel=Matern(), 
                                                         normalize_y=True,
                                                         n_restarts_optimizer = 10,
                                                          random_state=17))])

        self.gridsearch = GridSearchCV(self.pipe, self.grid, scoring=self.score,
                                       verbose=1, cv=5)
        """
        self.kernel =  Matern() + WhiteKernel()
        self.estimator = GaussianProcessRegressor(kernel=self.kernel, 
                                                         normalize_y=False,
                                                         n_restarts_optimizer = 10,
                                                          random_state=17)
        

    def predict(self, test_x):

        test_x = preprocessing.scale(test_x)

        y, y_std = self.estimator.predict(test_x, return_std = True)
        
        
        #Asymmetric cost adjustment
        cost = 1 - norm.cdf(np.divide(0.5-y,y_std))
        a = np.divide(W2+W3*cost, cost+1)
        b = W1*np.ones(100)
        y = y + np.multiply(y_std, norm.ppf(np.divide(a,a+b)))

        
        return y    


    def fit_model(self, train_x, train_y):

        
        sparse = np.where(train_x[:,0] >-0.5)[0]
        x_train = train_x[sparse]
        y_train = train_y[sparse]
        x_train = preprocessing.scale(x_train)
        
        """
        self.gridsearch.fit(x_train, y_train)
        self.estimator = self.gridsearch.best_estimator_
        print(self.gridsearch.best_estimator_)
        """
        self.estimator.fit(x_train, y_train)
        
        
        
 

def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)
    print(prediction)





if __name__ == "__main__":
    main()