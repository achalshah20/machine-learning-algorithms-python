from AchalML.base import LinearModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression(LinearModel):

    def __init__(self,alpha=0.01,verbose=False,l1=0,l2=0,costTolerance=0.01,maxIter=10000):
        self.__costHistory__ = []
        self.__theta__ = []
        self.__alpha__ = alpha
        self.__iterValue__ = 0
        self.__verbose__ = verbose
        self.__l1__ = l1
        self.__l2__ = l2
        self.__costTolerance__ = costTolerance
        self.__maxIter__ = maxIter

    # Return the coefficients of the model
    def coeffs_(self):
        return self.__theta__

    # Calculate the cost function - mean squared error
    def __calcCost__(self, X, y):

        n = X.shape[0]

        # l1 and l2 are lasso and ridge coefficients respectively
        # First part calculated the mean squared error
        # We are excluding intercept from the
        cost = (1/2*n) * sum((X.dot(self.__theta__) - y) ** 2) + self.__l1__* sum(self.__theta__[1:]) + self.__l2__ * sum(self.__theta__[1:] ** 2)

        return cost

    # Run step gradient descent
    def __gradientDescent__(self, X, y):

        n = X.shape[0]
        nGrad = X.shape[1]
        grad = np.zeros(nGrad)

        init_cost = self.__calcCost__(X, y)
        self.__costHistory__.append(init_cost)
        for iter in range(self.__maxIter__):

            prevCost = self.__calcCost__(X, y)

            for i in range(nGrad):
                grad[i] = (1/n) * (sum((X.dot(self.__theta__) - y) * X.iloc[:, i]) + self.__l2__ * (self.__theta__[i]) + self.__l1__)

            for i in range(nGrad):
                self.__theta__[i] = self.__theta__[i] - self.__alpha__ * grad[i]

            newCost = self.__calcCost__(X, y)
            #print("new cost: ",newCost)
            self.__costHistory__.append(newCost)

            if(iter % 10 == 0 and self.__verbose__):
                print("MSE improvement after %s iterations: "%(iter) ,newCost)

            if((prevCost - newCost) < 0.01):
                break

        self.__iterValue__ = iter + 1
        return self.__theta__

    # Plot learning graph
    def plotLearningGraph(self):

        plt.xlabel("Number of iterations",fontsize=16)
        plt.ylabel("Total cost",fontsize=16)
        plt.title("Learning path for alpha: %s " % (self.__alpha__), fontsize=20)
        plt.xlim([0, self.__iterValue__])

        plt.plot(self.__costHistory__)
        plt.show()

    # Train model
    def train(self, X, y):
        X_temp = pd.DataFrame(X).copy()
        X_temp.insert(0,"intercept",np.ones(X_temp.shape[0]))
        self.__theta__ = np.zeros(X_temp.shape[1])
        return self.__gradientDescent__(X_temp, y)

    # Predict over data
    def predict(self,X):
        X = pd.DataFrame(X).copy()
        X.insert(0,"intercept",np.ones(X.shape[0]))
        return X.dot(self.__theta__)


