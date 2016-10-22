import numpy as np

def root_mean_squared_error(y,y_hat):
    return (np.sqrt(sum((y - y_hat)**2)))

def mean_squared_error(y,y_hat):
    return (sum((y - y_hat)**2))

def mean_absolute_error(y,y_hat):
    return (sum((y - y_hat)))
