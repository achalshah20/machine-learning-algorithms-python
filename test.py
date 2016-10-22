from AchalML.Regression import LinearRegression
from AchalML.metrics import root_mean_squared_error
import pandas as pd
import numpy as np


if __name__ == '__main__':
    data = pd.read_csv("data/temp.txt")

    data.columns = ["A","B"]

    x_train = data.drop("B",axis=1)

    y = data.B


    reg_model = LinearRegression(alpha=0.02,verbose=False,costTolerance=0,maxIter=1000000)

    reg_model.train(x_train,y)
    #print(x_train)
    predictions = reg_model.predict(x_train)

    print("Root mean squared error: ",root_mean_squared_error(y,predictions))
    print("coeffs: ", reg_model.coeffs_())
