import pandas as pd
import numpy as np
import math
from metrics import * 
from base import DecisionTree  
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
np.random.seed(42)

import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, sep='\s+', header=None,
                   names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "model year", "origin", "car name"])
X=[]
y=[]
for index, row in data.iterrows():
       feature_values = row[:-1].values  
       target_value = row['mpg'] 

       features = [float(value) for value in feature_values]

            
       X.append(features)
       y.append(target_value)
    
def sktrain(X,y,max_depth=5,criterion='squared_error'):
    """Function to train and predict on estate dataset using sklearn decision tree"""
    # Dropping any rows with Nan values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

    regressor = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth) 

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    score = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rscore = math.sqrt(score)

    return rscore

def my_regr(X,y,max_depth=5,criterion="information_gain"):
    """Function to train and predict on estate dataset using my decision tree"""
    clf = DecisionTree(criterion=criterion,max_depth=max_depth)

    clf.fit(pd.DataFrame(X[0:330]),pd.Series(y[0:330]))

    # clf.plot()

    y = y[330:]

    y_hat = clf.predict(pd.DataFrame(X[330:]))

    y = pd.Series(y)

    print(rmse(y_hat,y))
    print(mae(y_hat,y))


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

print("\n")

print("\n")
print("1) using sklearn: ")
print("using RMSE score and MAE score respectively")
print(sktrain(X,y))
print(sktrain(X,y,criterion='squared_error'))

print("\n\n")

print("2) using my decision tree: ")
print("using RMSE score and MAE score respectively")
my_regr(X,y)

print("\n")

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")