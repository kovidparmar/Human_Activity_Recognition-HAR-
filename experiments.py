
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

""" Case: RIRO"""

learning_time = list()
predict_time = list()

for Ni in range(1,7):
    for step in range(6,42):
        N = Ni
        P = step
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
            
        start_time = time.time()
        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)
        end_time = time.time()
            
        learning_time.append(end_time-start_time)

        start_time = time.time()
        y_hat = tree.predict(X)
        end_time = time.time()
            
        predict_time.append(end_time-start_time)
plt.plot(list(learning_time))
plt.ylabel('RIRO : Fit time', fontsize=16)
plt.show()

plt.plot(list(predict_time))
plt.ylabel('RIRO : Predict time', fontsize=16)
plt.show()



""" Case: RIDO"""

learning_time = list()
predict_time = list()

for Ni in range(1,7):
    for step in range(6,42):
        N = Ni
        P = step
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
            
        start_time = time.time()
        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)
        end_time = time.time()
            
        learning_time.append(end_time-start_time)

        start_time = time.time()
        y_hat = tree.predict(X)
        end_time = time.time()
            
        predict_time.append(end_time-start_time)

plt.plot(list(learning_time))
plt.ylabel('RIDO : Fit time', fontsize=16)
plt.show()

plt.plot(list(predict_time))
plt.ylabel('RIDO : Predict time', fontsize=16)
plt.show()

""" Case: DIRO"""

learning_time = list()
predict_time = list()

for Ni in range(1,7):
    for step in range(6,42):
        N = Ni
        P = step
        X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(N))
            
        start_time = time.time()
        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)
        end_time = time.time()
            
        learning_time.append(end_time-start_time)

        start_time = time.time()
        y_hat = tree.predict(X)
        end_time = time.time()
            
        predict_time.append(end_time-start_time)


plt.plot(list(learning_time))
plt.ylabel('DIRO : Fit time', fontsize=16)
plt.show()

plt.plot(list(predict_time))
plt.ylabel('DIRO : Predict time', fontsize=16)
plt.show()

""" Case: DIDO"""

learning_time = list()
predict_time = list()

for Ni in range(1,7):
    for step in range(6,42):
        N = Ni
        P = step
        X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(P, size = N) , dtype="category")
            
        start_time = time.time()
        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)
        end_time = time.time()
            
        learning_time.append(end_time-start_time)

        start_time = time.time()
        y_hat = tree.predict(X)
        end_time = time.time()
            
        predict_time.append(end_time-start_time)

plt.plot(list(learning_time))
plt.ylabel('DIDO : Fit time', fontsize=16)
plt.show()

plt.plot(list(predict_time))
plt.ylabel('DIDO : Predict time', fontsize=16)
plt.show()