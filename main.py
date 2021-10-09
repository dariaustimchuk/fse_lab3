from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
import sys
sys.path.insert(1, '/home/daria/Documents/practice_project/useful_package
')
from module_a import polynom_3
from module_a import hyperbola
def regr(X):
    y = np.array(hyperbola(X))
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)

    print((snp.array(regr.predict(X, y) - y))**2)/len(y)

    y = np.array(polynom_3(X))
    X, y = make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)

    print((snp.array(regr.predict(X, y) - y))**2)/len(y)
