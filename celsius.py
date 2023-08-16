import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X=np.array([[68],[104],[140],[176],[212]])
y=np.array([[20],[40],[60],[80],[100]])
model=LinearRegression()
model.fit(X,y)
