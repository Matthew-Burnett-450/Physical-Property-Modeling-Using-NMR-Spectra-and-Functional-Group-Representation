import numpy as np
from DataCompolation.FeatureGenerator import get_data_test
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from skorch import NeuralNetRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load the data
x,Y,T,Names,features = get_data_test()

for i in range(len(Y)):
    y = Y[i]
    t = T[i]
    name = Names[i]

    plt.plot(t, y, label=f'{name}')
plt.legend()
plt.show()
