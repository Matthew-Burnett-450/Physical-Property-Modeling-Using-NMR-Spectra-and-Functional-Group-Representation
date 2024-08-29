import numpy as np
from DataCompolation.NMRFeatureGenerator import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NMRModel import NMR1DCNN
from scipy import signal

# Load the model
model = NMR1DCNN()
model.load_state_dict(torch.load('Model.pth'))
model.eval()

# Load the data from get_data_test
x, Y, T, Names, features2 = get_data()

A = np.ones((len(Y), 1))
B = np.ones((len(Y), 1))
for i in range(len(Y)):
    y = Y[i]
    y = np.array(y)
    y = np.log(y)
    t = T[i]
    t = np.array(t)

    y = y[t > 250]
    t = t[t > 250]
    y = y[t < 600]
    t = t[t < 600]
    name = Names[i]
    
    # Fit A + B*t
    a = np.ones((len(t), 1))
    b = t.reshape(-1, 1)
    X = np.concatenate((a, 1 / b), axis=1)
    linear = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X.dot(linear[0])
    r2 = r2_score(y, y_hat)
    print(f'{name} R2: {r2}')
    A[i] = linear[0][0]
    B[i] = linear[0][1]

Y = np.hstack((A, B))
print(Y.shape)

# Standardize the data
scaler = StandardScaler()
Y = scaler.fit_transform(Y)


# Load the data from JP5.csv
data = np.loadtxt('JP5.csv', delimiter=',', skiprows=1)
x_jet = data[:, 0]
features = data[:, 1]


#down sample till same length as x
x_jet = signal.decimate(x_jet, 10)
features = signal.decimate(features, 10)
#reshape features to 1,6554
features = features.reshape(1, 6554)

#plot features and x_jet features2[0]
plt.plot(x_jet, features[0], label='JP5')
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.legend()
plt.show()


#predict the y values for the JP5 data
features = torch.tensor(features.copy(),dtype=torch.float32).reshape(1, 1, 6554)
y_jet = model(features)
y_jet = y_jet.detach().numpy()

y_jet = scaler.inverse_transform(y_jet)

print(y_jet)

#predict the y values for the JP5 data
T = np.arange(250, 600, 1)

y_pred = np.exp(y_jet[0][0] + y_jet[0][1] * (1 / T)) *1000

Temp = [15+273.15, 25+273.15, 35+273.15, 45+273.15, 55+273.15, 65+273.15, 75+273.15]
Viscosity = [1.78, 1.61, 1.46, 1.34, 1.14, 0.978, 0.853]

#calculate MAPE
Temp = np.array(Temp)
Viscosity = np.array(Viscosity)
Visc_pred = np.exp(y_jet[0][0] + y_jet[0][1] * (1 / Temp)) *1000
MAPE = np.mean(np.abs((Viscosity - Visc_pred) / Viscosity)) * 100
print(f'Mean Absolute Percentage Error: {MAPE:.2f}%')

plt.plot(T, y_pred, label='Predicted',color='k',linestyle='--')
plt.scatter(Temp, Viscosity, label='Experimental',color='k')
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (cP)')
plt.legend()
plt.show()

#save the predicted values to a csv file and the T
data = np.concatenate((T.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
np.savetxt('JP5_predicted.csv', data, delimiter=',', header='T,Viscosity', comments='')

#save Temp and Viscosity to a csv file
data = np.concatenate((Temp.reshape(-1, 1), Viscosity.reshape(-1, 1)), axis=1)
np.savetxt('JP5_experimental.csv', data, delimiter=',', header='T,Viscosity', comments='')
