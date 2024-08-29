import numpy as np
from DataCompolation.FGFeatureGenerator import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FGModel import FGNN
from scipy import signal

# Load the model
model = FGNN(input_length=7)
model.load_state_dict(torch.load('FGModel.pth'))
model.eval()

# Load the data from get_data_test
FG, Target,T,Visc,Names = get_data()

# Standardize the data
scaler = StandardScaler()
Y = scaler.fit_transform(Target)


features = FG
"""    smarts_patterns = {
        'CH2': '[CH2]',                      # Matches general CH2 groups
        'CH3': '[CH3]',                      # Matches CH3 groups
        'CH': '[CH]',                        # Matches CH groups
        'C': '[C]([C])([C])([C])[C]',        # Matches C connected to exactly four other carbons
        'Benzyl': 'c1ccccc1',                    # Matches aromatic carbon with one hydrogen
        'CH2_cyclo': '[R][CH2]',             # Matches CH2 groups in a ring (cyclic structure)
        'CH2_chain_3plus': '[CH2]([CH2])[CH2]'  # Matches any CH2 that is part of a chain of 3 or more CH2 groups
    }
"""
Jet_FG = [1.011418,2.798,0.6946,0,.4929,.4810,4.366289]
Jet_FG = np.array(Jet_FG)

Visc_pred = model(torch.tensor(Jet_FG, dtype=torch.float32)).detach().numpy()

Visc_pred = scaler.inverse_transform(Visc_pred.reshape(1, -1))[0]

#predict the y values for the JP5 data
T = np.arange(250, 600, 1)

y_pred = np.exp(Visc_pred[0] + (Visc_pred[1] / T))*1000


Temp = [15+273.15, 25+273.15, 35+273.15, 45+273.15, 55+273.15, 65+273.15, 75+273.15]
Viscosity = [1.78, 1.61, 1.46, 1.34, 1.14, 0.978, 0.853]
V_hat = np.exp(Visc_pred[0] + (Visc_pred[1] / Temp))*1000
#calculate MAPE
Temp = np.array(Temp)
Viscosity = np.array(Viscosity)
MAPE = np.mean(np.abs((Viscosity - V_hat) / Viscosity)) * 100
print(f'Mean Absolute Percentage Error: {MAPE:.2f}%')

plt.plot(T, y_pred, label='Predicted',color='k',linestyle='--')
plt.scatter(Temp, Viscosity, label='Experimental',color='k')
plt.xlabel('Temperature (K)')
plt.ylabel('Viscosity (cP)')
plt.legend()
plt.show()

#save the predicted values to a csv file and the T
data = np.concatenate((T.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
np.savetxt('JP5_predicted_FG.csv', data, delimiter=',', header='T,Viscosity', comments='')

#save Temp and Viscosity to a csv file
data = np.concatenate((Temp.reshape(-1, 1), Viscosity.reshape(-1, 1)), axis=1)
np.savetxt('JP5_experimental_FG.csv', data, delimiter=',', header='T,Viscosity', comments='')
