import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import signal

#side by side plot the 1H spectra and the surface tension data Y and t

#load the data
with open('TrainingDB.json', 'r') as f:
    data = json.load(f)
    db_1H = data['1H']

print(len(db_1H))

for i in range(len(db_1H)):
    print(db_1H[i]['Smiles'])
#load t and y form db
t = db_1H[30]['t']
y = db_1H[30]['y']
#load the 1H spectra
x = db_1H[30]['Frequency (ppm)']
spectra_intensity = db_1H[30]['Intensity']

#plot the 1H spectra
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(x,spectra_intensity)
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.title('1H NMR Spectrum')
plt.gca().invert_xaxis()

#plot the surface tension data
plt.subplot(1,2,2)
plt.plot(t,y)
plt.xlabel('Temperature (K)')
plt.ylabel('Surface Tension (N/m)')
plt.title('Surface Tension Data')

plt.show()


for i in range(len(db_1H)):
    plt.plot(db_1H[i]['Frequency (ppm)'],db_1H[i]['Intensity'],label=db_1H[i]['Smiles'],alpha=0.5)
plt.gca().invert_xaxis()
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.title('1H NMR Spectra')
plt.show()

#decimate the 1H spectra to 100 points
for i in range(len(db_1H)):
    x = db_1H[i]['Frequency (ppm)']
    spectra_intensity = db_1H[i]['Intensity']
    x = signal.decimate(x,10)
    spectra_intensity = signal.decimate(spectra_intensity,10)
    #intes = np.cumsum(intes)
    db_1H[i]['Frequency (ppm)'] = x
    db_1H[i]['Intensity'] = spectra_intensity
    

for i in range(len(db_1H)):
    plt.plot(db_1H[i]['Frequency (ppm)'],db_1H[i]['Intensity'],label=db_1H[i]['Smiles'],alpha=0.5)
plt.gca().invert_xaxis()
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.title('1H NMR Spectra')
plt.show()
