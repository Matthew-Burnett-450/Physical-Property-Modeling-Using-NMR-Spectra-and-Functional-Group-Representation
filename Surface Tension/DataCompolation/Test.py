import json
import numpy as np
import matplotlib.pyplot as plt
#load ViscosityTrainingData.json
with open('ViscosityTrainingData.json', 'r') as infile:
    HydrocarbonData = json.load(infile)
n=202
#grab the 13C shift of the first molecule
X=HydrocarbonData[n]['13C_shift']
print(HydrocarbonData[n]['MolName'])
#plot the histogram of the 13C shift
x=np.arange(0,250,.1)

plt.plot(x,X)
plt.xlabel('13C shift')
plt.ylabel('Frequency')
plt.title('Histogram of 13C shift')
plt.show()