import json
import numpy as np
# Open the file
with open('ViscosityTrainingData.json') as f:
    data = json.load(f)

#init numpy array
names = np.array(['']*len(data), dtype='object')
#grab the names
for i in range(len(data)):
    names[i] = data[i]['MolName']
    print(data[i]['MolName'])
#save the names
    
np.savetxt('names.txt', names,delimiter='\t', fmt='%s')