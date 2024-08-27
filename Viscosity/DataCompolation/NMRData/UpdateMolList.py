#get names from xml files in NistData folder and update the molList.csv file

import os
import numpy as np

#load molecules.csv
orignial_molList = np.loadtxt('NMRData\molecules.csv',delimiter='\t',dtype='str',skiprows=1)

molList = []

for file in os.listdir(r'ThermoData\NistData'):
    if file.endswith('.xml'):
        if file[:-4] in orignial_molList[:,0]:
            print(file[:-4])
            molList.append([file[:-4],orignial_molList[orignial_molList[:,0]==file[:-4]][0][1]])
print(len(molList))

#save molList.csv
molList = np.array(molList)
np.savetxt('NMRData/molecules.csv',molList,delimiter='\t',fmt='%s')

