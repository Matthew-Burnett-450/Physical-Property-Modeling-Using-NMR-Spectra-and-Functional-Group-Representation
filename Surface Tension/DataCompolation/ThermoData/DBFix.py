import os
import numpy as np
import pubchempy as pcp

data = np.loadtxt(r'InitFuelList/NameCanadites.txt', dtype=str, delimiter='\t', skiprows=1)

CAS = data[:, 0]
smiles = data[:, 1]
names = data[:, 2]

#tolist
CAS = CAS.tolist()
smiles = smiles.tolist()
names = names.tolist()



#check NistData folder for Name.xml files and add those names to a NameCanadites if its not already there use pubchempy to get the smiles add names from NistData folder to NameCanadites
NistData = os.listdir('ThermoData/NistData')
for file in NistData:
    name = file.split('.')[0]
    if name not in names:
        #check if name has (1) in it
        if '(1)' not in name:
            #add Unkown to the list
            smiles.append('Unknown')
            names.append(name)
            CAS.append('Unknown')
            print(f"Failed to find name for {name}")
            continue



CAS_Smiles = np.column_stack((CAS, smiles, names))
np.savetxt(r'InitFuelList\NameCanadites.txt', CAS_Smiles, fmt='%s', delimiter='\t', header='CAS\tSMILES\tName', comments='')
