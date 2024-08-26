import numpy as np
import os

data = np.loadtxt(r'InitFuelList/NameCanadites.txt', dtype=str, delimiter='\t', skiprows=1)

CAS = data[:, 0]
smiles = data[:, 1]
names = data[:, 2]

paths = ["ThermoData/NistData/" + name + '.xml' for name in names]

#if a path does not exist, remove the entry and remove it from cas,smiles and names then save
names_filtered = []
smiles_filtered = []
CAS_filtered = []
for i in range(len(paths)):
    if os.path.exists(paths[i]):
        names_filtered.append(names[i])
        smiles_filtered.append(smiles[i])
        CAS_filtered.append(CAS[i])
    else:
        print(paths[i])

print(len(names_filtered))



CAS_Smiles = np.column_stack((CAS_filtered, smiles_filtered, names_filtered))
np.savetxt(r'InitFuelList\NameCanadites.txt', CAS_Smiles, fmt='%s', delimiter='\t', header='CAS\tSMILES\tName', comments='')