import numpy as np

data = np.loadtxt('InitFuelList\CAS_Smiles.txt', dtype=str, delimiter='\t', skiprows=1)
CAS = data[:, 0]
Smiles = data[:, 1]
Name = data[:, 2]

print(len(Smiles))
#if the name has a . or a [ or a ] in it, delete the entry
CAS_filtered = np.array([CAS[i] for i in range(len(Name)) if '.' not in Name[i] and '[' not in Name[i] and ']' not in Name[i]])
Smiles_filtered = np.array([Smiles[i] for i in range(len(Name)) if '.' not in Name[i] and '[' not in Name[i] and ']' not in Name[i]])
Name_filtered = np.array([Name[i] for i in range(len(Name)) if '.' not in Name[i] and '[' not in Name[i] and ']' not in Name[i]])
print(len(Smiles_filtered))

#save
CAS_Smiles = np.column_stack((CAS_filtered, Smiles_filtered, Name_filtered))
np.savetxt('InitFuelList/NameCanadites.txt', CAS_Smiles, fmt='%s', delimiter='\t', header='CAS,SMILES,Name', comments='')