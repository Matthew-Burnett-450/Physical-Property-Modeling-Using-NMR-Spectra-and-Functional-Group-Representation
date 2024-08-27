import numpy as np
import json
from ThermoMlReader import ThermoMLParser
import os
from NistDataGrabberSurfT import TargetConstructor
import TDEEquations as TDEEq


data = np.loadtxt(r'InitFuelList\NameCanadites.txt', dtype=str, delimiter='\t', skiprows=1)
CAS = data[:, 0]
smiles = data[:, 1]
names = data[:, 2]

paths = ["ThermoData/NistData/" + name + '.xml' for name in names]

print(len(names))
print(len(smiles))
print(len(paths))


EqDict={'TDE.PPDS9':TDEEq.PPDS9,'TDE.NVCLSExpansion.Version1':TDEEq.ViscosityL}

Constructor=TargetConstructor(EqDict)


t=[]
Y=[]
Eq_names=[]
print(len(paths))
for path in paths:
    x_,Y_,EqName = Constructor.GenerateTargets(path,'Viscosity, Pa*s',state='Liquid')
    #print first 5 values of x_,Y_,Tc__,EqName
    print(x_[:5],Y_[:5],EqName)
    t.append(x_)
    Y.append(Y_)
    Eq_names.append(EqName)

print(len(t))
print(len(Y))
print(len(Eq_names))

#list of dicts with t,y,Tc and name
HydrocarbonData=[]
for i in range(len(names)):
    print(names[i],smiles[i],len(t[i]),len(Y[i]),Eq_names[i])
    HydrocarbonData.append({'name':names[i],'smiles':smiles[i],'t':t[i],'y':Y[i]})



#save the data as json file
with open('ThermoData/NISTData.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)



