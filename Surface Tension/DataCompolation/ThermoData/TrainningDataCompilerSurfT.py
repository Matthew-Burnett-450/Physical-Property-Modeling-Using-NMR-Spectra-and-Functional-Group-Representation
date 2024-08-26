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

#wait for user to press enter
input('Press enter to continue')


EqDict={'TDE.PPDS14':TDEEq.PPDS14,'TDE.Watson':TDEEq.Watson,'TDE.ISTExpansion.SurfaceTension':TDEEq.ISETExpansion}

Constructor=TargetConstructor(EqDict)


t=[]
Y=[]
Tc=[]
Eq_names=[]
print(len(paths))
for path in paths:
    x_,Y_,Tc__,EqName = Constructor.GenerateTargets(path,'Surface tension liquid-gas, N/m')
    #print first 5 values of x_,Y_,Tc__,EqName
    print(x_[:5],Y_[:5],Tc__,EqName)
    t.append(x_)
    Y.append(Y_)
    Tc.append(Tc__)
    Eq_names.append(EqName)

print(len(t))
print(len(Y))
print(len(Tc))
print(len(Eq_names))

#list of dicts with t,y,Tc and name
HydrocarbonData=[]
for i in range(len(names)):
    print(names[i],smiles[i],len(t[i]),len(Y[i]),Tc[i],Eq_names[i])
    HydrocarbonData.append({'name':names[i],'smiles':smiles[i],'t':t[i],'y':Y[i],'Tc':Tc[i]})



#save the data as json file
with open('ThermoData/NISTData.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)



