import numpy as np
import json
from ThermoMlReader import ThermoMLParser
import os
from NistDataGrabberSurfT import TargetConstructor
import TDEEquations as TDEEq


name_ls = np.loadtxt('NMRData\molecules.csv',delimiter='\t',dtype='str',skiprows=1)
names = name_ls[:,0]
smiles = name_ls[:,1]
paths = ["ThermoData/NistData/" + name + '.xml' for name in names]

EqDict={'TDE.PPDS14':TDEEq.PPDS14,'TDE.Watson':TDEEq.Watson,'TDE.ISTExpansion.SurfaceTension':TDEEq.ISETExpansion}

Constructor=TargetConstructor(EqDict)


t=[]
Y=[]
Tc=[]
print(len(paths))
for path in paths:
    x_,Y_,Tc__=Constructor.GenerateTargets(path,'Surface tension liquid-gas, N/m')
    t.append(x_)
    Y.append(Y_)
    Tc.append(Tc__)


print(len(t))
print(len(Y))
print(len(Tc))

#list of dicts with t,y,Tc and name
HydrocarbonData=[]
for i in range(len(names)):
    HydrocarbonData.append({'name':names[i],'smiles':smiles[i],'t':t[i],'y':Y[i],'Tc':Tc[i]})



#save the data as json file
with open('ThermoData/NISTData.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)



