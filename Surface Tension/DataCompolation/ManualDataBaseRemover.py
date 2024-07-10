import json
with open('SurfTTrainingData.json', 'r') as infile:
    NMRData = json.load(infile)

#if not in list of names remove it 
#list of nalkanes plus 2,2,4-trimethylpentane, 2,2,4,4,6,8,8-heptamethylnonane, 2-methyl-butane

new_NMRData=[]
for Mol in NMRData:
    if Mol['MolName'].lower() in ['methane','ethane','propane','butane','pentane','hexane','heptane','octane','nonane','decane','undecane','dodecane','tridecane','tetradecane','pentadecane','hexadecane','2,2,4-trimethylpentane','2,2,4,4,6,8,8-heptamethylnonane','2-methylbutane']:
        new_NMRData.append(Mol)

with open('SurfTTrainingData_IsoAlkanes.json', 'w') as outfile:
    json.dump(new_NMRData, outfile)
