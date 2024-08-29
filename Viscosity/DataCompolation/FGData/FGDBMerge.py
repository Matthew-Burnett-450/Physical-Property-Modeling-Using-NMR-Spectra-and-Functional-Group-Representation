from FGGenorators import find_smiles_patterns
import json

# Load TDE data
with open(r'DataCompolation\ThermoData\NISTData.json', 'r') as f:
    data = json.load(f)

#iterate through the data smiles and add the fg data to the dictionary
remove_idx = []

for i in range(len(data)):
    smiles = data[i]['smiles']
    _,Values = find_smiles_patterns(smiles)
    data[i]['FG'] = Values
    #if y or t is empty remove the data
    if len(data[i]['y']) == 0 or len(data[i]['t']) == 0:
        remove_idx.append(i)

data = [data[i] for i in range(len(data)) if i not in remove_idx]    

# Save the updated data to FGTrainingDB.json
with open(r'DataCompolation\FGTrainingDB.json', 'w') as f:
    json.dump(data, f, indent=4)
