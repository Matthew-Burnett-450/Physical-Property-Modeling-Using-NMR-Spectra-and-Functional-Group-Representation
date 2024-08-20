import numpy as np 
import json

#load both json files
with open('NMRData/SpectraDB.json', 'r') as f:
    data = json.load(f)
    db_1H = data['1H']
    db_13C = data['13C']

with open('ThermoData/NISTData.json', 'r') as f:
    data = json.load(f)
    db_TDE = data

#combine the two databases spectra db is a dict with a list of dicts for 1H and 13C and TDE db is a list of dicts
#each dict in the list has a name,smiles,t,y,Tc
#the combined db will be a dict with a list of dicts for 1H and 13C and TDE
#each dict in the list will have a name,smiles,t,y,Tc and 1H and 13C spectra
combined_db = {"1H": [], "13C": []}

for entry in db_1H:
    name = entry['Smiles']
    for entry2 in db_TDE:
        if entry2['name'] == name:
            entry['t'] = entry2['t']
            entry['y'] = entry2['y']
            entry['Tc'] = entry2['Tc']
            break
    combined_db['1H'].append(entry)

for entry in db_13C:
    name = entry['Smiles']
    for entry2 in db_TDE:
        if entry2['name'] == name:
            entry['t'] = entry2['t']
            entry['y'] = entry2['y']
            entry['Tc'] = entry2['Tc']
            break
    combined_db['13C'].append(entry)

#if there are entries without TDE data, remove them check for empty t and y and tc
combined_db['1H'] = [entry for entry in combined_db['1H'] if entry['t'] and entry['y'] and entry['Tc']]
combined_db['13C'] = [entry for entry in combined_db['13C'] if entry['t'] and entry['y'] and entry['Tc']]

#save the combined db
with open('TrainingDB.json', 'w') as f:
    json.dump(combined_db, f, indent=4)

