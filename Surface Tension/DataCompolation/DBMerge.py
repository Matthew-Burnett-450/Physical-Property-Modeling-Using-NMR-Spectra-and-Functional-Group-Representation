import numpy as np 
import json

#load both json files
with open(r'NMRData/SpectraDB.json', 'r') as f:
    data = json.load(f)
    db_1H = data['1H']
    db_13C = data['13C']

with open(r'ThermoData/NISTData.json', 'r') as f:
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
# If there are entries without TDE data, remove them. 
# Check for empty t, y, and Tc. If there is a KeyError, remove the entry.

for spectra in combined_db['1H'][:]:
    try:
        if spectra.get('t') == [] or spectra.get('y') == [] or spectra.get('Tc') == []:
            combined_db['1H'].remove(spectra)
    except KeyError:
        combined_db['1H'].remove(spectra)

print('1H Finished')

for spectra in combined_db['13C'][:]:
    try:
        if spectra.get('t') == [] or spectra.get('y') == [] or spectra.get('Tc') == []:
            combined_db['13C'].remove(spectra)
    except KeyError:
        combined_db['13C'].remove(spectra)

# If intensity or frequency is None, or if there is a KeyError, remove the entry

for spectra in combined_db['1H'][:]:
    try:
        if None in spectra.get('t', []) or None in spectra.get('y', []):
            combined_db['1H'].remove(spectra)
    except KeyError:
        combined_db['1H'].remove(spectra)

for spectra in combined_db['13C'][:]:
    try:
        if None in spectra.get('t', []) or None in spectra.get('y', []):
            combined_db['13C'].remove(spectra)
    except KeyError:
        combined_db['13C'].remove(spectra)


#check if Intensity or Frequency is None
for spectra in combined_db['1H'][:]:
    try:
        if spectra.get('Intensity') == None or spectra.get('Frequency (ppm)') == None:
            combined_db['1H'].remove(spectra)
    except KeyError:
        combined_db['1H'].remove(spectra)
    
for spectra in combined_db['13C'][:]:
    try:
        if spectra.get('Intensity') == None or spectra.get('Frequency (ppm)') == None:
            combined_db['13C'].remove(spectra)
    except KeyError:
        combined_db['13C'].remove(spectra)

#if y or t keys dont exist remove the entry
for spectra in combined_db['1H'][:]:
    try:
        if spectra.keys() != {'Smiles', 'Frequency (ppm)', 'Intensity', 't', 'y', 'Tc'}:
            combined_db['1H'].remove(spectra)
    except KeyError:
        combined_db['1H'].remove(spectra)

for spectra in combined_db['13C'][:]:
    try:
        if spectra.keys() != {'Smiles', 'Frequency (ppm)', 'Intensity', 't', 'y', 'Tc'}:
            combined_db['13C'].remove(spectra)
    except KeyError:
        combined_db['13C'].remove(spectra)
        

#save the combined db dont update the file clear and write
with open('TrainingDB.json', 'w') as f:
    json.dump(combined_db, f, indent=4)

print("Combined database saved to TrainingDB.json")  # Confirm the save operation
