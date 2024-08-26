import numpy as np
from NMRSIM import NMR1HSpectraGenerator as NMR1H
from NMRSIM import NMR13CSpectraGenerator as NMR13C

import concurrent.futures
import json

#load SpectrumData.json to get lw and dl for 13C and 1H
with open('NMRData\RealFuelData\SpectrumData.json', 'r') as f:
    data = json.load(f)
H_lw = data['1H']['PeakWidth']
H_dl = data['1H']['DataLength']
H_ppm = data['1H']['ppm']

C_lw = data['13C']['PeakWidth']
C_dl = data['13C']['DataLength']
C_ppm = data['13C']['ppm']

def generate_spectra(smiles):
    y_1H = NMR1H.predict(smiles,H_ppm,H_lw)
    y_13C = NMR13C.predict(smiles,C_ppm,C_lw)
    return y_1H,y_13C

# Load the data
data = np.loadtxt(r'InitFuelList\NameCanadites.txt', dtype=str, delimiter='\t', skiprows=1)
CAS = data[:, 0]
Smiles = data[:, 1]
Names = data[:, 2]

# Load the database once before processing
db_1H = NMR1H.load_db('SpectraDB.json')
db_13C = NMR13C.load_db('SpectraDB.json')

results = []

# Use ThreadPoolExecutor to parallelize the task
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(generate_spectra, smiles,): name for name, smiles in zip(Names, Smiles)}
    
    for future in concurrent.futures.as_completed(futures):
        name = futures[future]
        try:
            y_1H,y_13C = future.result()
            results.append((name, H_ppm,C_ppm, y_1H,y_13C))
            print(f"Generated spectra for {name}")
        except Exception as e:
            print(f"An error occurred for {name}: {e}")

# Append all results to the database
for name, x_1H,x_13C, y_1H,y_13C in results:
    NMR1H.add_entry_to_db(db_1H, name, x_1H, y_1H)
    NMR13C.add_entry_to_db(db_13C, name, x_13C, y_13C)
    print(f"Added {name} to the database.")

# Save the updated database once at the end
db = {"1H": db_1H['1H'], "13C": db_13C['13C']}

NMR1H.save_db(db, 'NMRData\SpectraDB.json')
print("All spectra saved to the database.")