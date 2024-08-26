import numpy as np
import pubchempy as pcp

# Read the entire file into a list of lines
with open('InitFuelList/SMI_List.txt', 'r') as file:
    lines = file.readlines()

# Initialize lists for CAS and SMILES
CAS = []
Smiles = []

# Process each line to extract CAS and SMILES
for line in lines:
    parts = line.split()
    if len(parts) >= 3:  # Ensure there are enough parts to avoid index errors
        CAS.append(parts[1])  # CAS number is in the second column
        Smiles.append("".join(parts[2:]))  # SMILES starts from the third column onward

# Convert lists to numpy arrays
CAS = np.array(CAS)
Smiles = np.array(Smiles)

# Print initial lengths
print(len(Smiles))
print(len(CAS))

# Define the allowed characters
allowed_chars = set('CH()=123456789')

# Create a mask for valid SMILES
valid_mask = [all(char in allowed_chars for char in Smi) for Smi in Smiles]

# Apply the mask to filter out unwanted rows
CAS_filtered = CAS[valid_mask]
Smiles_filtered = Smiles[valid_mask]

# Print final lengths
print(len(Smiles_filtered))
print(len(CAS_filtered))

#cas to name with pubchempy
names = []
for cas in CAS_filtered:
    print(f"Searching for CAS: {cas}")
    try:
        compound = pcp.get_compounds(cas, 'name')
        names.append(compound[0].iupac_name)
    except:
        names.append("Unknown")
        print(f"Failed to find name for CAS: {cas}")

#if the name is unknown, delete the entry
CAS_filtered = np.array([CAS_filtered[i] for i in range(len(names)) if names[i] != "Unknown"])
Smiles_filtered = np.array([Smiles_filtered[i] for i in range(len(names)) if names[i] != "Unknown"])
names = np.array([name for name in names if name != "Unknown"])

#if Names is None, remove the entry
CAS_filtered = np.array([CAS_filtered[i] for i in range(len(names)) if names[i] is not None])
Smiles_filtered = np.array([Smiles_filtered[i] for i in range(len(names)) if names[i] is not None])
names = np.array([name for name in names if name is not None])



# Combine filtered CAS and Smiles into a new array and save it
CAS_Smiles = np.column_stack((CAS_filtered, Smiles_filtered, names))
np.savetxt('InitFuelList/CAS_Smiles.txt', CAS_Smiles, fmt='%s', delimiter='\t', header='CAS,SMILES,Name', comments='')

print("CAS and SMILES saved to CAS_Smiles.csv")
