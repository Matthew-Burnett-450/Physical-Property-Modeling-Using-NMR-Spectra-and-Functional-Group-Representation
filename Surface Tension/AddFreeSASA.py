from rdkit.Chem import rdFreeSASA
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import matplotlib.pyplot as plt

# Load the data from the JSON file
with open('SurfTTrainingData.json') as f:
    data = json.load(f)
FreeSASA_list = []
# Iterate through the data and calculate FreeSASA for each molecule
for i in range(len(data)):
        # Load molecule from InChI
        mol = Chem.MolFromInchi(data[i]['INChI'])
        if mol is None:
            print(f"Could not parse InChI: {data[i]['INChI']}")
            continue

        # Add hydrogen atoms
        hmol1 = Chem.AddHs(mol)
        
        # Generate conformers
        AllChem.EmbedMolecule(hmol1, randomSeed=42)
        AllChem.UFFOptimizeMolecule(hmol1)

        # Classify atoms and calculate FreeSASA
        radii = rdFreeSASA.classifyAtoms(hmol1)
        FreeSASA = rdFreeSASA.CalcSASA(hmol1, radii)
        
        # Print and store the calculated FreeSASA
        print(FreeSASA)
        FreeSASA_list.append(FreeSASA)
        data[i]['FreeSASA'] = FreeSASA
        #plot FreeSASA vs Tc
        plt.scatter(data[i]['Tc_True'],FreeSASA)
plt.xlabel('Tc')
plt.ylabel('FreeSASA')
plt.title('FreeSASA vs Tc')
plt.show()




# Optionally, save the updated data back to a JSON file
with open('SurfTTrainingData.json', 'w') as f:
    json.dump(data, f, indent=4)
