import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import json
from rdkit.Chem import Draw

# Load data
with open('SurfTTrainingData.json') as f:
    data = json.load(f)
#load each molecule and draw image
for i in data:
    Inchi = i['INChI']
    mol = Chem.MolFromInchi(Inchi)
    img = Draw.MolToImage(mol)

    smiles = Chem.MolToSmiles(mol)
    img.save(f'Images/{smiles}.png')
    print(smiles, Inchi)
