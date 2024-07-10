from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import json
from rdkit.Chem import Draw
import networkx as nx
def mol_to_graph(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atom=atom)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G

def is_valid_molecule(inchi):
    mol = Chem.MolFromInchi(inchi, sanitize=True)
    if mol is None:
        return False
    if not all(atom.GetAtomicNum() in [1,6] for atom in mol.GetAtoms()):
        return False
    num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    if num_carbons < 5:
        return False
    if num_carbons > 16:
        return False

    if any(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return False
    for bond in mol.GetBonds():
        if bond.GetStereo() not in [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY]:
            return False
    #remove triple bonds
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE or bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            return False
    
    return True


def get_functional_groups(mol):
    # Define SMARTS patterns for each functional group
    patterns = {
        'CH3': Chem.MolFromSmarts('[CX4H3]'),
        'CH': Chem.MolFromSmarts('[CX4H1]'),
        'Quaternary_C': Chem.MolFromSmarts('[CX4H0]'),
        'Benzyl_Rings': Chem.MolFromSmarts('c1ccccc1[CH2]'),
        'C=C': Chem.MolFromSmarts('C=C')
    }
    
    # SMARTS patterns for specific CH2 groups
    ch2_pattern = Chem.MolFromSmarts('[CH2]')
    cyclo_ch2_pattern = Chem.MolFromSmarts('[CH2]@*')

    # Count occurrences of general patterns
    functional_groups = {key: len(mol.GetSubstructMatches(pattern)) for key, pattern in patterns.items()}
    
    # Specific counts for CH2 groups
    ch2_matches = mol.GetSubstructMatches(ch2_pattern, useQueryQueryMatches=True)
    cyclo_ch2_matches = mol.GetSubstructMatches(cyclo_ch2_pattern, useQueryQueryMatches=True)
    #convert to ((0,1),) format to ((0,),)
    cyclo_ch2_matches = [tuple([x[0]]) for x in cyclo_ch2_matches]
    cyclo_ch2_count = len(cyclo_ch2_matches)

    # Grab InChI key
    inchi = Chem.MolToInchi(mol)
    if inchi == "InChI=1S/C6H12/c1-2-4-6-5-3-1/h1-6H2":
        print('True')
        img = Draw.MolToImage(mol)
        img.show()

    # Convert molecule to graph
    G = mol_to_graph(mol)

    # Determine CH2 chains
    cyclo_ch2_atoms = set(match[0] for match in cyclo_ch2_matches)
    ch2_atoms = set(match[0] for match in ch2_matches if match[0] not in cyclo_ch2_atoms)
    ch2_chain_count = 0
    ch2_in_chains = set()

    def traverse_chain(atom_idx, visited):
        stack = [atom_idx]
        chain = []
        while stack:
            current_idx = stack.pop()
            if current_idx in visited:
                continue
            visited.add(current_idx)
            chain.append(current_idx)
            neighbors = list(G.neighbors(current_idx))
            for neighbor in neighbors:
                if neighbor in ch2_atoms and neighbor not in visited:
                    stack.append(neighbor)
        return chain

    visited = set()
    for atom_idx in ch2_atoms:
        if atom_idx not in visited:
            chain = traverse_chain(atom_idx, visited)
            if len(chain) > 1:
                ch2_chain_count += len(chain)  # Count each CH2 group in the chain
                ch2_in_chains.update(chain)

    # General CH2 count: subtract CH2 in chains and cyclo CH2
    general_ch2_count = len(ch2_matches) - len(ch2_in_chains) - cyclo_ch2_count
    if general_ch2_count < 0:
        general_ch2_count = 0

    # Update the functional groups dictionary
    functional_groups.update({
        'CH2': general_ch2_count,
        'CH2_Chains': ch2_chain_count,
        'Cyclo_CH2': cyclo_ch2_count
    })
    
    #calculate molecular weight and add it to the dictionary
    functional_groups['MolecularWeight'] = rdMolDescriptors.CalcExactMolWt(mol)
    
    return functional_groups

# Load NMR data
with open('NMRData.json', 'r') as infile:
    NMRData = json.load(infile)

# Filter and characterize molecules
HydrocarbonData = []
for sample in NMRData:
    if is_valid_molecule(sample['INChI']):
        mol = Chem.MolFromInchi(sample['INChI'], sanitize=True)
        sample['functional_groups'] = get_functional_groups(mol)
        HydrocarbonData.append(sample)

print(len(HydrocarbonData))

# Save the data with functional group information
with open('HydrocarbonData.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)
