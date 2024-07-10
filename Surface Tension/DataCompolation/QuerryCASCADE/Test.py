import numpy as np
from rdkit import Chem
from CASCADE import CASCADE_C, CASCADE_H
import matplotlib.pyplot as plt

def count_ch_groups(smiles):
    # Parse the SMILES string using RDKit
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    
    # Initialize counters for each group
    ch3_count = 0
    ch2_count = 0
    ch_count = 0
    
    # Iterate over each carbon atom in the molecule
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == 'C':
            # Count the number of hydrogen atoms attached to this carbon
            num_hydrogens = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H')
            
            # Classify the carbon group
            if num_hydrogens == 3:
                ch3_count += 1
            elif num_hydrogens == 2:
                ch2_count += 1
            elif num_hydrogens == 1:
                ch_count += 1
    
    # Calculate the total number of groups
    total_groups = ch3_count + ch2_count + ch_count
    
    # Normalize the counts
    ch3_ratio = ch3_count / total_groups
    ch2_ratio = ch2_count / total_groups
    ch_ratio = ch_count / total_groups

    return ch3_ratio, ch2_ratio, ch_ratio

def count_ch_groups_1H(spectra):
    #intergrate the 1H NMR spectrum to get the number of CH3, CH2 and CH groups
    #ch3 .9 - 1.2 ppm
    #ch2 1.2 - 1.5 ppm
    #ch 1.5 - 2.0 ppm
    #integrate the area under the curve for each range
    ch3 = np.trapz(spectra[(spectra[:,0]>.9) & (spectra[:,0]<1.2)][:,1])
    ch2 = np.trapz(spectra[(spectra[:,0]>1.2) & (spectra[:,0]<1.5)][:,1])
    ch = np.trapz(spectra[(spectra[:,0]>1.5) & (spectra[:,0]<2.0)][:,1])
    total = ch3 + ch2 + ch
    return ch3/total, ch2/total, ch/total

def count_ch_groups_13C(spectra):
    #intergrate the 13C NMR spectrum to get the number of CH3, CH2 and CH groups
    #ch3 5 - 20 ppm
    #ch2 20 - 30 ppm
    #ch 30 - 50 ppm
    #integrate the area under the curve for each range
    ch3 = np.trapz(spectra[(spectra[:,0]>0) & (spectra[:,0]<35)][:,1])
    ch2 = np.trapz(spectra[(spectra[:,0]>35) & (spectra[:,0]<55)][:,1])
    ch = np.trapz(spectra[(spectra[:,0]>25) & (spectra[:,0]<55)][:,1])
    total = ch3 + ch2 + ch
    return ch3/total, ch2/total, ch/total

# Test the function
#predict hexane only and save the 13C and 1H NMR spectra
C_pred = CASCADE_C()
H_pred = CASCADE_H()

x_c, y_c = C_pred.get_spectrum('CCCCCC', show=True)
x_h, y_h = H_pred.get_spectrum('CCCCCC', show=True)

np.savetxt('hexane_13C.csv', np.column_stack((x_c, y_c)), delimiter=',')
np.savetxt('hexane_1H.csv', np.column_stack((x_h, y_h)), delimiter=',')

#plot the 13C and 1H NMR spectra
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(x_c,y_c)
plt.title('13C NMR Spectrum of n-hexane')
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.gca().invert_xaxis()
plt.subplot(1,2,2)
plt.plot(x_h,y_h)
plt.title('1H NMR Spectrum of n-hexane')
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.gca().invert_xaxis()
plt.show()



#nalkanes from 6 - 20
names_ls = ['hexane', 'heptane', 'octane', 'nonane', 'decane', 'undecane', 'dodecane', 'tridecane', 'tetradecane', 'pentadecane', 'hexadecane', 'heptadecane', 'octadecane', 'nonadecane', 'eicosane']
smiles_ls = ['CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC', 'CCCCCCCCCCC', 'CCCCCCCCCCCC', 'CCCCCCCCCCCCC', 'CCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCCC']

results = []
for name, smiles in zip(names_ls, smiles_ls):
    #C6H14
    CPred = CASCADE_C()
    HPred = CASCADE_H()
    x_c,y_c=CPred.get_spectrum(smiles,show=True)
    x_h,y_h=HPred.get_spectrum(smiles,show=True)

    print(count_ch_groups(smiles))
    print(count_ch_groups_1H(np.column_stack((x_h,y_h))))
    print(count_ch_groups_13C(np.column_stack((x_c,y_c))))
    results.append((name, count_ch_groups(smiles), count_ch_groups_1H(np.column_stack((x_h,y_h))), count_ch_groups_13C(np.column_stack((x_c,y_c)))))

#save the results
import pandas as pd
#unroll tuples
results = [(name, *ch, *ch1h, *ch13c) for name, ch, ch1h, ch13c in results]
df = pd.DataFrame(results, columns=['Name', 'CH3', 'CH2', 'CH', 'CH3_1H', 'CH2_1H', 'CH_1H', 'CH3_13C', 'CH2_13C', 'CH_13C'])
df.to_csv('results.csv', index=False)



plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(x_c,y_c)
plt.title('13C NMR Spectrum of n-hexane')
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.gca().invert_xaxis()
plt.subplot(1,2,2)
plt.plot(x_h,y_h)
plt.title('1H NMR Spectrum of n-hexane')
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.gca().invert_xaxis()
plt.show()

#ifft of the 1H NMR spectrum
from scipy.fft import ifft

y = y_h
yf = ifft(y)
#cut in half
yf = yf[:len(yf)//2]

plt.figure(figsize=(12,8))
plt.plot(yf)
plt.title('Inverse FFT of 1H NMR Spectrum of n-hexane')
plt.show()

