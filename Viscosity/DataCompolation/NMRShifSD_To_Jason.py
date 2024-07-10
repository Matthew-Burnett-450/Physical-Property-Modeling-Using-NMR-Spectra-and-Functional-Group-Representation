import numpy as np
import json

DataSet = []
Mol_dictionary = {}
C_shift_list = []
H_shift_list = []
firstiter = 0

with open(r'.\NMRData\nmrshiftdb2withsignals.sd', 'r', encoding='utf-8') as file:
    lastline = ''
    for line in file:
        line = line.strip(str('\n'))
        if firstiter == 0:
            if lastline.startswith('$$$$'):
                firstiter = 1
            else:
                lastline = line
                continue
        
        # Check for the start of a new molecule
        if lastline.startswith('$$$$'):
            if Mol_dictionary != {}:
                Mol_dictionary['13C_shift'] = C_shift_list
                Mol_dictionary['1H_shift'] = H_shift_list
                C_shift_list = []
                H_shift_list = []
                DataSet.append(Mol_dictionary)
                Mol_dictionary = {}
            Mol_dictionary['MolName'] = line

        if lastline.startswith('> <INChI key>'):
            Mol_dictionary['INChI key'] = line
        if lastline.startswith('> <INChI>'):
            Mol_dictionary['INChI'] = line

        if lastline.startswith('> <nmrshiftdb2 ID>'):
            Mol_dictionary['NMRShiftDB ID'] = line
        if lastline.startswith('> <Spectrum 13C'):
            for shift in line.split('|')[:-1]:
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                C_shift_list.append([shift_val, shift_idx])

        # Addition: Handle 1H shifts similarly
        if lastline.startswith('> <Spectrum 1H'):
            for shift in line.split('|')[:-1]:
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                H_shift_list.append([shift_val, shift_idx])

        lastline = line
        print(len(DataSet))

# Display part of the dataset to check
print(DataSet[:1])  # Print the first element to check

# Save the data as a JSON file
with open('NMRData.json', 'w') as outfile:
    json.dump(DataSet, outfile)
