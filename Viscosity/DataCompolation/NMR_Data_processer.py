import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
from Util import grab_histogram_13C
# load the data from json file
with open('HydrocarbonData.json', 'r') as infile:
    HydrocarbonData = json.load(infile)


def ShiftProcessing(data, binsize=1, maxshift=250):
    # Ensure data is a numpy array
    #bin
    shift_list=grab_histogram_13C(data)
    return shift_list


# Assuming HydrocarbonData is already defined and loaded
print(len(HydrocarbonData))

# Run shift processing on all molecules
HydrocarbonData2 = HydrocarbonData.copy()
for element in HydrocarbonData2:
    # Process 13C shifts
    C_shift_list = element.get('13C_shift', [])
    if len(C_shift_list) == 0:
        HydrocarbonData.remove(element)
        print('Removed (no 13C shifts):', element['MolName'])
        continue


    # Process shifts
    #C_shift_list = ShiftProcessing(C_shift_list)
    # Check for NaN, empty lists, inf, or only 0 in processed lists
    if (np.isnan(C_shift_list).any() or len(C_shift_list) == 0 or np.isinf(C_shift_list).any() or np.all(C_shift_list == 0)):
        HydrocarbonData.remove(element)
        continue

    # Update the dictionary with processed shifts
    element['13C_shift'] = C_shift_list

print(len(HydrocarbonData))
# Save the processed data as a JSON file
with open('HydrocarbonDataProcessed.json', 'w') as outfile:
    json.dump(HydrocarbonData, outfile)


