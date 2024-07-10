import json
import numpy as np
import matplotlib.pyplot as plt

# Load Data
with open('ViscTrainingData.json') as f:
    data = json.load(f)

#grab names,r2 scores, and params and put in a array for csv
names = []
MAPE=[]


for i in range(len(data)):
    names.append(data[i]['MolName'])
    MAPE.append(data[i]['MAPE'])


names = np.array(names)
MAPE = np.array(MAPE)

X = np.array([names, MAPE]).T


# Save Data
np.savetxt('ViscTrainingData.txt', X, delimiter='\t', fmt='%s', header='Name', comments='')

import pandas as pd

# Define column names
column_names = ['Name', 'MAPE']

# Reload the data with the correct column names
data = pd.read_csv('ViscTrainingData.txt', delimiter='\t', names=column_names, skiprows=1)

# Sort the fuels into categories with additional checks for methyl and alkenes
nalkanes = []
isoalkanes = []
cycloalkanes = []
aromatics = []
nalkenes = []
isoalkenes = []
MAPE_nalkanes = []
MAPE_isoalkanes = []
MAPE_cycloalkanes = []
MAPE_aromatics = []
MAPE_nalkenes = []
MAPE_isoalkenes = []

# Define the categories based on common naming conventions
for name,MAPE in zip(data['Name'],data['MAPE']):
    lower_name = name.lower()
    if 'cyclo' in lower_name:
        cycloalkanes.append(name)
        MAPE_cycloalkanes.append(MAPE)
    elif 'benzene' in lower_name or 'toluene' in lower_name or 'xylene' in lower_name or 'naphthalene' in lower_name:
        aromatics.append(name)
        MAPE_aromatics.append(MAPE)
    elif any(prefix in lower_name for prefix in ['iso', 'neo', 'methyl']):
        if 'ene' in lower_name:
            isoalkenes.append(name)
            isoalkanes.append(name)
        else:
            isoalkanes.append(name)
            MAPE_isoalkanes.append(MAPE)
    elif 'ene' in lower_name:
        nalkenes.append(name)
        MAPE_nalkenes.append(MAPE)
    else:
        nalkanes.append(name)
        MAPE_nalkanes.append(MAPE)

MAPE_nalkanes = np.mean(MAPE_nalkanes)
MAPE_isoalkanes = np.mean(MAPE_isoalkanes)
MAPE_cycloalkanes = np.mean(MAPE_cycloalkanes)
MAPE_aromatics = np.mean(MAPE_aromatics)
MAPE_nalkenes = np.mean(MAPE_nalkenes)
MAPE_isoalkenes = np.mean(MAPE_isoalkenes)


# Prepare data for pie chart
categories = [f'n-alkanes, MAPE {MAPE_nalkanes:.2f}%', f'iso-alkanes, MAPE {MAPE_isoalkanes:.2f}%',f'cycloalkanes, MAPE {MAPE_cycloalkanes:.2f}%', f'aromatics, MAPE {MAPE_aromatics:.2f}%']
counts = [len(nalkanes), len(isoalkanes), len(cycloalkanes), len(aromatics)]
total=sum(counts)


# Plot pie chart
plt.figure(figsize=(8, 5))
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140)
#add text in the top length n=total
plt.text(.9,.9,f'n = {total}',fontsize=12, ha='center')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save the plot
plt.savefig('fuel_distribution_pie_chart.png')

# Show the plot
plt.show()
