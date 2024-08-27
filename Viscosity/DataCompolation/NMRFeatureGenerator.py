import numpy as np
import json
from scipy import signal
from sklearn.model_selection import train_test_split

def get_data():

    # Load the data
    with open('DataCompolation\TrainingDB.json', 'r') as f:
        data = json.load(f)
        db_1H = data['1H']

    Names=[]
    idx_to_remove = []
    # Decimate the 1H spectra to 100 points
    for i in range(len(db_1H)):
        #if intensity or frequency is empty remove the entry after the loop
        x = db_1H[i]['Frequency (ppm)']
        spectra_intensity = db_1H[i]['Intensity']
        x = signal.decimate(x, 10)
        spectra_intensity = signal.decimate(spectra_intensity, 10) 
        db_1H[i]['Frequency (ppm)'] = x
        db_1H[i]['Intensity'] = spectra_intensity / np.max(spectra_intensity)
        Names.append(db_1H[i]['Smiles'])

    print(len(Names))
    # Remove entries with empty intensity or frequency
    db_1H = [db_1H[i] for i in range(len(db_1H)) if i not in idx_to_remove]
    print(len(db_1H))

    #save Names
    np.savetxt('Names.txt', Names, fmt='%s', delimiter='\t', header='Names', comments='')

    # Fit linear curve to y vs t and save the coefficients to the db y = Ax + B
    T = []
    Y = []
    for i in range(len(db_1H)):
        print(f"Processing spectrum index: {i}")
        t = db_1H[i]['t']
        y = db_1H[i]['y']
        Y.append(y)
        T.append(t)
        print(f"Data length - t: {len(t)}, y: {len(y)}")
        
        t = np.array(t)

        # Fit a linear curve to the data
        coeffs = np.polyfit(1/t, y, 1)
        A, B = coeffs[0], coeffs[1]
        
        # Save the coefficients to the database
        db_1H[i]['A'] = float(A)
        db_1H[i]['B'] = float(B)
        
        # Calculate R-squared to evaluate the fit
        y_fit = np.polyval(coeffs, t)
        residuals = y - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"R-squared for spectrum index {i}: {r2:.4f}")

    # Prepare features and target values for regression
    features = []
    target_A = []
    target_B = []

    for i in range(len(db_1H)):
        features.append(db_1H[i]['Intensity'])
        # Assuming 'A' and 'B' are saved in the database for each spectrum
        target_A.append(db_1H[i]['A'])
        target_B.append(db_1H[i]['B'])

    features = np.array(features)
    target_A = np.array(target_A)
    target_B = np.array(target_B)

    # Split data into training and testing sets
    X_train, X_test, y_train_A, y_test_A, y_train_B, y_test_B = train_test_split(features, target_A, target_B, test_size=0.2, random_state=42)
    #validation from test
    X_test, X_val, y_test_A, y_val_A, y_test_B, y_val_B = train_test_split(X_test, y_test_A, y_test_B, test_size=0.5, random_state=42)
    return X_train, X_test, X_val, y_train_A, y_test_A, y_val_A, y_train_B, y_test_B, y_val_B



def get_data_test():

    # Load the data
    with open('DataCompolation\TrainingDB.json', 'r') as f:
        data = json.load(f)
        db_1H = data['1H']

    Names=[]
    idx_to_remove = []
    # Decimate the 1H spectra to 100 points
    for i in range(len(db_1H)):
        #if intensity or frequency is empty remove the entry after the loop
        x = db_1H[i]['Frequency (ppm)']
        spectra_intensity = db_1H[i]['Intensity']
        x = signal.decimate(x, 10)
        spectra_intensity = signal.decimate(spectra_intensity, 10) 
        db_1H[i]['Frequency (ppm)'] = x
        db_1H[i]['Intensity'] = spectra_intensity / np.max(spectra_intensity)
        Names.append(db_1H[i]['Smiles'])

    print(len(Names))
    # Remove entries with empty intensity or frequency
    db_1H = [db_1H[i] for i in range(len(db_1H)) if i not in idx_to_remove]
    print(len(db_1H))

    #save Names
    np.savetxt('Names.txt', Names, fmt='%s', delimiter='\t', header='Names', comments='')

    # Fit linear curve to y vs t and save the coefficients to the db y = Ax + B
    T = []
    Y = []
    for i in range(len(db_1H)):
        print(f"Processing spectrum index: {i}")
        t = db_1H[i]['t']
        y = db_1H[i]['y']
        Y.append(y)
        T.append(t)
        print(f"Data length - t: {len(t)}, y: {len(y)}")
        
    # Prepare features and target values for regression
    features = []

    for i in range(len(db_1H)):
        features.append(db_1H[i]['Intensity'])

    features = np.array(features)


    return x,Y,T,Names,features

