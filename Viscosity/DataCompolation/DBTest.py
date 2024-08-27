import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal as sig
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
with open('TrainingDB.json', 'r') as f:
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
    db_1H[i]['Intensity'] = spectra_intensity
    Names.append(db_1H[i]['Smiles'])

print(len(Names))
# Remove entries with empty intensity or frequency
db_1H = [db_1H[i] for i in range(len(db_1H)) if i not in idx_to_remove]
print(len(db_1H))

#save Names
np.savetxt('Names.txt', Names, fmt='%s', delimiter='\t', header='Names', comments='')

# Peak finding
peaks_x = []
peaks_y = []

for i in range(len(db_1H)):
    x = db_1H[i]['Frequency (ppm)']
    y = db_1H[i]['Intensity']
    peaks, _ = sig.find_peaks(y)
    if len(peaks) > 0:
        peaks_x.extend(x[peaks])
        peaks_y.extend(y[peaks])

# Convert peak positions to a 2D array for clustering
peaks_array = np.array(peaks_x).reshape(-1, 1)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(peaks_array)

# Make unique colors and marker combinations
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'D']
color_marker = [(color, marker) for color in colors for marker in markers]
np.random.shuffle(color_marker)

# Print how many clusters were found
print(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")  # Subtract 1 if there are noise points

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(len(peaks_x)):
    plt.scatter(peaks_x[i], peaks_y[i], c=[color_marker[labels[i] % len(color_marker)][0]], marker=color_marker[labels[i] % len(color_marker)][1])
# Plot real spectra with low alpha
for i in range(len(db_1H)):
    plt.plot(db_1H[i]['Frequency (ppm)'], db_1H[i]['Intensity'], label=db_1H[i]['Smiles'], alpha=0.1)
plt.xlabel('Ppm')
plt.ylabel('Peak Intensity')
plt.title('KMeans Clustering of Peak Positions')
plt.show()

# Using intensity to calculate what groups a spectrum belongs to and in what relative amounts
# Create a mapping from peak index to cluster label
peak_labels = dict(zip(range(len(peaks_x)), labels))

# Assign clusters to each spectrum and calculate relative amounts
spectrum_cluster_info = []

for i in range(len(db_1H)):
    x = db_1H[i]['Frequency (ppm)']
    y = db_1H[i]['Intensity']
    peaks, _ = sig.find_peaks(y)
    
    if len(peaks) > 0:
        spectrum_peaks_x = x[peaks]
        spectrum_peaks_y = y[peaks]
        
        # Create an array of peak positions for this spectrum
        spectrum_peaks_array = np.array(spectrum_peaks_x).reshape(-1, 1)
        
        # Find clusters for this spectrum's peaks
        spectrum_peak_indices = [np.where(peaks_array[:, 0] == pos)[0][0] for pos in spectrum_peaks_x]
        spectrum_labels = [peak_labels[index] for index in spectrum_peak_indices]
        
        # Count occurrences of each cluster
        cluster_counts = defaultdict(int)
        for label in spectrum_labels:
            cluster_counts[label] += 1
        
        # Calculate relative amounts
        total_peaks = len(spectrum_labels)
        relative_amounts = {cluster: count / total_peaks for cluster, count in cluster_counts.items()}
        
        spectrum_cluster_info.append({
            'spectrum_index': i,
            'cluster_counts': cluster_counts,
            'relative_amounts': relative_amounts,
            'SumInt': np.sum(y)
        })

# Print or analyze the results
for info in spectrum_cluster_info:
    print(f"Spectrum {info['spectrum_index']}:")
    print("Cluster Counts:", info['cluster_counts'])
    print("Relative Amounts:", info['relative_amounts'])
    print()

# Fit linear curve to y vs t and save the coefficients to the db y = Ax + B
for i in range(len(db_1H)):
    print(f"Processing spectrum index: {i}")
    t = db_1H[i]['t']
    y = db_1H[i]['y']
    
    print(f"Data length - t: {len(t)}, y: {len(y)}")
    
    # Fit a linear curve to the data
    coeffs = np.polyfit(t, y, 1)
    A, B = coeffs[0], coeffs[1]
    
    # Save the coefficients to the database
    db_1H[i]['A'] = float(A)
    db_1H[i]['B'] = float(B)
    db_1H[i]['SumInt'] = float(np.sum(y))
    
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

for info in spectrum_cluster_info:
    relative_weights = [info['relative_amounts'].get(cluster, 0) for cluster in range(len(set(labels)))]
    features.append(relative_weights + [info['SumInt']])
    # Assuming 'A' and 'B' are saved in the database for each spectrum
    target_A.append(db_1H[info['spectrum_index']]['A'])
    target_B.append(db_1H[info['spectrum_index']]['B'])

features = np.array(features)
target_A = np.array(target_A)
target_B = np.array(target_B)

# Split data into training and testing sets
X_train, X_test, y_train_A, y_test_A = train_test_split(features, target_A, test_size=0.2, random_state=42)
X_train, X_test, y_train_B, y_test_B = train_test_split(features, target_B, test_size=0.2, random_state=42)

# Train regression models
model_A = LinearRegression()
model_A.fit(X_train, y_train_A)

model_B = LinearRegression()
model_B.fit(X_train, y_train_B)

# Predict and evaluate the models
y_pred_A = model_A.predict(X_test)
y_pred_B = model_B.predict(X_test)

mse_A = mean_squared_error(y_test_A, y_pred_A)
mse_B = mean_squared_error(y_test_B, y_pred_B)

print(f"Mean Squared Error for A: {mse_A}")
print(f"Mean Squared Error for B: {mse_B}")

# Calculate R^2 of test predictions
r2_A = model_A.score(X_test, y_test_A)
r2_B = model_B.score(X_test, y_test_B)
 
print(f"R^2 for A: {r2_A}")
print(f"R^2 for B: {r2_B}")

# Parity plot 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test_A, y_pred_A)
plt.plot([y_test_A.min(), y_test_A.max()], [y_test_A.min(), y_test_A.max()], 'k--', lw=2)
plt.xlabel('True A')
plt.ylabel('Predicted A')
plt.title('Parity Plot for A')

plt.subplot(1, 2, 2)
plt.scatter(y_test_B, y_pred_B)
plt.plot([y_test_B.min(), y_test_B.max()], [y_test_B.min(), y_test_B.max()], 'k--', lw=2)
plt.xlabel('True B')
plt.ylabel('Predicted B')
plt.title('Parity Plot for B')

plt.show()

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_A = mean_absolute_percentage_error(y_test_A, y_pred_A)
mape_B = mean_absolute_percentage_error(y_test_B, y_pred_B)

print(f"Mean Absolute Percentage Error for A: {mape_A:.2f}%")
print(f"Mean Absolute Percentage Error for B: {mape_B:.2f}%")
