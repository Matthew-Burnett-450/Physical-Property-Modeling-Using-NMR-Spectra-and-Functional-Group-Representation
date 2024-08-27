import numpy as np
from DataCompolation.FeatureGenerator import get_data_test, get_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from skorch import NeuralNetRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load the data
x,Y,T,Names,features = get_data_test()
# Load the data
X_train, X_test, X_val, y_train_A, y_test_A, y_val_A, y_train_B, y_test_B, y_val_B, spectra_x,_,_,_ = get_data()

# Display the shapes of the data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train_A shape: {y_train_A.shape}")
print(f"y_test_A shape: {y_test_A.shape}")
print(f"y_val_A shape: {y_val_A.shape}")

# Standardize y
scalerA = StandardScaler()
y_train_A = scalerA.fit_transform(y_train_A.reshape(-1, 1))
y_test_A = scalerA.transform(y_test_A.reshape(-1, 1))
y_val_A = scalerA.transform(y_val_A.reshape(-1, 1))

scalerB = StandardScaler()
y_train_B = scalerB.fit_transform(y_train_B.reshape(-1, 1))
y_test_B = scalerB.transform(y_test_B.reshape(-1, 1))
y_val_B = scalerB.transform(y_val_B.reshape(-1, 1))


#load A and B model
class NMR1DCNN(nn.Module):
    def __init__(self, input_length=6554):
        super(NMR1DCNN, self).__init__()
        
        # Conv1D layer 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        # MaxPooling layer 1
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv1D layer 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        # MaxPooling layer 2
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv1D layer 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1)
        # MaxPooling layer 3
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv1D layer 4
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1)
        # MaxPooling layer 4
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the flattened size after the last convolution and pooling layers
        self.flattened_size = self.calculate_flattened_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def calculate_flattened_size(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
 # Instantiate the models
model_A = NMR1DCNN(input_length=features.shape[-1])
model_B = NMR1DCNN(input_length=features.shape[-1])

# Load the state dictionaries
model_A.load_state_dict(torch.load('nmr_cnn_modelA.pth', map_location=torch.device('cpu')))
model_B.load_state_dict(torch.load('nmr_cnn_modelB.pth', map_location=torch.device('cpu')))

# Set the models to evaluation mode
model_A.eval()
model_B.eval()

# Convert features to tensor
features = torch.from_numpy(features).float().unsqueeze(1)

# Predict A and B
with torch.no_grad():  # No need to track gradients for inference
    A = model_A(features).squeeze(1)
    B = model_B(features).squeeze(1)

#detach all
A = A.detach().numpy().reshape(-1, 1)
B = B.detach().numpy().reshape(-1, 1)

#unscale
A = scalerA.inverse_transform(A)
B = scalerB.inverse_transform(B)

# Calculate the predicted y values from the T
Y_pred = []
for i, t in enumerate(T):
    t = np.array(t)
    y = np.polyval([A[i], B[i]], t)
    Y_pred.append(y)


# Calculate R-squared
Y = np.array(Y)

r2 = r2_score(Y, Y_pred)
print(f"R-squared: {r2:.4f}")

#parity plot    

plt.figure(figsize=(8, 8))
plt.scatter(Y, Y_pred, c='blue', edgecolors='k')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Parity plot')
plt.show()
