import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MinMaxScaler():
    def __init__(self):
        self.min=None
        self.max=None
    def fit(self,X):
        self.min=np.min(X)
        self.max=np.max(X)
    def transform(self,X):
        return (X-self.min)/(self.max-self.min)
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self,X):
        return X*(self.max-self.min)+self.min
    
#robust scaler
class MinMaxScaler():
    def __init__(self):
        self.median = None
        self.q1 = None
        self.q3 = None

    def fit(self, X):
        self.median = np.median(X)
        self.q1 = np.percentile(X, 25)  # Using 20th percentile for Q1
        self.q3 = np.percentile(X, 75)  # Using 80th percentile for Q3

    def transform(self, X):
        # Custom scaling logic to adjust the range after robust scaling
        return (((X - self.median) / (self.q3 - self.q1)) + 2) / 4

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # Corrected inverse transformation to match the custom forward transformation
        return (X * 4 - 2) * (self.q3 - self.q1) + self.median


def sample_latent_space(mu, log_var, num_samples=10):
    """
    Generate samples from the latent space distribution defined by mu and log_var.
    Args:
        mu (Tensor): Mean of the latent space distribution.
        log_var (Tensor): Log variance of the latent space distribution.
        num_samples (int): Number of samples to generate.
    Returns:
        List[Tensor]: A list of sampled latent vectors.
    """
    std = torch.exp(0.5 * log_var)
    samples = [mu + std * torch.randn_like(std) for _ in range(num_samples)]
    return samples

def grab_histogram_13C(data):
    X=[]
    for i in range(len(data)):

        shift=np.array(data[i]['13C_shift'])[:,0]
        #put list of shifts onto a histogram
        hist, bin_edges = np.histogram(shift, bins=2500, range=(0,250))
        #nan check
        if np.isnan(hist).any():
            print('nan in Hisotgram')
        
        #min max scaling
        hist_normalized = hist/np.sum(hist)
        #turn to just 0 or 1
        

        #nan check
        if np.isnan(hist_normalized).any():
            print('nan')
            #nan to num
            hist_normalized=np.nan_to_num(hist_normalized)
        hist_normalized = hist_normalized.tolist()
        X.append(hist_normalized)
    return X

def grab_histogram_1H(data):
    X=[]
    for i in range(len(data)):

        shift=np.array(data[i]['1H_shift'])[:,0]
        #put list of shifts onto a histogram
        hist, bin_edges = np.histogram(shift, bins=150, range=(0,15))
        #nan check
        if np.isnan(hist).any():
            print('nan in Hisotgram')
        
        #min max scaling
        hist_normalized = hist/np.sum(hist)
        #turn to just 0 or 1
        

        #nan check
        if np.isnan(hist_normalized).any():
            print('nan')
            #nan to num
            hist_normalized=np.nan_to_num(hist_normalized)
        hist_normalized = hist_normalized.tolist()
        X.append(hist_normalized)
    return X

def train_test_validation_split_torch(X, Y, test_size=0.1, validation_size=0.125):
    """
    Splits the dataset into training, validation, and testing sets.
    
    Args:
        X (Tensor): The input features.
        Y (Tensor): The targets.
        test_size (float): The proportion of the dataset to include in the test split.
        validation_size (float): The proportion of the training set to include in the validation split.
        random_state (int): Seed for the random number generator.
    
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    num_samples = X.size(0)
    
    # Split into train+val and test
    num_test_samples = int(num_samples * test_size)
    indices = torch.randperm(num_samples)
    train_val_indices = indices[num_test_samples:]
    test_indices = indices[:num_test_samples]
    
    X_train_val = X[train_val_indices]
    Y_train_val = Y[train_val_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    
    # Split train+val into train and val
    num_train_samples = X_train_val.size(0)
    num_val_samples = int(num_train_samples * validation_size)
    indices_train_val = torch.randperm(num_train_samples)
    val_indices = indices_train_val[:num_val_samples]
    train_indices = indices_train_val[num_val_samples:]
    
    X_train = X_train_val[train_indices]
    Y_train = Y_train_val[train_indices]
    X_val = X_train_val[val_indices]
    Y_val = Y_train_val[val_indices]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def get_batches(X, y, batch_size):
    """Yield successive n-sized chunks from X and y."""
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])



def adjusted_averaged_loss(actual, predicted, n):
    """
    Adjusted function to compute loss by averaging every n steps,
    correcting for arbitrary tensor sizes.
    
    Args:
    - actual (torch.Tensor): Tensor of actual values.
    - predicted (torch.Tensor): Tensor of predicted values.
    - n (int): Number of steps to average over.
    
    Returns:
    - torch.Tensor: The computed loss.
    """
    # Calculate the number of complete bins for both tensors
    num_complete_bins = min(actual.numel(), predicted.numel()) // n
    
    # Ensure we only include elements that form complete bins
    actual = actual[:num_complete_bins * n]
    predicted = predicted[:num_complete_bins * n]
    
    # Reshape tensors to have 'n' columns for averaging
    actual_reshaped = actual.view(-1, n)
    predicted_reshaped = predicted.view(-1, n)
    
    # Calculate averages across bins (along columns)
    actual_averages = actual_reshaped.sum(dim=1)
    predicted_averages = predicted_reshaped.sum(dim=1)
    
    # Compute the MSE loss between the averaged bins
    loss = F.mse_loss(actual_averages, predicted_averages)
    
    return loss

def augment_fft_data(fft_data, roll_range=100, max_additional_peaks=50, amplitude_range=.05,noiserange=100):
    """
    Augment FFT data by rolling and then adding a random amount of smaller peaks.
    
    Parameters:
    - fft_data: The FFT data to augment.
    - roll_range: The range for random rolling.
    - max_additional_peaks: The maximum number of additional smaller peaks to add around each significant peak.
    - amplitude_range: The range of amplitudes for the additional smaller peaks.
    
    Returns:
    - The augmented FFT data.
    """
    # Roll the data

    rolled_fft = np.roll(fft_data, np.random.randint(-roll_range, roll_range))

    # Find significant peaks (for simplicity, using a basic approach)
    peaks_indices = [i for i in range(1, len(rolled_fft) - 1) if rolled_fft[i] > 0]
    
    # Add random smaller peaks around each significant peak
    for peak_idx in peaks_indices:
        num_additional_peaks = np.random.randint(max_additional_peaks-5, max_additional_peaks + 1)
        for _ in range(num_additional_peaks):
            if peak_idx > 0 and peak_idx < len(rolled_fft) - 1:
                noise_roll=int(np.random.normal(0,noiserange))
                while noise_roll + peak_idx > len(rolled_fft)-1 or noise_roll + peak_idx < 0:
                    noise_roll=int(np.random.normal(0,noiserange))            
                amp_shift=np.random.normal(0,amplitude_range) * rolled_fft[peak_idx]
                if amp_shift + rolled_fft[peak_idx + noise_roll] < 0:
                    amp_shift=amp_shift*-1
                rolled_fft[peak_idx + noise_roll] += amp_shift

    
    return rolled_fft

def DataAug(X, Y, device, n_copies=3, roll_range=50):
    # Convert tensors to numpy arrays
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    # Determine the size of the augmented data
    total_samples = len(X) * (n_copies + 1)  # Original + n copies for each sample
    
    # Pre-allocate space for augmented data
    X_return = np.empty((total_samples, *X.shape[1:]), dtype=X.dtype)
    Y_return = np.repeat(Y, n_copies + 1, axis=0)  # Replicate each label n_copies times
    
    # Populate the first part of the return arrays with the original data
    X_return[:len(X)] = X
    
    # Augment data by adding small peaks and applying rolling
    for i in range(n_copies):
        for j in range(len(X)):
            X_aug = augment_fft_data(X[j], roll_range=roll_range)
            X_return[(i+1)*len(X)+j] = X_aug
    

    # Convert back to tensors and move to the specified device
    X_return = torch.tensor(X_return, dtype=torch.float).to(device)
    Y_return = torch.tensor(Y_return, dtype=torch.float).to(device)

    return X_return, Y_return
            

def find_prominent_peaks_with_stride(fft_amplitudes, window_size, stride):
    """
    Find the most prominent peaks using a moving window and stride approach on FFT data,
    returning an array of the same shape with zeros except at the peaks.
    
    Parameters:
    - fft_amplitudes: NumPy array of amplitudes from FFT analysis.
    - window_size: Size of the moving window.
    - stride: Number of indices to move the window at each step.
    
    Returns:
    - NumPy array of the same shape as fft_amplitudes, with zeros everywhere except at the peaks.
    """
    num_points = len(fft_amplitudes)
    peaks_output = np.zeros(num_points)  # Initialize output array with zeros
    
    # Slide the window across the FFT data with the specified stride
    for start_index in range(0, num_points - window_size + 1, stride):
        end_index = start_index + window_size
        subsection = fft_amplitudes[start_index:end_index]
        
        # Use np.argmax() to find the index of the highest peak in the current window relative to the subsection
        peak_index_relative = np.argmax(subsection)
        peak_index = start_index + peak_index_relative  # Convert to absolute index in the full array
        
        # Store the amplitude of the peak at the corresponding position in the output array
        peaks_output[peak_index] = fft_amplitudes[peak_index]
    
    return peaks_output




def adjusted_r_squared(X, y, y_pred):
    """
    Calculate the adjusted R-squared value.
    
    Parameters:
    - X: numpy array or a list of lists with the input features. Should not include the intercept.
    - y: numpy array or list of the actual values.
    - y_pred: numpy array or list of the predicted values by the model.
    
    Returns:
    - Adjusted R-squared value as a float.
    """
    # Ensure inputs are numpy arrays for efficiency
    X = np.array(X)
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    # Number of observations and predictors
    n = len(y)
    p = X.shape[1]
    
    # Calculate RSS (Residual Sum of Squares) and TSS (Total Sum of Squares)
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    
    # Calculate R-squared
    r_squared = 1 - (rss / tss)
    
    # Calculate Adjusted R-squared
    adjusted_r2 = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    
    return adjusted_r2

def high_pass_filter_with_pytorch(tensor, percentile=95):
    # Normalize the histogram
    tensor_normalized = tensor / tensor.sum()
    
    # We only consider non-zero elements for percentile calculation
    non_zero_elements = tensor_normalized[tensor_normalized != 0]
    
    # Sort the non-zero elements
    sorted_tensor, _ = torch.sort(non_zero_elements)
    
    # Calculate the index for the given percentile
    index = int(len(sorted_tensor) * (percentile / 100.0))
    
    # Find the threshold value
    threshold = sorted_tensor[index] if index < len(sorted_tensor) else sorted_tensor[-1]
    print(f"Threshold: {threshold}")
    
    # Apply high pass filter
    tensor_normalized[tensor_normalized < threshold] = 0
    
    # Re-normalize the tensor
    tensor_normalized = tensor_normalized / tensor_normalized.sum()
    
    return tensor_normalized

def make_mixtures_and_append(X, Y, num_samples, max_components):
    original_X = X.clone()  # Clone to avoid modifying the original tensor
    original_Y = Y.clone()  # Clone to avoid modifying the original tensor

    mixed_X = torch.empty((num_samples, X.shape[1]), dtype=X.dtype)  # Preallocate mixed X data tensor
    mixed_Y = torch.empty((num_samples, 2), dtype=Y.dtype)  # Preallocate Y tensor for mixed properties
    
    for i in range(num_samples):
        # Randomly decide the number of components for this mixture
        num_components = torch.randint(1, max_components + 1, (1,)).item()

        # Randomly select component indices without replacement
        component_indices = torch.randperm(X.shape[0])[:num_components]

        # Generate random mixture ratios that sum to 1
        ratios = torch.distributions.Dirichlet(torch.ones(num_components)).sample()

        # Calculate the mixed data for X, Amix, and Bmix
        X_mix = torch.zeros_like(X[0])
        A_mix = 0.0
        B_mix = 0.0
        for ratio, idx in zip(ratios, component_indices):
            X_mix += ratio * X[idx]
            A_mix += ratio * Y[idx, 0]
            B_mix += ratio * Y[idx, 1]
        #normalize X_mix
        X_mix = high_pass_filter_with_pytorch(X_mix)

        # Store the mixture info and calculated values
        mixed_X[i] = X_mix
        mixed_Y[i, 0] = A_mix
        mixed_Y[i, 1] = B_mix

    # Append mixed data to the original data
    appended_X = torch.cat((original_X, mixed_X), dim=0)
    appended_Y = torch.cat((original_Y, mixed_Y), dim=0)


    return appended_X, appended_Y