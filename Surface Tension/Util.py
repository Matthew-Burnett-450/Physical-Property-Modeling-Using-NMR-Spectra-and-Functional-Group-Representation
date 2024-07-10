import json 
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.interpolate import interp1d
from scipy.optimize import minimize

def load_data(Filename='SurfTTrainingData.json'):
    FG=[]
    X=[]
    Y=[]
    T=[]
    Tc=[]
    idx=[]
    Names=[]
    Tc_True = []
    id=0
    with open(Filename) as f:
        data = json.load(f)
    for i in data:
        Y.append(i['Y'])
        T.append(i['T'])
        try:
            Tc.append(i['Tc_Pred'])
        except:
            Tc.append(i['Tc_True'])
        Tc_True.append(i['Tc_True'])
        try:
            X.append(i['13C_shift'])
        except:
            X.append([])
        FG_dict=i['functional_groups']
        temp_vec = []
        for j in range(len(i['T'])):
            temp_vec.append(id)
        idx.append(temp_vec)
        id+=1
        Names.append(i['MolName'])
        #unpack functional groups  {"CH3": 2, "CH": 0, "Quaternary_C": 0, "Benzyl_Rings": 0, "C=C": 2, "CH2": 0, "CH2_Chains": 0, "Cyclo_CH2": 0}
        FG.append([FG_dict['CH3'],FG_dict['CH'],FG_dict['Quaternary_C'],FG_dict['Benzyl_Rings'],FG_dict['CH2'],FG_dict['CH2_Chains'],FG_dict['Cyclo_CH2']])
    return X,FG,T,Y,['CH3','CH','Quaternary_C','Benzyl_Rings','CH2','CH2_Chains','Cyclo_CH2'],idx,Names,Tc,Tc_True

class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def custom_loss(self, params, X, y):
        intercept = params[0]
        coef = params[1:]
        y_pred = np.dot(X, coef) + intercept
        return np.mean((1 - y_pred / y) ** 2)
    
    def MSE(self, params, X, y):
        intercept = params[0]
        coef = params[1:]
        y_pred = np.dot(X, coef) + intercept
        return np.mean((y_pred - y) ** 2)
    
    def MAE(self, params, X, y):
        intercept = params[0]
        coef = params[1:]
        y_pred = np.dot(X, coef) + intercept
        return np.mean(np.abs(y_pred - y))

    def fit(self, X, y):
        n_features = X.shape[1]
        initial_params = np.zeros(n_features + 1)
        result = minimize(self.MSE, initial_params, args=(X, y))
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

def fit_fourier_series(data, n_terms, plot=False):
    """
    Fits a Fourier series to the given 1D list of data.

    Parameters:
    data (list or np.array): The input data to fit.
    n_terms (int): The number of Fourier terms to use in the fit.
    plot (bool): If True, plots the original data and the fitted Fourier series.

    Returns:
    list: Fourier series coefficients.
    """
    n_terms = n_terms//2 + 1

    # Convert data to numpy array
    data = np.array(data)
    n = len(data)
    
    # Compute the FFT of the data
    fft_coeffs = fft(data)
    
    # Keep only the first n_terms coefficients
    fft_coeffs[n_terms:] = 0
    
    # Compute the inverse FFT to get the fitted data
    fitted_data = ifft(fft_coeffs).real
    
    # Prepare the coefficients for return
    a0 = fft_coeffs[0].real / n
    an = 2 * fft_coeffs[1:n_terms].real / n
    bn = -2 * fft_coeffs[1:n_terms].imag / n
    
    # Collect the coefficients into a list
    coefficients = [a0] + an.tolist() + bn.tolist()
    
    # Optional plot
    if plot:
        x = np.arange(n)
        plt.figure(figsize=(10, 5))
        plt.plot(x, data, label='Original Data')
        plt.plot(x, fitted_data, label='Fitted Fourier Series', linestyle='--')
        plt.legend()
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Fourier Series Fit')
        plt.show()
    
    return coefficients

#fit a curve of the form exp(A + B/T )
def fit_exp_curve(T,Y):
    """
    Fits a curve of the form exp(A + B/T ) to the given 1D list of data.

    Parameters:
    T (list or np.array): The input data to fit.
    Y (list or np.array): The input data to fit.

    Returns:
    list: coefficients.
    """
    # Convert data to numpy array
    T = np.array(T)
    Y = np.array(Y)
    
    #fit curve
    p = np.polyfit(1/T, np.log(Y), 1)
    
    return p

def cutoff_point_detector(x, y,idx, threshold=.1, plot=False):
    """
    Detects the cutoff point where the curve stops being linear and starts dropping off.
    
    Parameters:
    x (array): The x values of the data.
    y (array): The y values of the data.
    threshold (float): The threshold for detecting significant changes in the slope.
    plot (bool): Whether to plot the data and the detected cutoff point.
    
    Returns:
    x_truncated (array): The truncated x values.
    y_truncated (array): The truncated y values.
    cutoff_x (float): The x value where the cutoff point was detected.
    """
    # Calculate the first derivative
    dy_dx = np.gradient(y, x)
    dy2_dx2 = np.gradient(dy_dx, x)
    
    # Identify the cutoff point where the derivative changes significantly
    cutoff_index = np.argmax(np.abs(dy_dx) < threshold * np.max(np.abs(dy_dx)))
    cutoff_x = x[cutoff_index]
    
    if plot:
        # Plot the original data and the detected cutoff point
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Original Data')
        plt.axvline(cutoff_x, color='r', linestyle='--', label=f'Cutoff Point at x = {cutoff_x:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Detected Cutoff Point')
        plt.legend()
        plt.show()
        
        # Truncate the data at the cutoff point
        x_truncated = x[:cutoff_index]
        y_truncated = y[:cutoff_index]
        
        # Plot the truncated data
        plt.figure(figsize=(10, 6))
        plt.plot(x_truncated, y_truncated, label='Truncated Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Truncated Data after Cutoff Point')
        plt.legend()
        plt.show()
    
    # Truncate the data at the cutoff point
    x_truncated = x[:cutoff_index]
    y_truncated = y[:cutoff_index]
    idx_truncated = idx[:cutoff_index]
    
    return x_truncated, y_truncated,idx_truncated