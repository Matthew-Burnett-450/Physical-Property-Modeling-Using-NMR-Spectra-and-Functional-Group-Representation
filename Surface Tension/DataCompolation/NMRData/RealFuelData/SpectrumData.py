import numpy as np
import matplotlib.pyplot as plt
import json


#save to Spectrum.Json the file path for 13C and 1H spectra and an empty slot for their peak width and len of the data
data = {'1H':{'FilePath':'data\TrianaryMix1H.csv','PeakWidth':0,'DataLength':0},'13C':{'FilePath':'data\TrianaryMix13C.csv','PeakWidth':0,'DataLength':0}}
with open('data\SpectrumData.json', 'w') as f:
    json.dump(data, f)


def lorenzian(x,x0,linewidth):
    return 1/(1+((x-x0)/linewidth)**2)

def find_peak_width(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    #extract data
    ppm = data[:,0]
    intensity = data[:,1]

    #normalize the data
    intensity = intensity/np.max(intensity)

    #find the tallest lorentzian peak
    max_intensity = max(intensity)
    max_index = np.where(intensity == max_intensity)
    max_index = max_index[0][0]

    #fit a lorentzian to the tallest peak with gradient descent
    x0 = ppm[max_index]
    linewidth = 0.1
    learning_rate = 0.01
    for i in range(1000):
        gradient = 2*(lorenzian(ppm, x0, linewidth) - intensity)*lorenzian(ppm, x0, linewidth)**2
        x0 = x0 - learning_rate*np.sum(gradient*(ppm-x0))
        linewidth = linewidth - learning_rate*np.sum(gradient*(ppm-x0)**2)

    print('linewidth:', linewidth)
    print(len(ppm))
    return linewidth, len(ppm),ppm.tolist()

#save the peak width and data length to the json file
C_lw, C_dl,C_ppm = find_peak_width('data\TrianaryMix13C.csv')
H_lw, H_dl,H_ppm = find_peak_width('data\TrianaryMix1H.csv')

with open('data\SpectrumData.json', 'r') as f:
    data = json.load(f)

data['1H']['PeakWidth'] = H_lw
data['1H']['DataLength'] = H_dl
data['1H']['ppm'] = H_ppm
data['13C']['PeakWidth'] = C_lw
data['13C']['DataLength'] = C_dl
data['13C']['ppm'] = C_ppm

with open('data\SpectrumData.json', 'w') as f:
    json.dump(data, f)
