from WonPlot.WonPlot import WonPlot as wp
import numpy as np
import matplotlib.pyplot as plt

#load JP5_experimental_FG and JP5_experimental and JP5_predicted and JP5_predicted_FG from csv
data = np.loadtxt('JP5_experimental_FG.csv', delimiter=',', skiprows=1)
T_exp_FG,Y_exp_FG = data[:,0],data[:,1]
data = np.loadtxt('JP5_experimental.csv', delimiter=',', skiprows=1)
T_exp,Y_exp = data[:,0],data[:,1]
data = np.loadtxt('JP5_predicted_FG.csv', delimiter=',', skiprows=1)
T_pred_FG,Y_pred_FG = data[:,0],data[:,1]
data = np.loadtxt('JP5_predicted.csv', delimiter=',', skiprows=1)
T_pred,Y_pred = data[:,0],data[:,1]

#example
"""
#test ploting a simple plot
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
plt.figure(figsize=(10, 5))
wp = WonPlot('Test')
wp.plot(x, y, 'y=x^2')
wp.scatter(x, y, 'Points')
wp.xlabel('x')
wp.ylabel('y')
wp.plot_show(xstep=1, ystep=5)
plt.close()

#test ploting alot of data
import numpy as np
x = []
y = []
for i in range(1, 10):
    #random x curves
    x.append(np.linspace(0, 10, 10))
    y.append(np.linspace(0, 10, 10)+i)

plt.figure(figsize=(12, 8))
wp = WonPlot('Test')
for i in range(len(x)):
    wp.plot(x[i], y[i], f'y=x + {i+1}')
for i in range(len(x)):
    wp.scatter(x[i], y[i], f'Points {i+1}')
wp.xlabel('Measured')
wp.ylabel('Predicted')
wp.plot_show(xstep=4, ystep=4,minor_xstep=1,minor_ystep=1)
"""

#plot the data
plt.figure(figsize=(12, 8))
wp = wp('JP5 Viscosity Prediction')
wp.plot(T_pred, Y_pred, 'Predicted NMR')
wp.scatter(T_exp, Y_exp, 'Measured')
wp.plot(T_pred_FG, Y_pred_FG, 'Predicted FG')
wp.xlabel('Temperature (K)')
wp.ylabel('Viscosity (cP)')
#add error bars +/- e 0.02 cP to the experimental data
wp.plot_show(xstep=50, ystep=0.5,minor_xstep=10,minor_ystep=0.1)
plt.close()