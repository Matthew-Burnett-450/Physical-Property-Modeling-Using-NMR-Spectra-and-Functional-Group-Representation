import matplotlib.pyplot as plt
from WonPlot import WonPlot  # Ensure WonPlot.py is in the same directory as this script
# If WonPlot.py is in a subfolder, import it like this: from FolderName.SubFolderName.WonPlot import WonPlot

# Test plotting a simple plot
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
plt.figure(figsize=(10, 5))

# Instantiate WonPlot for every plot
wp = WonPlot('Test')

# Plotting with WonPlot
wp.plot(x, y, 'y=x^2')                      # (X, Y, Label, **Kwargs) - Kwargs can be passed to the plot function
wp.scatter(x, y, 'Points')                  # (X, Y, Label) - Kwargs can be passed to the scatter function
wp.xlabel('x')                              # Set x-axis label
wp.ylabel('y')                              # Set y-axis label
wp.plot_show(major_xstep=1, major_ystep=5)  # (xstep, ystep, minor_xstep, minor_ystep) - All optional for tick spacing
plt.close()                                 # WonPlot works with Matplotlib inline, so you can call matplotlib functions with WonPlot functions

# Test plotting a lot of data
# WonPlot will automatically cycle through colors; custom colors can be added to the rainbow_colors list 
# by using wp.rainbow_colors.append('color'). You can also change the plot color by passing color='color' to the plot function.
import numpy as np

x = []
y = []
for i in range(1, 10):
    # Generate random x curves
    x.append(np.linspace(0, 10, 10))
    y.append(np.linspace(0, 10, 10) + i)

plt.figure(figsize=(12, 8))
wp = WonPlot('Test')
for i in range(len(x)):
    wp.plot(x[i], y[i], f'y=x + {i+1}')
for i in range(len(x)):
    wp.scatter(x[i], y[i], f'Points {i+1}')
wp.xlabel('Measured')
wp.ylabel('Predicted')
wp.plot_show(major_xstep=4, major_ystep=4, minor_xstep=1, minor_ystep=1)

# You can also use normal matplotlib and call wp.plot_show() at the end to format the plot
wp = WonPlot('Test')
plt.plot(x[0], y[0], label='y=x+1')
plt.scatter(x[0], y[0], label='Points')
plt.xlabel('Measured')
plt.ylabel('Predicted')
wp.plot_show(major_xstep=4, major_ystep=4, minor_xstep=1, minor_ystep=1)

# If you want to format but not show the plot, you can call wp.format_plot() at the end
wp = WonPlot('Test')
plt.plot(x[0], y[0], label='y=x+1')
plt.scatter(x[0], y[0], label='Points')
plt.xlabel('Measured')
plt.ylabel('Predicted')
wp.format_plot(major_xstep=4, major_ystep=4, minor_xstep=1, minor_ystep=1)
# Continue with whatever else you're doing here
plt.show()

# If you want to set a size for the plot use plt.figure(figsize=(width, height)) before calling WonPlot
plt.figure(figsize=(12, 8))
wp = WonPlot('Test')
plt.plot(x[0], y[0], label='y=x+1')
plt.scatter(x[0], y[0], label='Points')
plt.xlabel('Measured')
plt.ylabel('Predicted')
wp.plot_show(major_xstep=4, major_ystep=4, minor_xstep=1, minor_ystep=1)
plt.show()
