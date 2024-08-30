import matplotlib.pyplot as plt

class WonPlot:

    
    def __init__(self, Title='plot'):
        self.title = Title
        self.plot_color_index = 0  # Track the current color index for plot
        self.scatter_color_index = 0  # Track the current color index for scatter
        self.rainbow_colors = ['black',
        'tab:blue', 'tab:orange', 'tab:green', 
        'tab:red', 'tab:purple', 'tab:brown', 
        'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    
    def next_plot_color(self):
        # Retrieve the next color for plot and update the index cyclically
        color = self.rainbow_colors[self.plot_color_index % len(self.rainbow_colors)]
        self.plot_color_index += 1
        return color
    
    def next_scatter_color(self):
        # Retrieve the next color for scatter and update the index cyclically
        color = self.rainbow_colors[self.scatter_color_index % len(self.rainbow_colors)]
        self.scatter_color_index += 1
        return color
    
    def plot(self, x, y, label, linestyle='-',**kwargs):
        color = self.next_plot_color()  # Get the next color in the cycle for plot
        plt.plot(x, y, label=label, color=color, linestyle=linestyle, **kwargs)
    
    def scatter(self, x, y, label):
        color = self.next_scatter_color()  # Get the next color in the cycle for scatter
        plt.scatter(x, y, label=label, color=color)
    
    def xlabel(self, label):
        plt.xlabel(label)
    
    def ylabel(self, label):
        plt.ylabel(label)
    
    def plot_show(self, xstep=None, ystep=None,minor_xstep=None,minor_ystep=None):
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['top'].set_color('black')
        plt.gca().tick_params(which='both', width=2, direction='in', length=4)
        if xstep is not None:
            plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xstep))
        if ystep is not None:
            plt.gca().yaxis.set_major_locator(plt.MultipleLocator(ystep))
        if minor_xstep is not None:
            plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(minor_xstep))
        if minor_ystep is not None:
            plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(minor_ystep))
        plt.gca().xaxis.label.set_size(28)
        plt.gca().yaxis.label.set_size(28)
        plt.gca().xaxis.label.set_weight('bold')
        plt.gca().yaxis.label.set_weight('bold')
        plt.gca().xaxis.set_tick_params(labelsize=16)
        plt.gca().yaxis.set_tick_params(labelsize=16)
        plt.legend(prop={'size': 16},loc='upper right',ncol=2)
        plt.grid(False)
        plt.savefig(self.title + '.png', dpi=300)
        plt.show()
       
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