import matplotlib.pyplot as plt

class WonPlot:
    """
    A class for creating and managing plots with consistent styling and color cycling.

    Parameters
    ----------
    Title : str, optional
        The title of the plot, which will be used as the filename when saving the plot. Default is 'plot'.
    """

    def __init__(self, Title='plot'):
        """
        Initialize the WonPlot class with a specified title and default settings.

        Parameters
        ----------
        Title : str, optional
            The title of the plot. Default is 'plot'.
        """
        # Set Arial as the font for the plot
        plt.rcParams['font.family'] = 'Arial'
        self.title = Title
        self.plot_color_index = 0  # Track the current color index for plot
        self.scatter_color_index = 0  # Track the current color index for scatter
        self.rainbow_colors = [
            'black', 'tab:blue', 'tab:orange', 'tab:green', 
            'tab:red', 'tab:purple', 'tab:brown', 
            'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        ]
    
    def next_plot_color(self):
        """
        Retrieve the next color for plot and update the index cyclically.

        Returns
        -------
        color : str
            The next color in the cycle for plotting.
        """
        color = self.rainbow_colors[self.plot_color_index % len(self.rainbow_colors)]
        self.plot_color_index += 1
        return color
    
    def next_scatter_color(self):
        """
        Retrieve the next color for scatter and update the index cyclically.

        Returns
        -------
        color : str
            The next color in the cycle for scatter plotting.
        """
        color = self.rainbow_colors[self.scatter_color_index % len(self.rainbow_colors)]
        self.scatter_color_index += 1
        return color
    
    def plot(self, x, y, label, linestyle='-', **kwargs):
        """
        Plot the given data with automatic color cycling.

        Parameters
        ----------
        x : array-like
            The x coordinates of the data points.
        y : array-like
            The y coordinates of the data points.
        label : str
            The label for the plot legend.
        linestyle : str, optional
            The line style of the plot. Default is '-'.
        **kwargs : optional
            Additional keyword arguments passed to `plt.plot`.
        """
        color = self.next_plot_color()  # Get the next color in the cycle for plot
        plt.plot(x, y, label=label, color=color, linestyle=linestyle, **kwargs)
    
    def scatter(self, x, y, label):
        """
        Scatter plot the given data with automatic color cycling.

        Parameters
        ----------
        x : array-like
            The x coordinates of the data points.
        y : array-like
            The y coordinates of the data points.
        label : str
            The label for the scatter plot legend.
        """
        color = self.next_scatter_color()  # Get the next color in the cycle for scatter
        plt.scatter(x, y, label=label, color=color)
    
    def xlabel(self, label):
        """
        Set the label for the x-axis.

        Parameters
        ----------
        label : str
            The label for the x-axis.
        """
        plt.xlabel(label)
    
    def ylabel(self, label):
        """
        Set the label for the y-axis.

        Parameters
        ----------
        label : str
            The label for the y-axis.
        """
        plt.ylabel(label)
    
    def plot_show(self, major_xstep=None, major_ystep=None, minor_xstep=None, minor_ystep=None):
        """
        Customize and display the plot with optional grid steps and save it as a PNG file.

        Parameters
        ----------
        major_xstep : float, optional
            Major step size for the x-axis grid. Default is None.
        major_ystep : float, optional
            Major step size for the y-axis grid. Default is None.
        minor_xstep : float, optional
            Minor step size for the x-axis grid. Default is None.
        minor_ystep : float, optional
            Minor step size for the y-axis grid. Default is None.
        """
        # Customize the plot's appearance
        ax = plt.gca()
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.tick_params(which='both', width=2, direction='in', length=4)
        
        if major_xstep is not None:
            ax.xaxis.set_major_locator(plt.MultipleLocator(major_xstep))
        if major_ystep is not None:
            ax.yaxis.set_major_locator(plt.MultipleLocator(major_ystep))
        if minor_xstep is not None:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_xstep))
        if minor_ystep is not None:
            ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_ystep))
        
        ax.xaxis.label.set_size(28)
        ax.yaxis.label.set_size(28)
        ax.xaxis.label.set_weight('bold')
        ax.yaxis.label.set_weight('bold')
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)

        # Set tick labels bold
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        
        plt.legend(prop={'size': 16}, loc='upper right', ncol=2)
        plt.grid(False)
        
        # Save and display the plot
        plt.savefig(self.title + '.png', dpi=300)
        plt.show()
    def format_plot(self, major_xstep=None, major_ystep=None, minor_xstep=None, minor_ystep=None):
        """
        Customize the plot with optional grid steps; will not save or show.

        Parameters
        ----------
        major_xstep : float, optional
            Major step size for the x-axis grid. Default is None.
        major_ystep : float, optional
            Major step size for the y-axis grid. Default is None.
        minor_xstep : float, optional
            Minor step size for the x-axis grid. Default is None.
        minor_ystep : float, optional
            Minor step size for the y-axis grid. Default is None.
        """
        # Customize the plot's appearance
        ax = plt.gca()
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.tick_params(which='both', width=2, direction='in', length=4)
        
        if major_xstep is not None:
            ax.xaxis.set_major_locator(plt.MultipleLocator(major_xstep))
        if major_ystep is not None:
            ax.yaxis.set_major_locator(plt.MultipleLocator(major_ystep))
        if minor_xstep is not None:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_xstep))
        if minor_ystep is not None:
            ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_ystep))
        
        ax.xaxis.label.set_size(28)
        ax.yaxis.label.set_size(28)
        ax.xaxis.label.set_weight('bold')
        ax.yaxis.label.set_weight('bold')
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)

        # Set tick labels bold
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        
        plt.legend(prop={'size': 16}, loc='upper right', ncol=2)
        plt.grid(False)
    

