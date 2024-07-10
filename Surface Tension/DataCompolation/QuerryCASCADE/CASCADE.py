from splinter import Browser
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import ElementClickInterceptedException
import logging
from bs4 import BeautifulSoup
import numpy as np
from nmrsim.plt import mplplot
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import random
from nmrsim.firstorder import first_order_spin_system

class CASCADE_C():
    def __init__(self):
        # Set up logging to suppress warnings
        logging.basicConfig(level=logging.INFO)
        logging.getLogger('selenium').setLevel(logging.CRITICAL)

        # Set up Edge options for headless mode
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920x1080')
    def get_spectrum(self,smiles,show=False):
        #numpy like docstring
        """
        Parameters
        ----------
        smiles : str
            The SMILES string for the molecule of interest.
            Must be a string.
            Must not be empty.
            Must only contain C, H, N, O, S, P, F, Cl Atoms
        show : bool
            Whether to display the plot of the predicted spectrum.
        Returns
        -------
        freqs : array of float

        intensities : array of float

        """
        #check if smiles is a string
        if not isinstance(smiles, str):
            raise TypeError('smiles must be a string')
        #check if show is a bool
        if not isinstance(show, bool):
            raise TypeError('show must be a bool')
        #check if smiles is empty
        if not smiles:
            raise ValueError('smiles cannot be an empty string')

    
        with Browser('chrome', options=self.chrome_options) as browser:
            browser.visit('https://nova.chem.colostate.edu/cascade/predict/')

            browser.fill('smiles', smiles)

            button = browser.find_by_tag('button')
            for btn in button:
                if btn['type'] == 'submit':
                    btn.click()
                    break
            #if this button is on screen, click it
            #<button type="button" class="btn btn-primary">Notify me</button>
            button = browser.find_by_tag('button')
            for btn in button:
                if btn['type'] == 'Notify me':
                    btn.click()

            #wait untill you see "complete"
            while browser.is_element_not_present_by_tag('h3'):
                pass


            buttons = browser.find_by_css('.btn.btn-success')
            for button in buttons:
                if button.text == 'View Conformers':
                    browser.execute_script("arguments[0].click();", button._element)
                    break

            #copy the table from page
            table = browser.find_by_tag('table')
            table_html = table.outer_html

            soup = BeautifulSoup(table_html, 'html.parser')
            # Find the table rows
            rows = soup.find_all('tr')[1:]  # Skip the header row

            # Extract the data
            data = []
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text for ele in cols]
                data.append(cols)

            # Convert the data to a numpy array
            shifts = np.array(data,dtype=np.float64)[:,1]

            
            # Calculate the relative density (proportion of each unique shift)
            unique_shifts, counts = np.unique(shifts, return_counts=True)
            relative_densities = counts / counts.sum()

            unique_shifts = unique_shifts.astype(np.float16)

            print((unique_shifts, relative_densities)) 

            data = list(zip(unique_shifts, relative_densities))
            #first order prediction
            freqs,intensities=mplplot(data,w=.01,hidden=show,limits=(250,0))
            plt.close()

            #save last spectrum as self
            self.freqs = freqs
            self.intensities = intensities
            self.unique_shifts = unique_shifts

            return freqs,intensities

    def __bootstrap_shifted(self, mean, std_dev, n_samples):
        bootstrapped_means = []
        for _ in range(n_samples):
            resample = self.unique_shifts
            shifted_resample = resample - np.random.normal(mean, std_dev, len(resample)) * np.random.choice([-1, 1], len(resample))
            bootstrapped_means.append(shifted_resample)
        return np.array(bootstrapped_means)

    def calculate_bootstrap_ci(self, mean=1.26, std_dev=.5,confidence_level=.95,n=1000):
        """
        Parameters
        ----------
        mean : float
            The mean of the normal distribution used to shift the data.
            std_dev : float
            The standard deviation of the normal distribution used to shift the data.
            confidence_level : float
            The confidence level for the confidence interval.
            n : int
            The number of bootstrap samples to generate.
            Returns
            -------
            lower_bound : float
            The lower bound of the confidence interval.
            upper_bound : float
            The upper bound of the confidence interval.
            unique_shifts : array of float
            The unique chemical shifts used to generate the confidence interval.
        """
        bootstrapped_means = self.__bootstrap_shifted(mean, std_dev, n_samples=n)
        lower_percentile = (1.0 - confidence_level) / 2.0 * 100
        upper_percentile = (1.0 + confidence_level) / 2.0 * 100
        lower_bound = np.percentile(bootstrapped_means, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrapped_means, upper_percentile, axis=0)
        return (lower_bound, upper_bound, self.unique_shifts)
    def plot_last_spectrum(self,show=True,plot_ci=True):
        """
        Parameters
        ----------
        show : bool
            Whether to display the plot of the predicted spectrum.
        plot_ci : bool
            Whether to plot the confidence intervals.
        Returns
        -------
        None
        """
        plt.plot(self.freqs,self.intensities)
        if plot_ci == True:
            conf_intervals = self.calculate_bootstrap_ci()
            lower_bound, upper_bound, unique_shifts = conf_intervals
            plt.vlines(lower_bound,0,.01,color='orange',linewidth=2)
            plt.vlines(upper_bound,0,.01,color='orange',linewidth=2)
            plt.vlines(unique_shifts,0,.01,color='k',linestyle='--')
        if show == True:
            plt.gca().invert_xaxis()
            plt.show()

"""
#CASCADE_C test
CPred = CASCADE_C()
CPred.get_spectrum('CCCCCC',show=True)
CPred.calculate_bootstrap_ci()

plt.figure(figsize=(12,8))
CPred.plot_last_spectrum(show=False)
plt.title('Spectrum of n-hexane')
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.savefig('n-hexane_spectrum.png')
plt.show()
"""

class CASCADE_H():
    def __init__(self):
        # Set up logging to suppress warnings
        logging.basicConfig(level=logging.INFO)
        logging.getLogger('selenium').setLevel(logging.CRITICAL)

        # Set up Edge options for headless mode
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920x1080')
    def get_spectrum(self,smiles,show=False):
        #numpy like docstring
        """
        Parameters
        ----------
        smiles : str
            The SMILES string for the molecule of interest.
            Must be a string.
            Must not be empty.
            Must only contain C, H, N, O, S, P, F, Cl Atoms
        show : bool
            Whether to display the plot of the predicted spectrum.
        Returns
        -------
        freqs : array of float

        intensities : array of float

        """
        #check if smiles is a string
        if not isinstance(smiles, str):
            raise TypeError('smiles must be a string')
        #check if show is a bool
        if not isinstance(show, bool):
            raise TypeError('show must be a bool')
        #check if smiles is empty
        if not smiles:
            raise ValueError('smiles cannot be an empty string')

    
        with Browser('chrome', options=self.chrome_options) as browser:
            browser.visit('https://nova.chem.colostate.edu/cascade/predict/')

            browser.fill('smiles', smiles)

            #check <label for="H_check"><span><sup>1</sup>H</span></label>
            checkbox_label = browser.find_by_xpath("//label[@for='H_check']")
            if checkbox_label:
                # Find the checkbox input element and check it
                checkbox_input = browser.find_by_id('H_check')
                checkbox_input.check()
                print("Checkbox checked successfully")

            button = browser.find_by_tag('button')
            for btn in button:
                if btn['type'] == 'submit':
                    btn.click()
                    break
            #if this button is on screen, click it
            #<button type="button" class="btn btn-primary">Notify me</button>
            button = browser.find_by_tag('button')
            for btn in button:
                if btn['type'] == 'Notify me':
                    btn.click()

            #wait untill you see "complete"
            while browser.is_element_not_present_by_tag('h3'):
                pass

            buttons = browser.find_by_css('.btn.btn-success')
            for button in buttons:
                if button.text == 'View Conformers':
                    browser.execute_script("arguments[0].click();", button._element)
                    break
 
            #copy the table from page
            table = browser.find_by_tag('table')
            table_html = table.outer_html

            soup = BeautifulSoup(table_html, 'html.parser')
            # Find the table rows
            rows = soup.find_all('tr')[1:]  # Skip the header row

            # Extract the data
            data = []
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text for ele in cols]
                data.append(cols)

            # Convert the data to a numpy array
            shifts = np.array(data,dtype=np.float64)[:,1]

            #count unique shifts rounded to .1
            #unique_shifts, counts = np.unique(np.round(shifts,3), return_counts=True)
            #relative_densities = counts / counts.sum() 

            data = list(zip(shifts, np.ones_like(shifts)))
            
            freqs,intensities=mplplot(data,w=.00025,hidden=show,limits=(10,0),points=20000)
            #norm intensities
            intensities = intensities / intensities.sum()
            plt.close()

            #save last spectrum as self
            self.freqs = freqs
            self.intensities = intensities
            self.shifts = shifts

            return freqs,intensities

    def __bootstrap_shifted(self, mean, std_dev, n_samples):
        bootstrapped_means = []
        for _ in range(n_samples):
            resample = self.unique_shifts
            shifted_resample = resample - np.random.normal(mean, std_dev, len(resample)) * np.random.choice([-1, 1], len(resample))
            bootstrapped_means.append(shifted_resample)
        return np.array(bootstrapped_means)

    def calculate_bootstrap_ci(self, mean=.1, std_dev=.05,confidence_level=.95,n=1000):
        """
        Parameters
        ----------
        mean : float
            The mean of the normal distribution used to shift the data.
            std_dev : float
            The standard deviation of the normal distribution used to shift the data.
            confidence_level : float
            The confidence level for the confidence interval.
            n : int
            The number of bootstrap samples to generate.
            Returns
            -------
            lower_bound : float
            The lower bound of the confidence interval.
            upper_bound : float
            The upper bound of the confidence interval.
            unique_shifts : array of float
            The unique chemical shifts used to generate the confidence interval.
        """
        bootstrapped_means = self.__bootstrap_shifted(mean, std_dev, n_samples=n)
        lower_percentile = (1.0 - confidence_level) / 2.0 * 100
        upper_percentile = (1.0 + confidence_level) / 2.0 * 100
        lower_bound = np.percentile(bootstrapped_means, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrapped_means, upper_percentile, axis=0)
        return (lower_bound, upper_bound, self.unique_shifts)
    def plot_last_spectrum(self,show=True,plot_ci=True):
        """
        Parameters
        ----------
        show : bool
            Whether to display the plot of the predicted spectrum.
        plot_ci : bool
            Whether to plot the confidence intervals.
        Returns
        -------
        None
        """
        plt.plot(self.freqs,self.intensities)
        if plot_ci == True:
            conf_intervals = self.calculate_bootstrap_ci()
            lower_bound, upper_bound, unique_shifts = conf_intervals
            plt.vlines(lower_bound,0,.001,color='orange',linewidth=2)
            plt.vlines(upper_bound,0,.001,color='orange',linewidth=2)
            plt.vlines(unique_shifts,0,.001,color='k',linestyle='--')
        if show == True:
            plt.gca().invert_xaxis()
            plt.show()

"""
#CASCADE_H test
CPred = CASCADE_H()
CPred.get_spectrum('CCCCCC',show=True)

plt.figure(figsize=(12,8))
CPred.plot_last_spectrum(show=False,plot_ci=False)
plt.title('Spectrum of n-hexane')
plt.xlabel('Frequency (ppm)')
plt.ylabel('Intensity')
plt.gca().invert_xaxis()
plt.savefig('n-hexane_spectrum.png')
plt.show()
    
"""
"""
smiles_list = ['CCCCCCC','CCCCCCCC','CCCCCCCCCC','CCCCCCCCCCCC','CCCCCCCCCCCCCCCC']
names = ['n-heptane','n-octane','n-decane','n-dodecane','n-hexadecane']

for Smiles,name in zip(smiles_list,names):

    CPred = CASCADE_C()
    HPred = CASCADE_H()
    x_c,y_c=CPred.get_spectrum(Smiles,show=True)
    x_h,y_h=HPred.get_spectrum(Smiles,show=True)


    # Integration using trapezoidal rule
    # CH3: integrate from 0.9 to 1.2
    mask_ch3 = (x_h >= 0.9) & (x_h <= 1.2)
    sum_ch3 = np.trapz(y_h[mask_ch3], x_h[mask_ch3])

    # CH2: integrate from 1.2 to 1.5
    mask_ch2 = (x_h >= 1.2) & (x_h <= 1.5)
    sum_ch2 = np.trapz(y_h[mask_ch2], x_h[mask_ch2])

    # CH: integrate from 1.5 to 2.0
    mask_ch = (x_h >= 1.5) & (x_h <= 2.0)
    sum_ch = np.trapz(y_h[mask_ch], x_h[mask_ch])

    #normalize ch2 ch 2h3 such that they sum to one together
    sum_ch2_ch3 = sum_ch2 + sum_ch3 + sum_ch
    sum_ch2 = sum_ch2 / sum_ch2_ch3
    sum_ch3 = sum_ch3 / sum_ch2_ch3
    sum_ch = sum_ch / sum_ch2_ch3

    print(f"CH3: {sum_ch3:.2f}")
    print(f"CH2: {sum_ch2:.2f}")
    print(f"CH: {sum_ch:.2f}")



    shifts = HPred.shifts
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.plot(x_c,y_c)
    plt.title(f'13C NMR Spectrum of {name}')
    plt.xlabel('Frequency (ppm)')
    plt.ylabel('Intensity')
    plt.gca().invert_xaxis()
    plt.subplot(1,2,2)
    plt.plot(x_h,y_h)
    plt.vlines(shifts,0,.00005,color='k',linestyle='--')
    plt.title(f'1H NMR Spectrum of {name}')
    #add text for integration do CH3:CH2:CH ratio as a fake legend
    plt.plot([],[],label=f"CH3: {sum_ch3:.2f}")
    plt.plot([],[],label=f"CH2: {sum_ch2:.2f}")
    plt.plot([],[],label=f"CH: {sum_ch:.2f}")
    plt.legend()
    plt.xlabel('Frequency (ppm)')
    plt.ylabel('Intensity')
    plt.gca().invert_xaxis()
    plt.savefig(f'{name}_spectrum.png')
    plt.show()

    #ifft of the 1H NMR spectrum
    from scipy.fft import ifft

    y = y_h
    yf = ifft(y)
    #cut in half
    yf = yf[:len(yf)//2]
    plt.figure(figsize=(12,8))
    plt.plot(yf)
    plt.title(f'Inverse FFT of 1H NMR Spectrum of {name}')
    plt.savefig(f'{name}_ifft.png')
    plt.show()


"""