import itertools
import datetime as dt

years = [[2012], [2013], [2014], [2015], [2016], [2017], [2018], [2019], [2020]]
dates = [
    [[1, 2], [1, 16], [2, 20], [4, 6], [5, 28], [7, 4], [9, 3], [11, 22], [12, 25]],
    [[1, 1], [1, 21], [2, 18], [3, 29], [5, 27], [7, 4], [9, 2], [11, 28], [12, 25]],
    [[1, 1], [1, 20], [2, 17], [4, 18], [5, 26], [7, 4], [9, 1], [11, 27], [12, 25]],
    [[1, 1], [1, 19], [2, 16], [4, 3], [5, 25], [7, 4], [9, 7], [11, 26], [12, 25]],
    [[1, 1], [1, 18], [2, 15], [3, 25], [5, 30], [7, 4], [9, 5], [11, 24], [12, 25]],
    [[1, 2], [1, 16], [2, 20], [4, 14], [5, 29], [7, 4], [9, 4], [11, 23], [12, 25]],
    [[1, 1], [1, 15], [2, 19], [3, 30], [5, 28], [7, 4], [9, 3], [11, 22], [12, 25]],
    [[1, 1], [1, 21], [2, 18], [4, 19], [5, 27], [7, 4], [9, 2], [11, 28], [12, 25]],
    [[1, 1], [1, 20], [2, 17], [4, 10], [5, 25], [7, 3], [9, 7], [11, 26], [12, 25]]]

def generate_holidays():
    dates_combined = [list(itertools.product(year, date)) for year, date in zip(years, dates)]
    holidays = list()
    for ele in dates_combined:
        holiday = [dt.date(year=t[0], month=t[1][0], day=t[1][1]) for t in ele]
        holidays.extend(holiday)
    return holidays

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


class PlotKernelDensityEstimator:
    """
    A object for plotting a series of KDE with given bandwidths and kernel functions
    """

    def __init__(self, data_points):

        # delete -inf, inf and nan numbers
        data_points = list(filter(lambda x: x != -np.inf, data_points))

        if isinstance(data_points, list):
            data_points = np.asarray(data_points)

        # &(data_points != np.inf)&(data_points != np.nan)

        self.data_points = data_points

        # Default parameters
        self.kernel = 'epanechnikov'
        self.x_grid = np.linspace(min(data_points), max(data_points), int(len(data_points) / 1))
        # self.x_plot = np.linspace(0, 1, 1000)
        # self.file_name = file_name

    def bandwidth_search(self, method):
        # x_grid = np.linspace(min(self.data_points), max(self.data_points), int(len(self.data_points)/10))
        print('Searching Optimal Bandwidth...')
        if method == 'gridsearch':
            grid = GridSearchCV(KernelDensity(),
                                {
                                    'bandwidth': self.x_grid,
                                },
                                cv=5)
            grid.fit(self.data_points.reshape(-1, 1))
            self.band_width = grid.best_params_['bandwidth']
        elif method == 'silverman':
            std = self.data_points.std()
            n = len(self.data_points)
            self.band_width = 1.06 * std * np.power(n, -1 / 5)
        return self.band_width

    def pdf_calcualtion(self, **kwargs):
        if 'bandwidth' in kwargs:
            self.bandwidth = kwargs['bandwidth']
        else:
            self.bandwidth = self.bandwidth_search(kwargs['method'])
            print(f"Bandwidth search method: {kwargs['method']}")
        kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X=self.data_points.reshape(-1, 1))
        self.pdf = np.exp(kde.score_samples(self.x_grid.reshape(-1, 1)))
        # self.log_densities[f'{bandwidth}'] = log_dens
        return self.pdf

    def plot_curve_hist_kde(self, bin_num=None, hist_density=True, bandwidth=None, method='silverman', linewidth=1,
                            linestyle='-', color='red'):
        if bandwidth is None:
            self.pdf_calcualtion(method=method)
        else:
            self.pdf_calcualtion(bandwidth=bandwidth)
        # fig = plt.figure(figsize=(15, 7))
        plt.hist(self.data_points, bins=bin_num, density=hist_density)
        plt.plot(self.x_grid, self.pdf, linestyle=linestyle, linewidth=linewidth, color=color)
