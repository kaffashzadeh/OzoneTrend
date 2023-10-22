"""
This program is written to count the data (of ozone and nox) in each year
and plots them as heatmap.
"""
# imports python standard libraries
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib

font = {'family':'sans Serif', 'weight':'normal', 'size':10}
matplotlib.rc('font', **font)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'najmeh.kaffashzadeh@gmail.com'


class Check:

    name = 'Check Data time series'

    def __init__(self):
        """
        This initializes the variables.
        """
        self.df = None
        self.stations = ['Mahallati', 'Masoudieh', 'Piroozi', 'Punak', 'Ray', 'Rose Park',
                         'Sadr', 'Setad bohran', 'Shad abad', 'Sharif uni', 'Tarbiat Modares',
                         'District 22', 'Aqdasiyeh', 'Darrous', 'District 2', 'District 4',
                         'District 10', 'District 11', 'District 16', 'District 19', 'District 21',
                         'Fath square', 'Golbarg', 'Geophysics', 'All stations']

    def read_meta(self):
        """
        It reads the meta data of the stations.
        """
        self.meta = pd.read_excel(sys.path[0] + '/../data/PollutantTehranStationsList.xlsx',
                                  index_col=0, header=0)

    def plot_heatmap(self):
        """
        It plots the heatmap.
        """
        df_cy = self.count_data()
        df_cy_o3 = df_cy.loc[:, df_cy.columns.get_level_values(1) == 'O3 (ppb)']
        df_cy_nox = df_cy.loc[:, df_cy.columns.get_level_values(1) == 'NOx (ppb)']
        fig, ax = plt.subplots(ncols=2, nrows=1, facecolor='w', figsize=(13,5))
        sp = sns.heatmap(df_cy_o3, xticklabels=self.stations,
                         yticklabels=df_cy_o3.index.year,
                         cmap="Greys", linewidths=.5, ax=ax[0], cbar=False)
        sp.invert_yaxis()
        ax[0].set_xlabel('stations name')
        ax[0].set_ylabel('years')
        ax[0].set_title('(a)')
        sp = sns.heatmap(df_cy_nox, xticklabels=self.stations,
                         yticklabels=df_cy_nox.index.year,
                         cmap="Greys", linewidths=.5, ax=ax[1])
        sp.invert_yaxis()
        ax[1].set_xlabel('stations name')
        ax[1].set_ylabel('years')
        ax[1].set_title('(b)')

    def plot_corrmap(self):
        """
        It plots the heatmap.
        """
        df_cy = self.df.resample('Y').mean()
        df_cy_o3 = df_cy.loc[:, df_cy.columns.get_level_values(1) == 'O3 (ppb)'].corr().abs()
        df_cy_nox = df_cy.loc[:, df_cy.columns.get_level_values(1) == 'NOx (ppb)'].corr().abs()
        fig, ax = plt.subplots(ncols=2, nrows=1, facecolor='w', figsize=(13,5))
        mask = np.triu(np.ones_like(df_cy_o3, dtype=bool))
        sp = sns.heatmap(df_cy_o3, xticklabels=self.stations,
                         yticklabels=self.stations, mask=mask,
                         cmap="Greys", linewidths=.5,
                         vmax=1, vmin=0, ax=ax[0], cbar=False)
        sp.invert_yaxis()
        ax[0].set_xlabel('')
        ax[0].set_ylabel('')
        ax[0].set_title('(a)')
        mask = np.triu(np.ones_like(df_cy_nox, dtype=bool))
        sp = sns.heatmap(df_cy_nox, xticklabels=self.stations,
                         yticklabels=self.stations, mask=mask,
                         vmax=1, vmin=0, cmap="Greys", linewidths=.5, ax=ax[1])
        sp.invert_yaxis()
        ax[1].set_xlabel('')
        ax[1].set_ylabel('')
        ax[1].set_title('(b)')

    def count_data(self):
        """
        It counts the available data in each year.

        Returns:
               the counted data as a pd.DataFrame
        """
        return self.df.resample('Y').count()

    def save_fig(self, fn='output'):
        """
        It saves figure.

        Args:
            fn(str): file name
        """
        plt.savefig('../plots/'+fn+'.png', bbox_inches='tight')
        plt.close()

    def read_data(self):
        """
        It reads the data at the given station.
        """
        self.path = sys.path[0] + '/../data/'
        self.df = pd.read_csv(self.path+'poll_teh.csv', index_col=0, parse_dates=True, header=[0,1])
        df_avg_o3 = pd.read_csv('../data/o3_avg_teh.csv', header=[0, 1], index_col=0, parse_dates=True)
        df_avg_nox = pd.read_csv('../data/nox_avg_teh.csv', header=[0, 1], index_col=0, parse_dates=True)
        self.df = pd.concat([self.df, df_avg_o3, df_avg_nox], axis=1)

    def run(self):
        """
        It reads the observation data and plots them.
        """
        self.read_data()
        self.read_meta()
        # station with no ozone data
        del self.df['Baharan']
        del self.df['District12']
        del self.df['Sheikh_safi']
        # self.plot_corrmap()
        # self.save_fig(fn='corr_y')
        self.plot_heatmap()
        self.save_fig(fn='count_y')

if __name__ == '__main__':
    Check().run()