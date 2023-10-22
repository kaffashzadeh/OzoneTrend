"""
This program is written to map the stations locations over city of Tehran.
"""
# imports python standard libraries
import os
import sys
import inspect
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx

font = {'family':'sans Serif', 'weight':'bold', 'size':10}
matplotlib.rc('font', **font)

# import local libraries
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir + '/../../../')
os.sys.path.insert(0, parentdir)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'najmeh.kaffashzadeh@gmail.com'


class Map:

    name = 'Map station location'

    def __init__(self, city='Tehran'):
        """
        This initializes the variables.

        Args:
            city(str): city name
        """
        self.location = city


    def read_meta(self):
        """
        It reads the meta data of the stations.
        """
        self.meta = pd.read_excel(sys.path[0] + '/Data/PollutantTehranStationsList.xlsx',
                                  index_col=0, header=0)

    def plot_map(self, type=None):
        """
        It plots map of the Tehran.

        Args:
            type(str): type of the plots
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        map = Basemap(projection='merc', llcrnrlat=35.55, urcrnrlat=35.85,
                      llcrnrlon=51.2, urcrnrlon=51.6, lat_0=35, lon_0=51, resolution='h', ax=ax)
        # map.drawmapboundary()
        # parallels = np.arange(35.55, 35.85, 0.1)
        # meridians = np.arange(51.2, 51.6, 0.1)
        # # labels = [left, right, top, bottom]
        # map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
        # map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)
        # map.readshapefile('../data/iran-roads/iran_roads',
        #                   'iran_roads', drawbounds=True, color='lightgrey')
        if type == 'corr':
            r = self.calc_corr(freq='M')
            print(r)
            x, y, vals = [], [], []
            for sta in self.meta.index:
                lon, lat = map(self.meta.loc[sta][4], self.meta.loc[sta][3])
                x.append(lon)
                y.append(lat)
                vals.append(r[self.meta.loc[sta][6]].values[0])
            plt.scatter(x, y, c=vals, cmap='gray_r', s=100, vmin=0, vmax=1, zorder=10, edgecolors='k')
            x1, y1 = map(51.35 , 35.68 )
            plt.scatter(x1, y1, c=r[-1], cmap='gray_r', marker='*', s=200, vmin=0, vmax=1, zorder=10, edgecolors='k')
            plt.colorbar()
            # fig.tight_layout()
            # plt.show()
            plt.savefig('../plots/StationCorrO3TMap_D.png', bbox_inches='tight')
            plt.close()
        else:
            for sta in self.meta.index:
                # lon, lat
                x, y = map(self.meta.loc[sta][4], self.meta.loc[sta][3])
                plt.plot(x, y, markersize=6, marker='o', color='k')
                plt.text(x-2000, y-800, str(int(self.meta.loc[sta][5])), color='k')
                x1, y1 = map(51.35 , 35.68 )
                plt.plot(x1, y1, markersize=10, marker='*', color='k')
            # fig.tight_layout()
            # plt.show()
            plt.savefig('../plots/StationMap.png', bbox_inches='tight')
            plt.close()

    def calc_corr(self, freq=None):
        """
        It calculates the correlation between ozone and temperature data series.
        """
        df_tmp = self.df.resample(freq).mean()
        o3 = df_tmp.loc[:, df_tmp.columns.get_level_values(1) == 'O3 (ppb)']
        # o3 = o3['2007':'2021'][('All stations', 'O3 (ppb)')]
        temp = self.df_temp[('All stations', 'ORG')].resample(freq).mean()
        return o3.corrwith(temp).abs()

    def read_data(self):
        """
        It reads the data at the given station.
        """
        self.path = sys.path[1] + '/../data/'
        self.df = pd.read_csv(self.path+'poll_teh.csv', index_col=0, parse_dates=True, header=[0,1])
        df_avg_o3 = pd.read_csv('../data/o3_avg_teh.csv', header=[0, 1], index_col=0, parse_dates=True)
        df_avg_nox = pd.read_csv('../data/nox_avg_teh.csv', header=[0, 1], index_col=0, parse_dates=True)
        self.df_temp = pd.read_csv('../data/at_me_mehr_spec.csv', header=[0, 1], index_col=0, parse_dates=True)
        self.df = pd.concat([self.df, df_avg_o3, df_avg_nox], axis=1)

    def run(self):
        """
        It creates the map of the station location.
        """
        self.read_meta()
        # self.plot_map()
        self.read_data()
        del self.df['Baharan']
        del self.df['District12']
        del self.df['Sheikh_safi']
        self.plot_map(type='corr')


if __name__=='__main__':
    Map(city='Tehran').run()