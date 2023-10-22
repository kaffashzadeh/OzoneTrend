"""
This program is written to plots the trend analysis of ozone time series.
"""
# imports python standard libraries
import os
import sys
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

# import local libraries
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir + '/../../../../PhD/TimeSeriesSpectralAnalysis/src/')
os.sys.path.insert(0, parentdir)

font = {'family':'sans Serif', 'weight':'normal', 'size':16}
matplotlib.rc('font', **font)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'n.kaffashzadeh@gmail.com'


class Plots:

    name = 'Plots'

    def __init__(self):
        """
        This initializes the variables.
        """
        self.df = None

    def read_obs_data(self, var=None):
        """
        It reads the observed data form the given file.
        """
        return pd.read_csv(sys.path[1] + '/../data/' + var + '.csv',
                           parse_dates=True, index_col=0, header=[0,1])

    def sel_sum(self):
        """
        It select data in summer season.
        """
        return self.df[(self.df.index.month == 6) |
                       (self.df.index.month == 7) |
                       (self.df.index.month == 8) ].dropna()

    def plot_pie_var(self, ax=None, vals=None):
        """
        It plots the relative frequency of the variance of the series.

        Args:
            ax(object): axes
        """
        ax.pie(vals, labels=['SH','SE','LT'],
                autopct='%1.2f', colors=['lightgrey', 'darkgrey', 'black'],
                wedgeprops=dict(linewidth=1, edgecolor='w'), textprops=dict(color='w'))

    def plot_exceed_box(self):
        """
        It plots the series as box plot and exceeded days as bar plot.
        """
        self.df = self.df[('All stations', 'ORG')]
        df_sum = self.sel_sum()
        df_org = df_sum['2007':'2021'].dropna()
        a = df_org.resample('Y').max()
        b = df_org.resample('Y').min()
        print(a-b)
        exit()
        # print(df_org['2008'].max())
        count = df_org[df_org > 70].resample('Y').count() # exceed
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13., 8), facecolor='w')
        vals = []
        for y in np.arange(2007, 2022,1):
            df_1y = df_org[df_org.index.year==y]
            vals.append(df_1y.values)
        ax.boxplot(vals, showmeans=True, showfliers=False,
                    medianprops=dict(linestyle='-', linewidth=1.5, color='black'),
                    meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='k'))
        labels_year = np.arange(2007,2022,1)
        ax.set_xticklabels(labels_year, color='k', rotation=90, ha='right')
        ax2 = ax.twinx()
        ax2.bar(np.arange(1,16,1), count.values, color='dimgray', width=0.35, align='center', alpha=0.5)
        ax.set_ylabel('MDA8 O${_3}$ (ppb)', fontsize=16)
        ax2.set_ylabel('exceedance days (#)', fontsize=16, color='dimgray', alpha=0.9)
        plt.savefig('../plots/mda8-exceed.jpg', bbox_inches='tight')
        plt.close()

    def plot_time_series_spect(self):
        """
        It plots the time series of the spectral components.
        """
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13., 8.), facecolor='w')
        df = self.df['2007':'2021']
        sta = 'All stations'
        for i,j,sp,t in [(0, 0, 'ORG', '(a)'), (0, 1, 'SH', '(b)'),
                         (1, 0, 'SE', '(c)'), (1, 1, 'LT', '(d)')]:
            df[(sta,sp)].plot(ax=ax[i,j], title=t,legend=False, c='k')
            ax[i,j].set_xlabel(' ')
        fig.tight_layout()
        # plt.show()
        plt.savefig('../plots/ts_spect_o3.jpg', bbox_inches='tight')
        plt.close()

    def plot_pie(self, y=None):
        """
        It plots the relative contributions of the spectral to total variances.

        Args:
            y(list): the years
        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8., 5), facecolor='w')
        for i, sy, ey, tit in [(0, y[0], y[1], '(a)'),
                               (1, y[2], y[3], '(b)')]:
              df_tmp = self.sel_sum().dropna()
              df_tmp = df_tmp[sy:ey]
              vals = df_tmp.var()
              vals= [vals[(self.sta, 'SH')],vals[(self.sta, 'SE')],vals[(self.sta, 'LT')]]
              self.plot_pie_var(ax=ax[i], vals=vals)
              ax[i].set_title(tit)
        # fig.tight_layout()
        # plt.show()
        plt.savefig('../plots/rel_var_spec_dma8_sum_pie.jpg', bbox_inches='tight')
        plt.close()

    def plot_ts(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8., 5), facecolor='w')
        ax2 = ax.twinx()
        self.df['o3_adj'].plot(ax=ax, color='k', linewidth=3, style='.')
        self.df['no2'].plot(ax=ax2, color='lightgrey', linewidth=3, style='.')
        # self.df['no2'].shift(365).plot(ax=ax2, color='lightgrey', linewidth=3, style='--')
        ax.set_ylabel('Adjusted MDA8 (ppb)', color='k')
        ax2.set_ylabel('NO$_{2}$ (ppb)', color='grey')
        # ax2.set_ylabel('NO (ppb)', color='grey')
        fig.tight_layout()
        # plt.show()
        # exit()
        plt.savefig('../plots/adj_o3_no2.jpg', bbox_inches='tight')
        plt.close()

    def run(self):
        """
        It reads the observation data and analyses them.
        """
        self.sta = 'All stations'
        y1, y2, y3, y4 = '2007', '2014-07-17', '2014-07-18', '2021'
        spect = 'ORG'
        # Read data and plot exceed, time series of spects, relative variance
        # self.df = self.read_obs_data(var='at_ma_mehr_spec')
        self.df = self.read_obs_data(var='o3_avg_dma8_teh_spec')
        self.plot_exceed_box()
        exit()
        self.plot_time_series_spect()
        exit()
        # self.plot_pie(y=[y1, y2, y3, y4])
        # Read adj ozone data and plot them versus precursors
        dfno = self.read_obs_data(var='no_avg_dmax_teh_spec')[y1:y4][(self.sta, spect)]
        dfno2 = self.read_obs_data(var='no2_avg_dmax_teh_spec')[y1:y4][(self.sta, spect)]
        dfnox = self.read_obs_data(var='nox_avg_dmax_teh_spec')[y1:y4][(self.sta, spect)]
        dfo3_adj = self.read_obs_data(var='adj_o3_avg_dma8_teh_spec')[y1:y4][(self.sta, spect)]
        self.df = pd.concat([dfo3_adj, dfno, dfno2, dfnox], axis=1)
        self.df.columns = ['o3_adj','no', 'no2','nox']
        # self.df = self.sel_sum().dropna()
        self.plot_ts()

if __name__ == '__main__':
    Plots().run()