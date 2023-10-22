"""
This program is written for the statistical analysis of ozone trend.
"""
# imports python standard libraries
import os
import sys
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import datetime
import matplotlib.dates as md
from scipy.stats import f_oneway
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
import matplotlib.cm as cm
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import  scipy.stats as sps

# import local libraries
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir + '/../../../../PhD/TimeSeriesSpectralAnalysis/src/')
os.sys.path.insert(0, parentdir)
#from CheckDataSeries import Check

font = {'family':'sans Serif', 'weight':'normal', 'size':16} # size=16 'DejaVu Sans'
matplotlib.rc('font', **font)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'n.kaffashzadeh@gmail.com'


class TrdAna:

    name = 'Trend Analysis'

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
                           parse_dates=True, index_col=0, header=[0, 1])

    def sel_sum(self):
        """
        It select data in summer season.
        """
        return self.df[(self.df.index.month == 6) |
                       (self.df.index.month == 7) |
                       (self.df.index.month == 8)].dropna()

    def stand(self, df=None):
        """
        It standardizes data.

        Args:
            df(pd.DataFrame): data series

        Returns:
                standardized data series
        """
        return (df - df.mean()) / df.std()

    def mlr(self, predictant=None, predictors=None):
        """"
        It does multiple linear regressions.

        Args:
            predictant(str): independent variables
            predictors(str): dependent variable

        Returns:
                regression outputs
        """
        predictors_n = sm.add_constant(predictors)
        return sm.OLS(predictant, predictors_n).fit()

    def estimate_trend(self, var=None):
        """"
        It estimates the trend.

        Args:
            var(str): the variable name
        """
        try:
            series = self.df[var]
            series = series.resample('A').mean()
            df_tmp = series.dropna()
            df_tmp.index.name = 'date'
            df_tmp.name = 'LT'
            x = np.arange(1, len(df_tmp.index) + 1)
            y = df_tmp.values
            res = stats.theilslopes(y, x, 0.90)
            print(res)
        except TypeError:
            raise

    def plot_aic_f(self, aic=None, f_stat=None):
        """
        It plots aci and f value.

        Args:
            aic(list): AIC values
            f_stat(list): f values
        """
        predictors = [
                      'TMAX', 'TMEAN', 'TMIN', 'DRH', 'DWS',
                      'TMAX + TMEAN', 'TMAX + TMIN', 'TMAX + DRH', 'TMAX + DWS', 'TMEAN + TMIN',
                      'TMEAN + DRH', 'TMEAN + DWS', 'TMIN + DRH', 'TMIN + DWS', 'DRH + DWS',
                      'TMAX + TMEAN + TMIN', 'TMAX + TMEAN + DRH', 'TMAX + TMEAN + DWS', 'TMAX + DWS + DRH',
                      'TMEAN + TMIN + DRH', 'TMEAN + TMIN + DWS', 'TMEAN + DWS + DRH', 'TMIN + DWS + DRH',
                      'TMAX + TMEAN + TMIN + DRH', 'TMAX + TMEAN + TMIN + DWS', 'TMEAN + TMIN + DRH + DWS',
                      'TMAX + TMIN + DRH + DWS', 'TMAX + TMEAN + DRH + DWS',
                      'TMAX + TMEAN + TMIN + DRH + DWS'
                      ]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13., 8), facecolor='w')
        for x, y, lab in zip(f_stat, aic, predictors):
            ax.scatter(x, y, label=lab, s=100)
        colormap = plt.cm.gray
        colorst = [colormap(i) for i in np.linspace(0, 0.95, len(ax.collections))]
        for t, j1 in enumerate(ax.collections):
            j1.set_color(colorst[t])
        ax.legend(loc='upper center', fontsize='small', fancybox=True,
                  bbox_to_anchor=(0.5, -0.15), ncol=3, title='predictors')
        ax.set_xlabel('F-statistic')
        ax.set_ylabel('AIC')
        fig.tight_layout()
        # plt.show()
        plt.savefig('../plots/aic-f.jpg', bbox_inches='tight')
        plt.close()

    def calc_reg1(self):
        """"
        It calculates the regressions.
        """
        self.df = self.df.resample('Y').mean()
        # self.df['lag1-o3'] = (self.df['o3_sh'] + self.df['o3_se']).shift(1)
        df_std = self.stand(df=self.df).dropna()
        #df_std = self.df.dropna()
        #print(self.df.columns)
        #predictors = ['tma', 'lag1-o3']
        predictors = ['co']
        res = self.mlr(predictant=df_std['o3'],
                       predictors=df_std[predictors])
        print(res.summary())
        #print(res.pvalues)
        exit()
        try:
            aic_list = []
            f_list = []
            predictors= [
                         'tma', 'tme', 'tmi', 'rh', 'ws',
                          ['tma', 'tme'], ['tma', 'tmi'], ['tma', 'rh'], ['tma', 'ws'], ['tme', 'tmi'],
                          ['tme', 'rh'], ['tme', 'ws'], ['tmi', 'rh'], ['tmi', 'ws'], ['rh', 'ws'],
                          ['tma', 'tme', 'tmi'], ['tma', 'tme', 'rh'], ['tma', 'tme','ws'], ['tma', 'ws', 'rh'],
                          ['tme', 'tmi', 'rh'], ['tme', 'tmi', 'ws'], ['tme', 'ws', 'rh'], ['tmi', 'ws', 'rh'],
                          ['tma', 'tme','tmi', 'rh'], ['tma', 'tme','tmi', 'ws'], ['tme','tmi', 'rh', 'ws'],
                          ['tma', 'tmi', 'rh', 'ws'], ['tma', 'tme', 'rh', 'ws'],
                          ['tma', 'tme','tmi', 'rh', 'ws']
                         ]
            # self.df['lag1-o3'] = (self.df['o3_sh'] + self.df['o3_se']).shift(1)
            # predictors = [['tma', 'lag1-o3']]  # 'lag1-o3-sh', 'lag1-o3-se']]
            df_std = self.stand(df=self.df).dropna()
            for predictor in predictors:
                res = self.mlr(predictant=df_std['o3'],
                               predictors=df_std[predictor])
                aic_list.append(res.aic)
                f_list.append(res.fvalue)
                # print(res.summary(), res.pvalues[1], res.rsquared_adj)
            self.plot_aic_f(aic=aic_list, f_stat=f_list)
            exit()
            df_std['pre'] = res.params['const'] + \
                            df_std['tma'] * res.params['tma'] + \
                            df_std['lag1-o3'] * res.params['lag1-o3']
            self.df['pre'] = df_std['pre'] * self.df['o3'].std() + self.df['o3'].mean()
            df_std['resi'] = df_std['o3'] - df_std['pre']
            self.df['resi'] = df_std['resi'] * self.df['o3'].std() + self.df['o3'].mean()
            self.df['resi'].to_csv('../data/resi.csv') # adj_o3_avg_dma8_teh_spec.csv
        except TypeError:
            raise TypeError

    def check_linearity(self, data=None, var=None):
        """
        It checks the linearity between predictors and response variables.

        Args:
             data(pd.DataFrame): input data
             var(str): variable name
        """
        sns.lmplot(x='Actual', y='Predicted', data=data, scatter_kws={"color": "grey"},
                   line_kws={"color": "k"},fit_reg=False, height=7)
        line_coords = np.arange(data.min().min(), data.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                 color='k', linestyle='--')
        plt.ylabel('Predicted ' + var)
        plt.xlabel('Original ' + var)
        plt.title('Linearity Check')
        # plt.show()
        plt.savefig('../plots/linearity_reg_' + var + '.jpg', bbox_inches='tight')
        plt.close()

    def check_normality(self, data=None, var=None):
        """
        It checks normality (normal distributed) of the error terms.

        Args:
             data(pd.DataFrame): input data
             var(str): variable name
        """
        p_value = normal_ad(data['Residuals'])[1]
        p_value_thresh = 0.05
        plt.subplots(figsize=(12, 6))
        plt.title('Normality Check')
        sns.distplot(data['Residuals'], color = 'grey')
        # plt.show()
        plt.savefig('../plots/normality_reg_' + var + '.jpg', bbox_inches='tight')
        plt.close()

    def check_multicollinearity(self, data=None, var=None):
        """
        It check to see either predictors are correlated with each other or not.

        Args:
             data(pd.DataFrame): input data
             var(str): variable name
        """
        # VIF = [variance_inflation_factor(data, i) for i in range(data.shape[1])]
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(data, columns=['tma', 'lag1-o3']).corr(), annot=True)
        plt.title('Correlation of Variables')
        # plt.show()
        plt.savefig('../plots/collinearity_reg_' + var + '.jpg', bbox_inches='tight')
        plt.close()

    def check_autocorrelation(self, data=None, var=None):
        """
        It checks the autocorrection in the residuals.

        Args:
             data(pd.DataFrame): input data
             var(str): variable name
        """
        durbinWatson = durbin_watson(data['Residuals'])
        print('Durbin-Watson = ' + str('%.2f' % durbinWatson))
        # ax.text(1.1, 2.5, 'Durbin-Watson = ' + str('%.2f' % durbinWatson))

    def check_homoscedasticity(self, data=None, var=None):
        """
        It checks the homoscedasticity (variance of the error terms).

        Args:
             data(pd.DataFrame): input data
             var(str): variable name
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8., 5), facecolor='w')
        data = data.reset_index()
        plt.scatter(x=data.index, y=data['Residuals'], alpha=0.5, color='grey')
        plt.plot(np.repeat(0, data.index.max()), color='k', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Homoscedasticity Check')
        plt.ylabel('Residuals ' + var)
        plt.xlabel('Time periods')
        # plt.show()
        plt.savefig('../plots/homoscedasticity_reg_' + var + '.jpg', bbox_inches='tight')
        plt.close()

    def calc_reg2(self):
        """
        It calculates the regressions and tests regression assumptions.
        """
        self.df['lag1-o3'] = (self.df['o3_sh'] + self.df['o3_se']).shift(1)
        self.df = self.df.resample('Y').mean()
        df_std = self.stand(df=self.df).dropna()
        model = LinearRegression()
        predictors = ['tma', 'lag1-o3']
        predictors = ['co']
        model.fit(df_std[predictors], df_std['o3'])
        r2 = model.score(df_std[predictors], df_std['o3'])
        print('intercept = ', model.intercept_, 'coef = ', model.coef_)
        exit()
        # print('R2: {0}'.format(r2))
        predict = model.predict(df_std[predictors])
        res = pd.DataFrame({'Actual': df_std['o3'], 'Predicted': predict})
        res['Residuals'] = abs(res['Actual']) - abs(res['Predicted'])
        var = 'MDA8 (ppb)'
        self.check_linearity(data=res, var=var)
        self.check_normality(data=res, var=var)
        # self.check_multicollinearity(data=df_std, var=var)
        self.check_autocorrelation(data=res, var=var)
        self.check_homoscedasticity(data=res, var=var)
        exit()
        # Calculate residuals
        res['res'] = res['Residuals'] * self.df['o3'].std() + self.df['o3'].mean()
        res['res'].to_csv('resi.csv')

    def run(self):
        """
        It reads the observation data and analyses them.
        """
        sta = 'All stations'
        #y1 = '2007'
        #y2 = '2014-07-17'
        y1 = '2014-07-18'
        y2 = '2021'
        spec = 'LT'
        # Read all data
        dfs = self.read_obs_data(var = 'o3_avg_dma8_teh_spec')[y1:y2][(sta, spec)]
        dftme = self.read_obs_data(var='at_me_mehr_spec')[y1:y2][(sta, spec)]
        dftma = self.read_obs_data(var='at_ma_mehr_spec')[y1:y2][(sta, spec)]
        dftmi = self.read_obs_data(var='at_mi_mehr_spec')[y1:y2][(sta, spec)]
        dfrh= self.read_obs_data(var='rh_mehr_spec')[y1:y2][(sta, spec)]
        dfws = self.read_obs_data(var='ws_mehr_spec')[y1:y2][(sta, spec)]
        dfno = self.read_obs_data(var='no_avg_dmax_teh_spec')[y1:y2][(sta, spec)]
        dfno2 = self.read_obs_data(var='no2_avg_dmax_teh_spec')[y1:y2][(sta, spec)]
        dfnox = self.read_obs_data(var='nox_avg_dmax_teh_spec')[y1:y2][(sta, spec)]
        dfco = self.read_obs_data(var='co_avg_dmax_teh_spec')[y1:y2][(sta, spec)]
        dfs1 = self.read_obs_data(var='o3_avg_dma8_teh_spec')[y1:y2][(sta, 'SH')]
        dfs2 = self.read_obs_data(var='o3_avg_dma8_teh_spec')[y1:y2][(sta, 'SE')]
        dfadj_o3 = self.read_obs_data(var='adj_o3_avg_dma8_teh_spec')[y1:y2][(sta, spec)]
        self.df = pd.concat([dfs, dftme, dftma, dftmi, dfrh, dfws, dfno, dfno2, dfnox,
                             dfco, dfs1, dfs2, dfadj_o3], axis=1)
        self.df.columns = ['o3', 'tme', 'tma', 'tmi', 'rh', 'ws',
                           'no', 'no2', 'nox', 'co', 'o3_sh', 'o3_se', 'o3_adj']
        # contour plot
        # self.df = self.df[['o3_adj', 'co', 'nox']].resample('M').mean().dropna()
        # self.df.columns=['Adj_O3 (ppb)', 'CO (ppm)', 'NOx (ppb)']
        # Method 1
        # sns.relplot(data=self.df, x='CO (ppm)', y='NOx (ppb)', size='O3 (ppb)',
        # hue='O3 (ppb)', palette='gray_r')
        # plt.savefig('relplot.png', bbox_inches='tight')
        # plt.close()
        # Method 2
        # fig, ax = plt.subplots()
        # offset = 0.25
        # x = self.df['CO (ppm)']
        # y = self.df['NOx (ppb)']
        # z = self.df['Adj_O3 (ppb)']
        # xmin = 0#x.min() - offset
        # xmax = 10#x.max() + offset
        # ymin = 0#y.min() - offset
        # ymax = 500#y.max() + offset
        # X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        # positions = np.vstack([X.ravel(), Y.ravel()])
        # values = np.vstack([x, y])
        # kernel = sps.gaussian_kde(values, weights=z)
        # Z = np.reshape(kernel(positions).T, X.shape)
        # ax.imshow(np.rot90(Z), cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax], aspect='auto')
        # sns.scatterplot(data=self.df, x='CO (ppm)', y='NOx (ppb)', size='Adj_O3 (ppb)',
        #                 hue='Adj_O3 (ppb)', palette='gray_r', size_norm=(30,70))
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([ymin, ymax])
        # # ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        # plt.show()
        # exit()
        # plt.savefig('adjo3_nox_co_p1_org.png', bbox_inches='tight')
        # plt.close()
        # exit()
        # Method 3
        # fig, ax = plt.subplots()
        # x = self.df['CO (ppm)']
        # y = self.df['NOx (ppb)']
        # z = self.df['O3 (ppb)']
        # levels = [10, 20, 30, 40, 50, 60]
        # Z = self.df.pivot_table(index='CO (ppm)', columns='NOx (ppb)', values='O3 (ppb)').fillna(0).T.values
        # print(Z)
        # X_unique = np.sort(x.unique())
        # Y_unique = np.sort(y.unique())
        # X, Y = np.meshgrid(X_unique, Y_unique)
        # ax.contourf(X, Y, Z)
        # plt.show()

        # Estimate Trend
        #self.df = self.sel_sum().dropna()
        #self.estimate_trend(var='co')
        #exit()
        # Do regressions
        self.calc_reg1()
        self.calc_reg2()
        # exit()

if __name__ == '__main__':
    TrdAna().run()