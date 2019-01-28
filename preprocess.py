#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ericyuan
requirement:
    (1) numpy='1.15.4'
    (2) matplotlib='3.0.2'
    (3) seaborn='0.9.0'
    (4) pandas='0.24.0'
    (5) scipy='1.1.0'
    (6) statsmodels='0.9.0'
    (7) sklearn='0.20.2'
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import interpolate
from statsmodels.stats.diagnostic import lilliefors 
from sklearn import preprocessing
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

class Distribution:
    '''
    class for distribution analysis
    Input：pandas.series/list/tuple like data
    '''
    def __init__(self, data):
        self.data = data
    # whole analysis
    def analysis(self, qqplot = True):
        print('-'*10, ' DESCRIPTION OF DATA ', '-'*10)
        print(pd.DataFrame(self.data).describe())
        print()
        print('-'*10, ' DISTRIBUTION ', '-'*10)
        plt.figure()
        mpl.rc('font',family='Times New Roman')
        sns.distplot(self.data)
        plt.xlabel('The range of data', fontsize = 16)
        plt.ylabel('Freqency', fontsize = 16)
        plt.title('Distribution', fontsize = 16)
        self.normTest()
        print('-'*10, ' DESCRIPTION OF DISTRIBUTION ', '-'*10)
        print("SKEWNESS: {0}".format(round(pd.Series(self.data).skew(), 4)))
        print("KURTOSIS: {1}".format(round(pd.Series(self.data).kurt(), 4)))
        if qqplot == True:
            print('-'*10, ' QQ-PLOT ', '-'*10)
            self.qqPlot()
    # qq plot
    def qqPlot(self):
        # cal
        S_mean = np.mean(self.data)
        S_std = np.std(self.data)
        S_return = pd.DataFrame(self.data)
        S_return_norm = (S_return - S_mean)*1.0/S_std
        S_return_norm.columns = ['return_norm']
        S_return_norm_sort = S_return_norm.sort_values(by = 'return_norm')
        S_return_norm_sort.index = range(len(S_return_norm_sort))
        S_return_norm_sort['percentage'] = [(i+1)*1.0/len(S_return_norm_sort) \
                          for i in range(len(S_return_norm_sort))]
        S_return_norm_sort['norm'] = S_return_norm_sort['percentage'].map(stats.norm(0,1).ppf)
        x = S_return_norm_sort.iloc[10:-10]['return_norm']
        y = S_return_norm_sort.iloc[10:-10]['norm']
        # plot
        plt.figure()
        plt.scatter(x, y, marker = ".")
        plt.scatter(x, x, marker = ".")
        plt.xlabel('{0} Theoretical Quantile {1}'.format('-'*10, '-'*10), fontsize = 16)
        plt.ylabel('{0} Sample Quantile {1}'.format('-'*10, '-'*10), fontsize = 16)
        plt.title('{0} QQ plot {1}'.format('-'*10, '-'*10), fontsize = 16)
    # normal distribution test
    def normTest(self, p = 0.05):
        # D'Agostino-Pearson Test, sample size 20-50
        if 20 < len(self.data) <= 50:
            p_value = stats.normaltest(self.data)[1]
            name = 'normaltest (D Agostino-Pearson)'
        elif len(self.data) <= 20:
            p_value = stats.shapiro(self.data)[1]
            name = 'shapiro'
        elif 300 >= len(self.data) >= 50:
            # Hubert Lilliefors
            p_value = lilliefors(self.data)
            name = 'lillifors'
        elif len(self.data) > 300:
            p_value = stats.kstest(self.data, 'norm')[1]
            name = 'KStest'
            
        print('-'*10, ' NORMAL TEST ', '-'*10)
        if p_value < p:
            print("USE: ", name)
            print("Conclusion: data are not normally distributed")
            return False
        else:
            print("USE: ", name)
            print("Conclusion: data are normally distributed")
            return True
class Scale:
    '''
    class for scale data
    Input：pandas.series/list/tuple/array like data
    return: numpy array
    '''
    def __init__(self):
        pass
    def minmax(self, data):
        data = np.array(data)
        minData = min(self.data)
        maxData = max(self.data)
        newData = (self.data - minData)/(maxData - minData)
        return newData
class Regular:
    '''
    class for regulization
    Input：pandas.series/list/tuple/array/dataframe like data (matrix)
    return: numpy array
    '''
    def __init__(self):
        pass
    def norm(self, data, norm = 'l2', axis = 1):
        aryData = np.array(data)
        X_normalized = preprocessing.normalize(aryData, norm = norm, axis = axis)
        return X_normalized
class Outlier():
    '''
    class for removing outliers
    Input：pandas.dataframe, col(list), up/low_bound(sigma)
    return: dataframe
    '''
    def __init__(self, data):
        self.data = data
    def __drawback(self, x, mean, std):
        '''
        transform
        '''
        if x >= self.up_bound:
            return self.up_bound*std + mean
        elif x <= self.low_bound:
            return self.low_bound*std + mean
        else:
            return x*std + mean
    def box(self, col, up_bound, low_bound, drawback = True):
        '''
        data: dataframe object
        col: columns' name, list
        '''
        # std
        self.up_bound = up_bound
        self.low_bound = low_bound
        for each_col in col:
            # mean
            mean = self.data[each_col].mean()
            # std
            std = self.data[each_col].std()
            self.data['help'] = self.data[each_col].map(lambda x: (x - mean)*1.0/std)
            if drawback:
                self.data[each_col] = self.data['help'].map(lambda x: self.__drawback(x, mean, std))
            else:
                self.data = self.data[(self.data['help'] <= self.up_bound)&(self.data['help'] >= self.low_bound)]
        del self.data['help']
        return self.data
class Missing(object):
    '''
    class for filling missing data
    Input：pandas.dataframe
           col(list like)
           value (Series/Dataframe(according to index) or value, list)
           method (list like (ffill, bfill, value))
    return: dataframe
    '''
    def __init__(self, data):
        '''
        data: dataframe (alter object df)
        '''
        self.data = data
    # fill missing data
    def fill(self, col, method, value):
        '''
        col: column, list
        method: method, list (ffill, bfill, value)
        value: Series/Dataframe(according to index) or value, list
        '''
        for each_col, each_method, each_value in zip(col, method, value):
            if each_method == 'value':
                self.data[each_col].fillna(value = each_value, inplace = True)
            else: 
                self.data[each_col].fillna(method = each_method, inplace = True)
        return self.data
    
    def interpolate(self, x, y, method):
        '''
        default:
        x: list like
        y: list like
        method: list like, 'nearest', 'zero', 'linear', 'quadratic'
                Specifies the kind of interpolation as a string 
                (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, 
                ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and 
                ‘cubic’ refer to a spline interpolation of zeroth, first, second 
                or third order; ‘previous’ and ‘next’ simply return the previous 
                or next value of the point) or as an integer specifying the order 
                of the spline interpolator to use. Default is ‘linear’.
        return: function for interpolation
        '''
        return interpolate.interp1d(x, y, method)
    
