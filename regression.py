#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ericyuan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
from sklearn.model_selection import cross_validate

from pykalman import KalmanFilter

class CRESULT:
    '''class for storage results '''
    def __init__(self):
        pass
class Reg:
    '''class for OLS estimation'''
    def __str__(self):
        return("Regression class for outputting better results")
    # R sqaure
    def rsquare(self, y, yhat):
        SST = sum((y - np.mean(y))**2)
        SSReg = sum((yhat - np.mean(y))**2)
        Rsquared = SSReg/SST
        return Rsquared
    def ols(self, x, y, cv = 5, scoring = ['explained_variance', 'neg_mean_squared_error']):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        # Create linear regression object
        OLS = linear_model.LinearRegression(fit_intercept = False)
        # Train the model using the training sets
        OLS.fit(x, y)
        # get the results
        OLSresult = CRESULT()
        coefs = OLS.coef_
        scores = cross_validate(OLS, x, y, scoring=scoring, \
                                cv=cv, return_train_score=False)
        # store the results
        OLSresult.e1 = scores['test_'+scoring[0]]
        OLSresult.e2 = scores['test_'+scoring[1]]
        OLSresult.coefs = coefs
        # fitted value
        fitted = OLS.predict(x.values)
        OLSresult.fitted = fitted
        OLSresult.r2 = self.rsquare(y.values, fitted)[0]
        return OLSresult
# remove outliers
class Outlier(object):
    def __init__(self, data, up_bound, low_bound):
        '''
        data: dataframe
        '''
        self.data = data
        # x std
        self.up_bound = up_bound
        # x std
        self.low_bound = low_bound
    def __transform(self, x, mean, std):
        '''
        transform
        '''
        if x >= self.up_bound:
            return self.up_bound*std + mean
        elif x <= self.low_bound:
            return self.low_bound*std + mean
        else:
            return x*std + mean
    def remove(self, col, transform = True):
        '''
        data: dataframe object
        col: columns' name, list
        '''
        for each_col in col:
            # mean
            mean = self.data[each_col].mean()
            # std
            std = self.data[each_col].std()
            self.data['help'] = self.data[each_col].map(lambda x: \
                     (x - mean)*1.0/std)
            if transform:
                self.data[each_col] = self.data['help'].map(lambda x: \
                         self.__transform(x, mean, std))
            else:
                self.data = self.data[(self.data['help'] <= self.up_bound)&(self.data['help'] >= self.low_bound)]
        del self.data['help']
        return self.data
class Kalman:
    '''
    Kalman filter for Regression Estimation
    Input: all dataset including two assets which we want to explore
    Output: Analysis Result
    Notice: index must be time format
    '''
    def __init__(self, data):
        self.data = data
    def analysis(self, asset1, asset2, visual = False):
        # Kalman Filter
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.vstack([self.data[asset1], \
                             np.ones(self.data[asset1].shape)]).T[:, np.newaxis]
        # set parameters
        kf = KalmanFilter(n_dim_obs = 1, n_dim_state = 2,
                          initial_state_mean=np.zeros(2),
                          initial_state_covariance = np.ones((2, 2)),
                          transition_matrices = np.eye(2),
                          observation_matrices = obs_mat,
                          observation_covariance = 1.0,
                          transition_covariance = trans_cov)
        # calculate rolling beta and intercept
        state_means, state_covs = kf.filter(self.data[asset2].values)
        beta_slope = pd.DataFrame(dict(slope=state_means[:, 0], \
                         intercept=state_means[:, 1]), index = self.data.index)
        if visual == True:
            # visualization for correlation
            cm = plt.cm.get_cmap('jet')
            colors = np.linspace(0.1, 1, len(self.data))
            sc = plt.scatter(self.data[asset1], self.data[asset2], \
                             s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
            cb = plt.colorbar(sc)
            cb.ax.set_yticklabels([str(p.date()) for p in \
                                   self.data[::len(self.data)//9].index]);
            plt.xlabel(asset1)
            plt.ylabel(asset2)
            plt.show()
           
            # plot beta and slope
            beta_slope.plot(subplots = True)
            plt.show()

            # visualize the correlation between assest prices over time
            cm = plt.cm.get_cmap('jet')
            colors = np.linspace(0.1, 1, len(self.data))
            sc = plt.scatter(self.data[asset1], self.data[asset2], \
                             s=50, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
            cb = plt.colorbar(sc)
            cb.ax.set_yticklabels([str(p.date()) for p in self.data[::len(self.data)//9].index]);
            plt.xlabel(asset1)
            plt.ylabel(asset2)
            
            # add regression lines
            step = 5
            xi = np.linspace(self.data[asset1].min(), self.data[asset1].max(), 2)
            colors_l = np.linspace(0.1, 1, len(state_means[::step]))
            for i, beta in enumerate(state_means[::step]):
                plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[i]))
        return beta_slope
# from index name based on dict, get industry and rating
def getX(name, dicty):
    res = dicty[dicty['code'] == name]['name'].values[0].split(' ')
    ind = res[0]
    rating = res[1]
    return(ind, rating)
# parameters estimation
def est(data, indexName, dicty):
    ind, rating = getX(indexName, dicty)
    bondReg = Reg()
    try:
        resUtilitiesBBB = bondReg.ols(x = data[[ind, rating]], y = data[indexName])
        return(resUtilitiesBBB.coefs[0], ind + ' ' + rating, indexName, \
               resUtilitiesBBB.e1, resUtilitiesBBB.e2, resUtilitiesBBB.r2)
    except:
        return 0
# plot
#def bondPlot(industry, rating, index, res, data):
#    y = data[index]
#    slope_rating = res[res['indexname'] == index]['ratingCoef']
#    slope_sec = res[res['indexname'] == index]['indRes']
#    intercept = res[res['indexname'] == index]['intercept']
#    
#    x = slope_sec*data[industry] + slope_rating*data[rating] + intercept
#    print(slope_rating, slope_sec, intercept, data[industry], data[rating])
#    plotdf = pd.DataFrame({'x':x, 'y':y})
#    sns.regplot('x','y',plotdf)
#    plt.xlabel('x')
#    plt.ylabel('y')
#    return plotdf
def bondplot(industry, rating, index, data):
    reg = Reg()
    res = reg.ols(data[[industry, rating]], data[index])
    # plt.scatter(new_df['IGUUIA3M Index'], res.fitted.T.tolist()[0])
    sns.regplot(data[index], res.fitted.T.tolist()[0])
    print(np.corrcoef(data[index], res.fitted.T.tolist()[0]))
    plt.show()
    plotdata = pd.DataFrame({'x': data[index], 'y': res.fitted.T.tolist()[0]})
    plotdata.index = data['date']
    return plotdata

def rollingReg(end, step, data, x, y, start = 0):
    length = len(data)
    reg = Reg()
    ax = plt.gca()
#    minx = 0
#    maxx = 0
    # loop
    while end < length:
        regdata = data.iloc[start:end]
        res = reg.ols(regdata[x], regdata[y])
        # max, min value for fitted
#        minx = min(minx, min(res.fitted))
#        maxx = max(maxx, max(res.fitted))
        # plot
        sns.regplot(regdata[y], res.fitted.T.tolist()[0], ax = ax)
        # update
        start += step
        end += step
    plt.xlim(-5, 5)
    plt.ylim(min(data[y]), max(data[y]))
    
    
    