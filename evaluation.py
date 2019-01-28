#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:04:25 2019

@author: ericyuan
"""
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from . import regression
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

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

# rolling simple regression
def rollingReg(end, step, data, x, y, start = 0):
    '''
    data: dataframe
    x: string
    y: string
    '''
    length = len(data)
    reg = regression.Reg()
    ax = plt.gca()
    # loop
    while end < length:
        regdata = data.iloc[start:end]
        res = reg.ols(regdata[x], regdata[y])
        # plot
        yhat = res.fitted.T.tolist()[0]
        sns.regplot(regdata[y], yhat, ax = ax)
        # update
        start += step
        end += step
    plt.xlim(-abs(min(yhat))*10, max(yhat)*10)
    plt.ylim(min(data[y]), max(data[y]))
    
    