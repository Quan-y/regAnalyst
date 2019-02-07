#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ericyuan
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_validate

class CRESULT:
    '''class for storage results '''
    def __init__(self):
        pass
class Reg:
    '''class for OLS estimation'''
    def __str__(self):
        return("Regression class for outputting better results")
    # R sqaure
    def __rsquare(self, y, yhat):
        SST = sum((y - np.mean(y))**2)
        SSReg = sum((yhat - np.mean(y))**2)
        Rsquared = SSReg/SST
        return Rsquared
    def ols(self, x, y, model, fit_intercept, 
            cv = 5, scoring = 'explained_variance'):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        # Create linear regression object
        if model == 'ols':
            OLS = linear_model.LinearRegression(fit_intercept = fit_intercept)
        elif model == 'ridge':
            OLS = linear_model.Ridge()
        elif model == 'lasso':
            OLS = linear_model.Lasso()
        elif model == 'elasticNet':
            OLS = linear_model.ElasticNet(random_state=0)
        # Train the model using the training sets
        OLS.fit(x, y)
        # get the results
        OLSresult = CRESULT()
        coefs = OLS.coef_
        scores = cross_validate(OLS, x, y, scoring=scoring, \
                                cv=cv, return_train_score=False)
        # store the results
        OLSresult.score= scores['test_'+scoring]
        OLSresult.coefs = coefs
        # fitted value
        fitted = OLS.predict(x.values)
        OLSresult.fitted = fitted
        OLSresult.r2 = self.__rsquare(y.values, fitted)[0]
        return OLSresult



