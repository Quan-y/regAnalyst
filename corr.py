#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ericyuan
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class Corr(object):
    def __init__(self, data):
        self.data = data
    def analyze(self, col = None):
        if col == None:
            col = self.data.columns
        return self.data[col].corr()
    def visual(self, col = None):
        if col == None:
            col = self.data.columns
        f, ax = plt.subplots(figsize=(10, 8))
        corDf = self.data[col].corr()
        sns.heatmap(corDf, mask=np.zeros_like(corDf, dtype=np.bool), \
                    cmap=sns.diverging_palette(220, 10), square=True, ax=ax)
