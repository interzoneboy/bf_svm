"""
This module implements various tests of SVMs for density modeling with simulated (or otherwise really)
fusion data.
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import pandas as pd

from sklearn.cross_validation import ShuffleSplit
from sklearn import preprocessing
from sklearn import svm



d = pd.read_csv("./exampleData.csv")
nuMax = d[['aucs1','aucs2']].max(axis=1)
d['aucsMax'] = nuMax
d['improved'] = d.aucsFused > d.aucsMax
d['improvement'] = d.aucsFused - d.aucsMax

d_x = d[['aucs1','aucs2','corrs']]
d_y = d['improved']

scaler = preprocessing.StandardScaler()
imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer.fit(d_x)
scaler.fit(imputer.transform(d_x))
d_x_scaled = scaler.transform(imputer.transform(d_x))

clf = svm.SVC()
clf.fit(d_x_scaled, d_y)

xx, yy = np.meshgrid(np.linspace(0.5, 1.0, 500),
        np.linspace(0.5, 1.0, 500))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx,yy,Z, levels=[0], linewidths=2, linetypes='--')


plt.scatter(d_x.iloc[:,0], d_x.iloc[:,1], s=30, c=d_y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([0.5, 1.0, 0.5, 1.0])
plt.show()

