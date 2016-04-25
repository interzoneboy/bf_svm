"""
This module implements various tests of SVMs for density modeling with simulated (or otherwise really)
fusion data.
"""

import numpy as np
import scipy
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

print "Read data file"

d_x = d[['aucs1','aucs2','corrs']]
d_y = d['improved']

scaler = preprocessing.StandardScaler()
imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer.fit(d_x)
scaler.fit(imputer.transform(d_x))
d_x_scaled = scaler.transform(imputer.transform(d_x))

print "Data scaled and imputed"

clf = svm.SVC()
clf.fit(d_x_scaled, d_y)

print "SVM fit"

xx, yy = np.meshgrid(np.linspace(0.5, 1.0, 200),
            np.linspace(0.5, 1.0, 200))

zz = scipy.full(xx.shape, 0.0)

Z_pre = clf.decision_function(scaler.transform(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]))

print "Decision function calculated. Plotting..."

Z = Z_pre.reshape(xx.shape)

# To show in imshow below, just take an example slice through the middle of the correlation axis
# ::UPDATE:: Now we're just generating Z_pre as a slice through corr axis, set using the scipy.full call up there.
Z_show = Z

plt.imshow(Z_show, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z_show, levels=[0], linewidths=2, linetypes='--')


#plt.scatter(d_x.iloc[:,0], d_x.iloc[:,1], s=30, c=d_y, cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([0.5, 1.0, 0.5, 1.0])
plt.show()

