# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:24:52 2023

@author: 91808
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np


df = pd.read_csv('D:\gdataset.csv')


df['Change %'] = df['Change %'].str.replace('%', '').astype(float)


df['Avg. Volume'] = df['Avg. Volume'].str.replace(',', '').astype(float)


x = df[['Close', 'High', 'Low','Change %', 'Avg. Volume']]  
y = df[['Open']]


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
x = imp.fit_transform(x)
y = imp.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


lsr = LinearRegression()
lsr.fit(x_train, y_train)


pred = lsr.predict(x_test)
