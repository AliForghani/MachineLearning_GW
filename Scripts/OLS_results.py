import numpy as np
import tensorflow as tf
import os
from sklearn import preprocessing
import statsmodels.api as sm
import pandas as pd
raw_csv_data = pd.read_csv("data/Data.out",delim_whitespace=True)
raw_csv_data["ratio"]=raw_csv_data["Ext"]/raw_csv_data["Inj"]
scaled_data = preprocessing.scale(raw_csv_data)
Scaled_DF=pd.DataFrame(scaled_data,columns=raw_csv_data.columns)
y1=Scaled_DF[['REN_1_1', 'REN_1_2', 'REN_2_1', 'REN_2_2', 'REN_3_1', 'REN_3_2', 'REN_4_1', 'REN_4_2']]
y=Scaled_DF['REN_3_2']
x1=Scaled_DF[['K', 'Inj', 'Por', 'b', 'CHD', 'ratio', 'DSP']]
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())



