# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:51:06 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

path="D:\AanshFolder\datasets\house.csv"
df = pd.read_csv(path)
print(df.head())
print(df.shape)

saleprice = df['SalePrice']

mean = saleprice.mean()
median = saleprice.median()
mode = saleprice.mode()
print("Mean:",mean," Median:",median," Mode:",mode[0])

#-------------------- histogram with mean,median,mode---------------------------
plt.figure(figsize=(8,4))
plt.hist(saleprice,bins=100,color='grey')
plt.axvline(mean,color='red',label='Mean')
plt.axvline(median,color='green',label='Median')
plt.axvline(mode[0],color='yellow',label='Mode')
plt.xlabel('saleprice')
plt.ylabel('frequency')
plt.legend()
plt.show()

print("Start Value:",saleprice.cumsum().head())
print("Min:",saleprice.min())
print("Max:",saleprice.max())
print("Range:",saleprice.max()-saleprice.min())
print("Variance:",saleprice.var())

from math import sqrt

std=sqrt(saleprice.var())
print("Standard deviation:",std)
print("Skewness:",saleprice.skew())
print("Kurtosis:",saleprice.kurt())

h=np.asarray(df['SalePrice'])
h=sorted(h)

#------------------------------------ Distribution Curve--------------------------------

fit = stats.norm.pdf(h,np.mean(h),np.std(h))
plt.plot(h,fit,'--',linewidth=2,label="Normal distribution with same mean and variance")
plt.hist(h,density=True,bins=100,label="Actual Distribution")
plt.legend()
plt.show()

#------------------------------ HeatMap -------------------------------
import seaborn as sns
get_ipython().run_line_magic('matplotlib','inline')
correlation = df[['LotArea','GrLivArea','GarageArea','SalePrice']].corr()
print(correlation)
sns.heatmap(correlation)

#---------------------------- BoxPlot ----------------------------------
print("Second Quartile(Q2):",saleprice.quantile(0.5))
q3=saleprice.quantile(0.75)
print("First Quartile(Q3):",q3)
q1=saleprice.quantile(0.25)
print("Third Quartile(Q1):",q1)
print("Inter Quartile Range(IQR):",q3-q1)
plt.figure(figsize=(8,4))
plt.boxplot(saleprice)

