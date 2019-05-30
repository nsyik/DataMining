#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
df= pd.read_csv('C:/Users/echo/Desktop/stock_df.csv')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',100)
df.isnull().sum()
df.columns
#data preprocessing：use replace function to reolace '-' and '0' to np.NAN, so that python can recognize
dfa=df.replace('-',np.NaN)
dfb=dfa.replace(0,np.NaN)
dfc=dfb.replace(0.00,np.NaN)
dfd=dfc.replace(0.000,np.NaN)
dfe=dfd.replace('0.000 / 0',np.NaN)
#deal the stockTime and stockDate variables, combine them into one variable 'stockDate', while still remaining the stockTime column
dfe['stockDate']=dfe['stockDate']+' '+dfe['stockTime']
dfe.stockDate=pd.to_datetime(dfe.stockDate)
#dfe


# In[10]:


#check the null,drop na and duplicates
dfe.isnull().sum()
dfe.dropna(inplace=True)
dfe.head()
dfe.drop_duplicates(subset = None, keep = 'first')
#shape to show final row x column 
dfe.shape 


# In[14]:


#according to stockCode, count the number of rows belonging to each company  
# choose ten companies with relative complete data. 
dfg=pd.DataFrame(dfe.groupby('stockCode').size())
print(dfg[dfg[0]>270].sort_values(by=[0]))
#pick several objective companies with least null 


# In[15]:


#according to stockCode, count the number of rows belonging to each company  
# choose ten companies with relative complete data.
#7181       274
#1066       275
#06516F     275
#4707       276
#06516G     276
#9873       276
#3662WC     277
#06515B     278
#3662       279
#06516J     280
df1=dfe[dfe['stockCode']=='7181']
df2=dfe[dfe['stockCode']=='1066']
df3=dfe[dfe['stockCode']=='06516F']
df4=dfe[dfe['stockCode']=='4707']
df5=dfe[dfe['stockCode']=='06516G']
df6=dfe[dfe['stockCode']=='9873']
df7=dfe[dfe['stockCode']=='3662WC']
df8=dfe[dfe['stockCode']=='06515B']
df9=dfe[dfe['stockCode']=='3662']
df10=dfe[dfe['stockCode']=='06515J']


# In[21]:


# the analysis of df6a,df7a,df8a 
df6a=pd.DataFrame(df6.drop_duplicates(subset = None, keep = 'first'))
df6a.columns=['stockCode','stockName', 'stockDate', 'stockTime', 'Open', 'High',
       'Low', 'Last', 'Chg', 'Chg', 'Vol', 'bug/v','SelldVol']
df7a=pd.DataFrame(df7.drop_duplicates(subset = None, keep = 'first'))
df7a.columns=['stockCode','stockName', 'stockDate', 'stockTime', 'Open', 'High',
       'Low', 'Last', 'Chg', 'Chg', 'Vol', 'bug/v','SelldVol']
df8a=pd.DataFrame(df8.drop_duplicates(subset = None, keep = 'first'))
df8a.columns=['stockCode','stockName', 'stockDate', 'stockTime', 'Open', 'High',
       'Low', 'Last', 'Chg', 'Chg', 'Vol', 'bug/v','SelldVol']
df6a.head(5)
df7a.head(5)
df8a.head(5)


# In[82]:


#PAA and SAX of variable 'Last' of df6a
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

# PAA transform (and inverse transform) of the data
n_paa_segments = 12
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
paa_dataset_inv = paa.inverse_transform(paa.fit_transform(df6a['Last']))
n_sax_symbols = 20
sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
sax_dataset_inv = sax.inverse_transform(sax.fit_transform(df6a['Last']))
# 1d-SAX transform
n_sax_symbols_avg = 9
n_sax_symbols_slope = 9
one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                alphabet_size_slope=n_sax_symbols_slope)
one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(df6a['Last']))

plt.figure()
plt.subplot(2, 2, 1)  # First, raw time series
plt.plot(df6a['Last'].ravel(), "b-")
plt.title("Raw time series")

plt.subplot(2, 2, 2)  # Second, PAA
plt.plot(df6a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(paa_dataset_inv[0].ravel(), "b-")
plt.title("PAA")

plt.subplot(2, 2, 3)  # Then SAX
plt.plot(df6a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(sax_dataset_inv[0].ravel(), "b-")
plt.title("SAX, %d symbols" % n_sax_symbols)

plt.subplot(2, 2, 4)  # Finally, 1d-SAX
plt.plot(df6a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
plt.title("1d-SAX, %d symbols (%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope,
                                          n_sax_symbols_avg,
                                          n_sax_symbols_slope))

plt.tight_layout()
plt.show()


# In[86]:


#PAA and SAX on variable of 'Last' of df7a 
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

# PAA transform (and inverse transform) of the data
n_paa_segments = 12
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
paa_dataset_inv = paa.inverse_transform(paa.fit_transform(df7a['Last']))
n_sax_symbols = 20
sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
sax_dataset_inv = sax.inverse_transform(sax.fit_transform(df7a['Last']))
# 1d-SAX transform
n_sax_symbols_avg = 9
n_sax_symbols_slope = 9
one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                alphabet_size_slope=n_sax_symbols_slope)
one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(df7a['Last']))

plt.figure()
plt.subplot(2, 2, 1)  # First, raw time series
plt.plot(df7a['Last'].ravel(), "b-")
plt.title("Raw time series")

plt.subplot(2, 2, 2)  # Second, PAA
plt.plot(df7a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(paa_dataset_inv[0].ravel(), "b-")
plt.title("PAA")

plt.subplot(2, 2, 3)  # Then SAX
plt.plot(df7a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(sax_dataset_inv[0].ravel(), "b-")
plt.title("SAX, %d symbols" % n_sax_symbols)

plt.subplot(2, 2, 4)  # Finally, 1d-SAX
plt.plot(df7a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
plt.title("1d-SAX, %d symbols (%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope,
                                          n_sax_symbols_avg,
                                          n_sax_symbols_slope))

plt.tight_layout()
plt.show()


# In[90]:


#PAA and SAX on variable 'Last'df8a 
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

# PAA transform (and inverse transform) of the data
n_paa_segments = 12
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
paa_dataset_inv = paa.inverse_transform(paa.fit_transform(df8a['Last']))
n_sax_symbols = 20
sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
sax_dataset_inv = sax.inverse_transform(sax.fit_transform(df8a['Last']))
# 1d-SAX transform
n_sax_symbols_avg = 9
n_sax_symbols_slope = 9
one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                alphabet_size_slope=n_sax_symbols_slope)
one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(df8a['Last']))

plt.figure()
plt.subplot(2, 2, 1)  # First, raw time series
plt.plot(df8a['Last'].ravel(), "b-")
plt.title("Raw time series")

plt.subplot(2, 2, 2)  # Second, PAA
plt.plot(df8a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(paa_dataset_inv[0].ravel(), "b-")
plt.title("PAA")

plt.subplot(2, 2, 3)  # Then SAX
plt.plot(df8a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(sax_dataset_inv[0].ravel(), "b-")
plt.title("SAX, %d symbols" % n_sax_symbols)

plt.subplot(2, 2, 4)  # Finally, 1d-SAX
plt.plot(df8a['Last'].ravel(), "b-", alpha=0.4)
plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
plt.title("1d-SAX, %d symbols (%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope,
                                          n_sax_symbols_avg,
                                          n_sax_symbols_slope))

plt.tight_layout()
plt.show()


# In[15]:


#final model : random decison tree(prediction)
#7181       274
#1066       275
#06516F     275
#4707       276
#06516G     276
#9873       276
#3662WC     277
#06515B     278
#3662       279
#06516J     280
import numpy as np
df11=df1.append([df2,df3,df4,df5,df6,df7,df8,df9,df10],ignore_index=True)
df11.columns=['stockCode','stockName', 'stockDate', 'stockTime', 'Open', 'High',
       'Low', 'Last', 'Chg', 'Chgp', 'Vol', 'bug/v','SelldVol']
df11=df11.apply(pd.to_numeric,errors='ignore')
df11['Vol']=df11['Vol'].astype(str)
df11['Vol'] = df11['Vol'].str.replace(',', '')
df11['Vol']=df11['Vol'].astype(int)
#correlation
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cross_validation import KFold
colormap=plt.cm.RdBu
fig=plt.figure(figsize=(14,12))
#plt.figure(figsize(14,12))
plt.title('Pearson correlation of Features',size=15)

sns.heatmap(df11[['Open', 'High','Low', 'Last', 'Chg','Chgp','Vol']].astype(float).corr(),linewidths=0.1,vmax=1.0,cmap=colormap,linecolor='white',annot=True)


# In[29]:


from sklearn.preprocessing import LabelEncoder
class_le=LabelEncoder()
df11['stockCode']=class_le.fit_transform(df11['stockCode'].values)
df11.columns
df11=df11[['stockCode','Open', 'High', 'Low', 'Last', 'Chg', 'Chgp', 'Vol']]
#final model to predict the vol using random forest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
X=df11.iloc[:,:-1].values
y=df11['Vol'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


#co-variance
pd.options.display.float_format = '{:,.3f}'.format
mean_vec = np.mean(X, axis=0)
cov_matrix = pd.DataFrame((X- mean_vec).T.dot((X - mean_vec)) / (X.shape[0]-1))
cov_matrix =cov_matrix.apply(pd.to_numeric,errors='ignore')
print('Covariance matrix \n%s' %cov_matrix)


from sklearn.ensemble import RandomForestRegressor
feat_labels=df11.iloc[:,:-1].columns[:]
tree=RandomForestRegressor(n_estimators=3000,random_state=5,n_jobs=2)
tree=tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)
tree=tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)
from sklearn.metrics import mean_squared_error
tree_train=mean_squared_error(y_train,y_train_pred)
tree_test=mean_squared_error(y_test,y_test_pred)
print('MSE %.3f/%.3f'%(tree_train,tree_test))
print(len(y_train),len(y_test))

tree.feature_importances_
importances=tree.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) % -*s %f" %(f+1,30,feat_labels[indices[f]],importances[indices[f]]))
fig=plt.figure()
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()

