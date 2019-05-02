# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:16:47 2019

@author: Utkarsh Kumar
"""

import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('POCs/Airline_encoders/encoders.csv')
data.columns
data.info()
data.describe()
data = data.dropna(how='all',axis=1)
data1 = data.drop(['date','updated_at'],axis=1)

""" Checking the failure and non failure scanners"""
never_fail =  data[data.err < 12]
routine_failure = data[data.err > 12]


"""Removing values -1 as its error or can say remove value 0 from output"""

(data['errf'] ==0).sum() #Counting 0 in errf
data1 = data1[data1.errf != 0] #Removing 0

sns.boxplot(y='errf',data=data1) #Visualising outliers
sns.boxplot(y='err',data=data1) #Visualising outliers

#Removing the outliers upto 3*SD
data1 = data1[(np.abs(stats.zscore(data1['errf'])) < 3)]
data1 = data1[(np.abs(stats.zscore(data1['err'])) < 3)]

sns.boxplot(y='errf',data=data1) #Visualising outliers
sns.boxplot(y='err',data=data1) #Visualising ouliers




""" Creating dummy variable for scanner"""
#Creating dummy
dummies = pd.get_dummies(data1['scanner'])
#Concat original with dumy
data2 = pd.concat([data1,dummies],axis=1)
#Drop original column
data2 = data2.drop(['scanner'],axis = 1)

data2['scanner'].value_counts()#Checking scanner count
data2['scanner'].nunique() #Same as above
data2['id'].value_counts()#Checking id count .. to confirm wheather we have to make dummy or not


(data1['min'] ==-1).sum()
(data1['max'] ==-1).sum()
(data1['minf'] ==-1).sum()
(data1['maxf'] ==-1).sum()
data1.isnull().sum()
""" Upto here our data is ready for the algorithm.. """

never_fail_cleaned =  data2[data2.err < 12]
routine_failure_cleaned = data2[data2.err > 12].id.value_counts() 


#Now we need to saperate data for predicting err and errf - Lnear regression

data_err = data2.drop(['minf','maxf','errf'],axis=1)
data_errf = data2.drop(['min','max','err'],axis=1)

#Now saperating dependent and indepenent variables for err prediction
err_X = data_err.drop(['err'],axis=1)
err_Y = data_err.iloc[:,3]

#Now saperating dependent and indepenent variables for err prediction
errf_X = data_errf.drop(['errf'],axis=1)
errf_Y = data_errf.iloc[:,4]

""" Doing the train test split for both the data """
from sklearn.model_selection import train_test_split
err_X_train,err_X_test,err_Y_train,err_Y_test = train_test_split(err_X,err_Y,test_size=0.2)

errf_X_train,errf_X_test,errf_Y_train,errf_Y_test = train_test_split(errf_X,errf_Y,test_size=0.2)

from sklearn import linear_model

""" Predicting for err"""
err_model = linear_model.LinearRegression(normalize=True).fit(err_X_train,err_Y_train)
err_model.coef_
err_model.intercept_

err_prediction = err_model.predict(err_X_test) #Actual prediction

from sklearn.metrics import mean_squared_error, r2_score  #importing evaluation metric
mean_squared_error(err_Y_test,err_prediction) # Calculating mean square error
r2_score(err_Y_test,err_prediction)  # Calculating r2 score

plt.scatter(err_model.predict(err_X_train),err_model.predict(err_X_train)- err_Y_train, c= 'b', s=40, alpha = 0.5)
plt.scatter(err_model.predict(err_X_test),err_model.predict(err_X_test)- err_Y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax= 50)
plt.title('Residual Plot')


""" err prediction, evaluation and visualization done"""

""" Predicting for errf"""
errf_model = linear_model.LinearRegression(normalize=True).fit(errf_X_train,errf_Y_train)
errf_model.coef_
errf_model.intercept_\

errf_prediction = errf_model.predict(errf_X_test) #Actual prediction

from sklearn.metrics import mean_squared_error, r2_score  #importing evaluation metric
mean_squared_error(errf_Y_test,errf_prediction) # Calculating mean square error
r2_score(errf_Y_test,errf_prediction)  # Calculating r2 score

plt.scatter(errf_model.predict(errf_X_train),errf_model.predict(errf_X_train)- errf_Y_train, c= 'b', s=40, alpha = 0.5)
plt.scatter(errf_model.predict(errf_X_test),errf_model.predict(errf_X_test)- errf_Y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax= 50)
plt.title('Residual Plot')



""" errf prediction, evaluation and visualization done"""

""" 
OBSERVATION IS THAT IF THE ERR VALUE IS ABOVEE 12% THEN THE ENCODER CAN FAIL ..
IF THE ERR VALUE IS BELOW 12% THEN THE ENCODER WILL NEVER FAIL..
SAME GOES WITH THE VALUE OF ERRF(0.5)  
"""
 