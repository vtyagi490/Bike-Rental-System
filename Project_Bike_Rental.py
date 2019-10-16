# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:16:06 2019

@author: pc
"""

#Import libraries
import os
import pandas as pd
import numpy as np
​
#import libraries for plots
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline  
#Set working directory
os.chdir("C:\\Users\\pc\\Desktop\\R\\projects\\Bike-Rental-master")
print(os.getcwd())
#Read the csv file
day = pd.read_csv("day.csv", sep=",")
#Get the number of rows and columns
day.shape
#Get first 5 rows
print(day.head())
​
#Get the data types of variables
print(day.dtypes)
#Create a new dataframe containing required columns and creating new columns
df = day.copy()
df.head()
​
#Create new columns
df['actual_temp'] = day['temp'] * 39
df['actual_feel_temp'] = day['atemp'] * 50
df['actual_windspeed'] = day['windspeed'] * 67
df['actual_hum'] = day['hum'] * 100
​
df['actual_season'] = day['season'].replace([1,2,3,4],["Spring","Summer","Fall","Winter"])
df['actual_yr'] = day['yr'].replace([0,1],["2011","2012"])
df['actual_holiday'] = day['holiday'].replace([0,1],["Working day","Holiday"])
df['actual_weathersit'] = day['weathersit'].replace([1,2,3,4],["Clear","Cloudy/Mist","Rain/Snow/Fog","Heavy Rain/Snow/Fog"])
#Check the data types od variables
df.dtypes
​
#Change the data types
df['weathersit'] = df['weathersit'].astype('category')
df['holiday'] = df['holiday'].astype('category')
df['yr'] = df['yr'].astype('category')
df['season'] = df['season'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['mnth'] = df['mnth'].astype('category')
df['actual_season'] = df['actual_season'].astype('category')
df['actual_yr'] = df['actual_yr'].astype('category')
df['actual_holiday'] = df['actual_holiday'].astype('category')
df['actual_weathersit'] = df['actual_weathersit'].astype('category')
​
df.dtypes
#Check the count of values of categorical variables
print(df.workingday.value_counts())
print(df.weekday.value_counts())
print(df.mnth.value_counts())
print(df.actual_yr.value_counts())
print(df.actual_holiday.value_counts())
print(df.actual_weathersit.value_counts())
#Check if there are missing values
df.isnull().sum()
#Check the bar graph of categorical Data using factorplot
sns.set_style("whitegrid")
sns.factorplot(data=df, x='actual_season', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='actual_weathersit', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='workingday', kind= 'count',size=4,aspect=2)
​
#Check the distribution of numerical data using histogram
plt.hist(data=df, x='actual_temp', bins='auto', label='Temperature')
plt.xlabel('Temperature in Celcius')
plt.title("Temperature Distribution")
#Check the distribution of numerical data using histogram
plt.hist(data=df, x='actual_hum', bins='auto', label='Temperature')
plt.xlabel('Humidity')
plt.title("Humidity Distribution")
#Check for outliers in data using boxplot
sns.boxplot(data=df[['actual_temp','actual_feel_temp','actual_windspeed','actual_hum']])
fig=plt.gcf()
fig.set_size_inches(8,8)
#Remove outliers in Humidity
q75, q25 = np.percentile(df['actual_hum'], [75 ,25])
print(q75,q25)
iqr = q75 - q25
print(iqr)
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(min)
print(max)
​
df = df.drop(df[df.iloc[:,19] < min].index)
df = df.drop(df[df.iloc[:,19] > max].index)
#Remove outliers in Windspeed
q75, q25 = np.percentile(df['actual_windspeed'], [75 ,25])
print(q75,q25)
iqr = q75 - q25
print(iqr)
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(min)
print(max)
​
df = df.drop(df[df.iloc[:,18] < min].index)
df = df.drop(df[df.iloc[:,18] > max].index)
#Check for collinearity using corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
#Check the distribution of Temperature and Humdity against Bike rental count using scatter plot
fig, axs = plt.subplots(1,2, figsize=(15, 5), sharey=True)
axs[0].scatter(data=df, x='actual_temp', y='cnt')
axs[1].scatter(data=df, x='actual_hum', y='cnt', color = 'red')
fig.suptitle('Scatter plot for Temperature and Humidity')
plt.xlabel("Humidity")
plt.ylabel("Count of bikes")
#Check the distribution of Feel Temperature and Windspeed against Bike rental count using scatter plot
fig, axs = plt.subplots(1,2, figsize=(15, 5), sharey=True)
axs[0].scatter(data=df, x='actual_feel_temp', y='cnt')
axs[1].scatter(data=df, x='actual_windspeed', y='cnt', color = 'red')
fig.suptitle('Scatter plot for Feel Temperature and Windspeed')
plt.xlabel("Windspeed")
plt.ylabel("Count of bikes")
df = df.drop(columns=['holiday','instant','dteday','atemp','casual','registered','actual_temp','actual_feel_temp',
                      'actual_windspeed','actual_hum','actual_season','actual_yr','actual_holiday','actual_weathersit'])
DECISION TREE
#MAPE: 18.40%

#Accuracy: 81.60%

#Import Libraries for decision tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
#Divide data into train and test
train,test = train_test_split(df, test_size = 0.2, random_state = 123)
#Train the model
dt_model = DecisionTreeRegressor(random_state=123).fit(train.iloc[:,0:9], train.iloc[:,9])
#Predict the results of test data
dt_predictions = dt_model.predict(test.iloc[:,0:9])
df_dt = pd.DataFrame({'actual': test.iloc[:,9], 'pred': dt_predictions})
df_dt.head()
#Function for Mean Absolute Percentage Error
def MAPE(y_actual,y_pred):
    mape = np.mean(np.abs((y_actual - y_pred)/y_actual))
    return mape
#Calculate MAPE for decision tree
MAPE(test.iloc[:,9],dt_predictions)
#MAPE: 18.40%
#Accuracy: 81.60%
Random Forest
#MAPE: 13.10%

#Accuracy:86.90%

#Import library for RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
#Train the model
rf_model = RandomForestRegressor(n_estimators=500,random_state=123).fit(train.iloc[:,0:9], train.iloc[:,9])
#Predict the results of test data
rf_predictions = rf_model.predict(test.iloc[:,0:9])
#Create a dataframe for actual values and predicted values
df_rf = pd.DataFrame({'actual': test.iloc[:,9], 'pred': rf_predictions})
df_rf.head()
#Calculate MAPE
MAPE(test.iloc[:,9],rf_predictions)
#MAPE: 13.10%
#Accuracy:86.90%
Linear Regression
#MAPE:17.07%

#Accuracy: 82.93%

#Adjusted r2: 0.852

#F-stat: 122.7

#import libraries for Linear regression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
#Train the model
lr_model = sm.OLS(train.iloc[:,9].astype(float), train.iloc[:,0:9].astype(float)).fit()
#Check the summary of model
lr_model.summary()
#Predict the results of test data
lr_predictions = lr_model.predict(test.iloc[:,0:9])
##Create a dataframe for actual values and predicted values
df_lr = pd.DataFrame({'actual': test.iloc[:,9], 'pred': lr_predictions})
df_lr.head()
#Calclulate MAPE
MAPE(test.iloc[:,9],lr_predictions)
#MAPE:19.08%
#Accuracy: 81.92% 
#Adjusted r2: 0.967
#F-stat: 1852
#Create continuous data. Save target variable first
train_lr = train[['cnt','temp','hum','windspeed']]
test_lr = test[['cnt','temp','hum','windspeed']]
##Create dummies for categorical variables
cat_names = ["season", "yr", "mnth", "weekday", "workingday", "weathersit"]
​
for i in cat_names:
    temp1 = pd.get_dummies(train[i], prefix = i)
    temp2 = pd.get_dummies(test[i], prefix = i)
    train_lr = train_lr.join(temp1)
    test_lr = test_lr.join(temp2)
#Train the model
lr_model = sm.OLS(train_lr.iloc[:,0].astype(float), train_lr.iloc[:,1:34].astype(float)).fit()
#summary of model
lr_model.summary()
#Predict the results of test data
lr_predictions = lr_model.predict(test_lr.iloc[:,1:34])
##Create a dataframe for actual values and predicted values
df_lr = pd.DataFrame({'actual': test_lr.iloc[:,0], 'pred': lr_predictions})
df_lr.head()
#Calclulate MAPE
MAPE(test_lr.iloc[:,0],lr_predictions)
#MAPE:17.07%
#Accuracy: 82.93%
#Adjusted r2: 0.852
#F-stat: 122.7
​
