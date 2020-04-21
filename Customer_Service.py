# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:27:18 2020

@author: TUSHAR
"""

'''----------Importing Libraries-----------'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

'''reading the dataset'''

df = pd.read_csv("NYC_311_Service_Req.csv", parse_dates = ['Created Date', 'Closed Date'])

df_model = df.copy()

df.info()

'''-----finding missing values-----'''

def missing_values_table(df_model):
        mis_val = df_model.isnull().sum()
        mis_val_percent = 100 * df_model.isnull().sum() / len(df_model)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df_model.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

missing_values_table(df_model)

df_model.info()

bool_data = pd.isna(df_model['Borough']).sum()
bool_data

'''dropping all the columns where null value is 84.5% to 100%'''

df_model.drop(['School or Citywide Complaint','Vehicle Type','Taxi Company Borough',
                         'Taxi Pick Up Location','Garage Lot Name','Ferry Direction',
                         'Ferry Terminal Name','Bridge Highway Segment','Road Ramp','Bridge Highway Direction',
                         'Bridge Highway Name','Landmark','Intersection Street 2','Intersection Street 1']
                         ,axis=1, inplace=True)
df_model.info()

'''-------Treating the missing values of Closed Date------'''
df_model.rename(columns = {'Created Date': 'Created_Date','Closed Date':'Closed_Date'}, inplace=True)
type(df_model.Closed_Date[0])
df_model['Closed_Date'].fillna(method="ffill", inplace=True)
df_model['Location Type'].fillna(method="ffill", inplace=True)
df_model['Closed_Date'].isna().sum()
df_model['Location Type'].isna().sum()

'''Inserting a new column and assigning the values'''
df_model.insert(3,"Request_Closing_Time", None)
df_model.insert(4,"Request_Closing_hours",None)
df_model['Request_Closing_Time']= df_model['Closed_Date'] - df_model['Created_Date']

'''------------Finding the top complaints-------'''
complaint_count=pd.crosstab(df_model['Complaint Type'],df_model['Complaint Type']).max().sort_values(ascending=False)

'''--------Plotting the top 6 complaints in a pie chart------'''
complaint_count[0:6]
labels = 'Blocked Driveway','Illegal Parking','Noise - Street/Sidewalk','Noise - Commercial','Derelict Vehicle','Noise - Vehicle'
size = [77044,75361,48612,35577,17718,17083]
explode=(0.1,0,0,0,0,0) #only "explode" the first slice i.e 'Blocked Driveway'
fig1, ax1 = plt.subplots()
ax1.pie(size, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

'''------------------Data Manipulation-------------'''
req_data = df_model[['Created_Date', 'Closed_Date','Request_Closing_Time','Request_Closing_hours','Complaint Type','Location Type','Location']]
req_data.rename(columns = {'Complaint Type':'Complaint_Type','Location Type':'Location_Type'}, inplace=True)

##Converting the request closing time to integer values
req_data['Request_Closing_hours'] = req_data['Request_Closing_Time'].astype('timedelta64[h]')+1

req_data[['Request_Closing_Time','Request_Closing_hours']].head()

avg_req_closing_time = req_data['Request_Closing_hours'].mean()
std_dev_req_closing_time = req_data['Request_Closing_hours'].std()
avg_req_closing_time
std_dev_req_closing_time

'''---------Plotting the data as per the location type'''
plot_data = req_data[((req_data['Request_Closing_hours']-avg_req_closing_time)/std_dev_req_closing_time) < 1]
plot_data['Request_Closing_hours'].hist(bins=10)
plt.xlabel('Time taken to close')
plt.ylabel('No of Complaints')
plt.title('Complaints Closed within Avg Response Time')
plt.show()

'''-------Hypothesis Testing---------'''
from scipy.stats import f_oneway
'''Normality Test:- Performed to check the normal distribution of a sample--- Shapiro Wilk's Test'''
print(stats.shapiro(req_data.Request_Closing_hours)) #P-value - 0.510
print(stats.shapiro(complaint_count)) ## p-value - 0.602

'''One way ANOVA is performed to The one-way analysis of variance (ANOVA) is used to determine whether 
there are any statistically significant differences between the means of two or more independent (unrelated) groups 
(although you tend to only see it used when there are a minimum of three, rather than two groups).'''

stat, p = f_oneway(req_data.Request_Closing_hours, req_data.Complaint_Type.value_counts())

print(stat,p)


'''-------Chi Square Test for checking if the complaint types and location are related----'''

framed_data = pd.crosstab(req_data['Complaint_Type'],req_data['Location'])

from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency(framed_data)

print(stat,p)
