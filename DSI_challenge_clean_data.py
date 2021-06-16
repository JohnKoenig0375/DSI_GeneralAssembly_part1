#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: DSI Challenge - Breast Cancer Data Set - Clean Data
Date: 13JUN2021
Author: John Koenig
Purpose: Download and clean data for data visualization and machine learning
Inputs: breast-cancer.csv (by url)
Outputs: breast_cancer.csv (clean)
Notes: 
    
'''

#%%
# Import Libraries

import os

import pandas as pd

project_dir = os.getcwd()
data_dir = project_dir + r'/data/'


#%%
# Download and Clean Project DataFrame

# download data
url = r'https://gist.githubusercontent.com/jeff-boykin/b5c536467c30d66ab97cd1f5c9a3497d/raw/5233c792af49c9b78f20c35d5cd729e1307a7df7/breast-cancer.csv'
df = pd.read_csv(url, header=None)

# create column labels
input_variables_labels = ['var' + str(i) for i in range(len(df.columns) - 2)]
column_labels = ['patient_ID', 'target'] + input_variables_labels
df.columns = column_labels

# recode target column as 1/0
df['target'] = df['target'].str.contains('M').astype(int)

# save cleaned data
df_filename = 'breast_cancer.csv'
df.to_csv(data_dir + df_filename, index=False)

