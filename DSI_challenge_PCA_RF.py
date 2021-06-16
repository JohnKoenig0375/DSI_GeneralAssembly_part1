#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: DSI Challenge - Breast Cancer Data Set - PCA + Random Forest
Date: 13JUN2021
Author: John Koenig
Purpose: Conduct data preparation and predicitive modeling on the given breast cancer
         data set using Principal Component Analysis and Random Forest Classifier

Inputs: breast_cancer.csv (clean)

Outputs: train_test_metrics.csv
         scaler_model.pkl
         pca_model_components{final_components}.pkl
         pca_scaled_df.csv
         ROC Curve -- Train -- {model_name}.png               (one per model)
         ROC Curve -- Test -- {model_name}.png                (one per model)
         RF_trees{n_trees}_sample{samples}_depth{depth}.pkl
         train_test_results
         
Notes: Results of the grid search for the optimal model hyperparameters are
       stored in train_test_metrics.csv
    
'''

#%%
# import libraries

import os
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

warnings.filterwarnings("ignore")  # supress DataConversionWarning

project_dir = os.getcwd()
data_dir = project_dir + r'/data/'
models_dir = project_dir + r'/models/'
plots_dir = project_dir + r'/plots/'
ROC_train_dir = plots_dir + r'/train/'
ROC_test_dir = plots_dir + r'/test/'

# set dpi
dpi = 300

# set decimal precision
decimals = 4

# save files or not
save_files = True

#%%
# Training Hyperparameters

n_trees = 10000
samples_percent_options = [s/100 for s in range(5, 100, 5)] + [.99]
depth_options = list(range(1,21))


#%%
# Load Cleaned Breast Cancer Dataset

# load cleaned dataframe
df_filename = 'breast_cancer.csv'
df = pd.read_csv(data_dir + df_filename)
df_columns = list(df.columns)

# extract input variable labels
input_variables_labels = df_columns[2:]
input_variables = df.iloc[:,2:]

# separate data
target = pd.DataFrame(df['target'])
patient_IDs = pd.DataFrame(df['patient_ID'])

# calculate % of dataset that is the positive class for the target variable
percent_target = np.round(target.sum() / target.size, 2)


#%%
# Split Dataset for Training 75-25, StandardScaler and Conduct PCA

test_split = .25
n_components = 20
final_components = 3

scaled_columns = ['s_var' + str(c) for c in range(len(input_variables_labels))]
pca_columns = [f'pc{p}' for p in range(final_components)]

# split dataset into train/test partitions
x = pd.concat([patient_IDs, input_variables], axis=1)
y = target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, stratify=y)

# save IDs
patient_IDs_train = x_train.iloc[:,0]
patient_IDs_test = x_test.iloc[:,0]

# drop ID column
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

# replace column labels
x_train.columns = [f's_{i}' for i in input_variables_labels]
x_test.columns = [f's_{i}' for i in input_variables_labels]
y_train = pd.DataFrame(y_train, index=x_train.index, columns=['target'])
y_test = pd.DataFrame(y_test, index=x_test.index, columns=['target'])

# scale input variables
scaler = MinMaxScaler()
scale_fit = scaler.fit(pd.concat([x_train,x_test], axis=0))
scaled_output_train = scale_fit.transform(x_train)
scaled_output_test = scale_fit.transform(x_test)
scaled_output_train = pd.DataFrame(scaled_output_train, index=x_train.index, columns=scaled_columns)
scaled_output_test = pd.DataFrame(scaled_output_test, index=x_test.index, columns=scaled_columns)

# conduct PCA
pca = PCA(n_components=n_components)

pca_model = pca.fit(x_train)
pca_variance = np.round(pca_model.explained_variance_ratio_, decimals=3) * 100
pca_variance_cum = np.cumsum(np.round(pca_model.explained_variance_ratio_, decimals=3) * 100)

# get eigenvalues
vars_covar = np.cov(x_train)
eigen_values, eigen_vectors = np.linalg.eig(vars_covar)
eigen_values = np.round(eigen_values.astype(float), 2)[:n_components]

# make predictions
pca_output_train = pca_model.transform(scaled_output_train)[:,:final_components]
pca_output_test = pca_model.transform(scaled_output_test)[:,:final_components]
pca_output_train = pd.DataFrame(pca_output_train, index=x_train.index, columns=pca_columns)
pca_output_test = pd.DataFrame(pca_output_test, index=x_test.index, columns=pca_columns)

if save_files:
    
    # save StandardScaler model
    scaler_filename = 'scaler_model.pkl'
    pickle.dump(scale_fit, open(models_dir + scaler_filename, 'wb'))
    
    # save PCA model
    pca_model_filename = f'pca_model_components{final_components}.pkl'
    pickle.dump(pca_model, open(models_dir + pca_model_filename, 'wb'))
    
    # concatenate model validation data and results
    train_list = ['Train' for t in range(len(patient_IDs_train))]
    train_series = pd.Series(train_list, index=x_train.index, name='train_test')
    
    train_df = pd.concat([patient_IDs_train,
                          train_series,
                          y_train,
                          x_train,
                          scaled_output_train,
                          pca_output_train], axis=1)
    
    
    test_list = ['Test' for t in range(len(patient_IDs_test))]
    test_series = pd.Series(test_list, index=x_test.index, name='train_test')
    
    test_df = pd.concat([patient_IDs_test,
                         test_series,
                         y_test,
                         x_test,
                         scaled_output_test,
                         pca_output_test], axis=1)
    
    output_df = pd.concat([train_df, test_df], axis=0).sort_index()
    
    # save scaled/pca data    
    output_df_filename = 'pca_scaled_df.csv'
    output_df.to_csv(data_dir + output_df_filename, index=False)


#%%
# Train Random Forest Model

# Create dataframe to store results
results_df_columns = ['model_name',
                      'train_test',
                      'test_split',
                      'total_records',
                      'n_trees',
                      'sample_percent',
                      'depth',
                      'true_positive',
                      'true_negative',
                      'false_positive',
                      'false_negative',
                      'true_positive_rate',
                      'true_negative_rate',
                      'false_positive_rate',
                      'false_negative_rate',
                      'accuracy',
                      'precision',
                      'recall',
                      'F1_score',
                      'ROC_area']

results_df = pd.DataFrame(columns=results_df_columns)

# conduct grid search
for samples in samples_percent_options:
    for depth in depth_options:
        
        model_name = f'RF_trees{n_trees}_sample{samples}_depth{depth}'
        
        # train random forest
        train_test = 'Train'
        
        print(f'Training Model - {model_name}')
        
        x = pca_output_train
        y = y_train
        
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    max_depth=depth,
                                    bootstrap=True,
                                    max_samples=samples)
        
        rf_model = rf.fit(x, y)
        rf_train_predict = rf.predict(x)
        
        # get confusion matrix and metrics for train partition
        confusion_matrix_train = metrics.confusion_matrix(rf_train_predict, y)
        
        total_records = len(x)
        
        true_positive = confusion_matrix_train[0,0]
        true_negative = confusion_matrix_train[1,1]
        false_positive = confusion_matrix_train[0,1]
        false_negative = confusion_matrix_train[1,0]
        
        true_positive_rate = np.round(true_positive / (true_positive + false_positive), decimals)
        true_negative_rate = np.round(true_negative / (true_negative + false_negative), decimals)
        false_positive_rate = np.round(false_positive / (true_positive + false_positive), decimals)
        false_negative_rate = np.round(false_negative / (true_negative + false_negative), decimals)
        
        accuracy = np.round((true_positive + false_negative) / total_records, decimals)
        precision = np.round(true_positive / (true_positive + false_positive), decimals)
        recall = np.round(true_positive / (true_positive + false_negative), decimals)
        F1_score = np.round(2 / ((1 / precision) + (1 / recall)), decimals)
        
        ROC_area = np.round(metrics.roc_auc_score(y_train, rf_train_predict), decimals)
        
        # plot training ROC curve
        fig_train, ax_train = plt.subplots(figsize=(8,6))
        ax_train.plot([0, 1], [0, 1], transform=ax_train.transAxes, color='grey', ls='--')
        metrics.plot_roc_curve(rf_model, x, y, ax=ax_train)
        
        ROC_train_string = f'ROC Curve -- {train_test} -- {model_name}'
        ax_train.set_title(ROC_train_string)
        
        # save model training metrics
        results_df_tmp = pd.DataFrame([[model_name,
                                        train_test,
                                        test_split,
                                        total_records,
                                        n_trees,
                                        samples,
                                        depth,
                                        true_positive,
                                        true_negative,
                                        false_positive,
                                        false_negative,
                                        true_positive_rate,
                                        true_negative_rate,
                                        false_positive_rate,
                                        false_negative_rate,
                                        accuracy,
                                        precision,
                                        recall,
                                        F1_score,
                                        ROC_area]], columns=results_df_columns)
        
        results_df = pd.concat([results_df, results_df_tmp], axis=0)
        results_df.index = range(len(results_df))
        
        
        
        # test random forest
        train_test = 'Test'
        
        print(f' Testing Model - {model_name}')
        
        # get predictions
        x = pca_output_test
        y = y_test
        
        rf_test_predict = rf_model.predict(x)
        
        # get confusion matrix and metrics for test partition
        confusion_matrix_train = metrics.confusion_matrix(rf_test_predict, y)
        
        total_records = len(x)
        
        true_positive = confusion_matrix_train[0,0]
        true_negative = confusion_matrix_train[1,1]
        false_positive = confusion_matrix_train[0,1]
        false_negative = confusion_matrix_train[1,0]
        
        true_positive_rate = np.round(true_positive / (true_positive + false_positive), decimals)
        true_negative_rate = np.round(true_negative / (true_negative + false_negative), decimals)
        false_positive_rate = np.round(false_positive / (true_positive + false_positive), decimals)
        false_negative_rate = np.round(false_negative / (true_negative + false_negative), decimals)
        
        accuracy = np.round((true_positive + false_negative) / total_records, decimals)
        precision = np.round(true_positive / (true_positive + false_positive), decimals)
        recall = np.round(true_positive / (true_positive + false_negative), decimals)
        F1_score = np.round(2 / ((1 / precision) + (1 / recall)), decimals)
        
        ROC_area = np.round(metrics.roc_auc_score(y_test, rf_test_predict), decimals)
        
        # plot testing ROC curve
        fig_test, ax_test = plt.subplots(figsize=(8,6))
        ax_test.plot([0, 1], [0, 1], transform=ax_test.transAxes, color='grey', ls='--')
        metrics.plot_roc_curve(rf_model, x, y, ax=ax_test)
        
        ROC_test_string = f'ROC Curve -- {train_test} -- {model_name}'
        ax_test.set_title(ROC_test_string)
        
        # save model test results
        results_series = pd.DataFrame([[model_name,
                                        train_test,
                                        test_split,
                                        total_records,
                                        n_trees,
                                        samples,
                                        depth,
                                        true_positive,
                                        true_negative,
                                        false_positive,
                                        false_negative,
                                        true_positive_rate,
                                        true_negative_rate,
                                        false_positive_rate,
                                        false_negative_rate,
                                        accuracy,
                                        precision,
                                        recall,
                                        F1_score,
                                        ROC_area]], columns=results_df_columns)
        
        results_df = pd.concat([results_df, results_series], axis=0)
        
        if save_files:
            
            # save ROC curve plot
            ROC_train_plot_filename = f'{ROC_train_string}.png'
            fig_train.savefig(ROC_train_dir + ROC_train_plot_filename, dpi=dpi)
            
            # save ROC curve plot
            ROC_test_plot_filename = f'{ROC_test_string}.png'
            fig_test.savefig(ROC_test_dir + ROC_test_plot_filename, dpi=dpi)
            
            # save random forest model
            rf_model_filename = f'{model_name}.pkl'
            pickle.dump(rf_model, open(models_dir + rf_model_filename, 'wb'))
            

results_df.index = range(len(results_df))

if save_files:
    
    # save results
    results_df_filename = 'train_test_results.csv'
    results_df.to_csv(data_dir + results_df_filename, index=False)

