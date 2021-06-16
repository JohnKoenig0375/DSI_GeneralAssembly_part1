#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: DSI Challenge - Breast Cancer Data Set - Validate Model
Date: 13JUN2021
Author: John Koenig
Purpose: Validate optimal model parameters using k-folds

Inputs: pca_scaled_df.csv
        
Outputs: pca_scaled_predict_df.csv
         rf_validation_output_kfold{kf_counter}.csv
         {optimal_model_name}_kfold{kf_counter}.pkl
         ROC Curve -- Train -- KFolds{n_splits}.png
         ROC Curve -- Test -- KFolds{n_splits}.png
         train_test_metrics_KFolds{n_splits}.csv
         
Notes: 

'''


#%%
# import libraries

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold

project_dir = os.getcwd()
data_dir = project_dir + r'/data/'
validation_dir = project_dir + r'/validation/'

optimal_model_name = 'RF_trees10000_sample0.25_depth8'
n_trees = 10000
test_split = 0.25
samples = 0.25
max_depth = 8
decimals = 4

# set dpi
dpi = 300

# save files or not
save_files = True


#%%
# Load Random Forest Model Input

# load processed data
df_filename = 'pca_scaled_df.csv'
df = pd.read_csv(data_dir + df_filename)
df_columns = list(df.columns)
df_head = df.iloc[:100,:]

# isolate principal components
pca_output = df.iloc[:,-3:]
pca_output_labels = list(pca_output.columns)

# separate data
target = df['target']
patient_IDs = pd.DataFrame(df['patient_ID'])
combined_input = df.iloc[:,2:]

# calculate % of dataset that is positive class for target variable
percent_target = np.round(target.sum() / target.size, 2)


#%%
# Conduct K-Folds Validation of Optimal Model Hyperparameters

n_splits = 4

results_df_columns = ['KFold',
                      'n_splits',
                      'model_name',
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

# Create dataframe to store train/test results
results_df = pd.DataFrame(columns=results_df_columns)

# create ROC curve training fix, ax
fig_train, ax_train = plt.subplots(figsize=(8,6))
ax_train.plot([0, 1], [0, 1], transform=ax_train.transAxes, color='grey', ls='--')
ax_train.set_title(f'ROC Curve -- Train -- KFolds{n_splits}')

# create ROC curve testing fix, ax
fig_test, ax_test = plt.subplots(figsize=(8,6))
ax_test.plot([0, 1], [0, 1], transform=ax_test.transAxes, color='grey', ls='--')
ax_test.set_title(f'ROC Curve -- Test -- KFolds{n_splits}')

# assign x, y
x = pca_output
y = target

# split by K-Folds
kf = KFold(n_splits=n_splits, random_state=375, shuffle=True)

kf_counter = 0

# conduct K-Folds validation
for train_index, test_index in kf.split(pca_output):
    
    # split dataset
    x_train = x.filter(train_index, axis=0)
    x_test = x.filter(test_index, axis=0)
    y_train = y.filter(train_index, axis=0)
    y_test = y.filter(test_index, axis=0)
    patient_IDs_train = patient_IDs.filter(train_index, axis=0)
    patient_IDs_test = patient_IDs.filter(test_index, axis=0)
    
    # train model
    train_test = 'Train'
    
    rf = RandomForestClassifier(n_estimators=n_trees,
                                max_depth=max_depth,
                                bootstrap=True,
                                max_samples=samples)
    
    rf_model = rf.fit(x_train, y_train)
    rf_train_predict = pd.DataFrame(rf.predict(x_train), index=x_train.index, columns=['predict'])
    
    # get train partition results
    confusion_matrix_train = metrics.confusion_matrix(rf_train_predict, y_train)
    
    total_records = len(x_train)
    
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
    metrics.plot_roc_curve(rf_model,
                           x_train,
                           y_train,
                           ax=ax_train)
    
    # save model training results
    results_df_tmp = pd.DataFrame([[kf_counter,
                                    n_splits,
                                    optimal_model_name,
                                    train_test,
                                    test_split,
                                    total_records,
                                    n_trees,
                                    samples,
                                    max_depth,
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
    
    # test model
    train_test = 'Test'
    
    rf_test_predict = pd.DataFrame(rf.predict(x_test), index=x_test.index, columns=['predict'])
    
    # get test partition results
    confusion_matrix_test = metrics.confusion_matrix(rf_test_predict, y_test)
    
    total_records = len(x_test)
    
    true_positive = confusion_matrix_test[0,0]
    true_negative = confusion_matrix_test[1,1]
    false_positive = confusion_matrix_test[0,1]
    false_negative = confusion_matrix_test[1,0]
    
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
    metrics.plot_roc_curve(rf_model,
                           x_test,
                           y_test,
                           ax=ax_test)
    
    # save model testing results
    results_df_tmp = pd.DataFrame([[kf_counter,
                                    n_splits,
                                    optimal_model_name,
                                    train_test,
                                    test_split,
                                    total_records,
                                    n_trees,
                                    samples,
                                    max_depth,
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
    
    # concatenate model validation data and results
    train_list = ['Train' for t in range(len(patient_IDs_train))]
    train_series = pd.Series(train_list, index=x_train.index, name='train_test')
    combined_input_train = combined_input.filter(train_index, axis=0)
    combined_input_test = combined_input.filter(test_index, axis=0)
    
    train_df = pd.concat([patient_IDs_train,
                          train_series,
                          rf_train_predict,
                          y_train,
                          combined_input_train], axis=1)
    
    
    test_list = ['Test' for t in range(len(patient_IDs_test))]
    test_series = pd.Series(test_list, index=x_test.index, name='train_test')
    
    test_df = pd.concat([patient_IDs_test,
                         test_series,
                         rf_test_predict,
                         y_test,
                         combined_input_test], axis=1)
    
    output_df = pd.concat([train_df, test_df], axis=0).sort_index()
    
    # save scaled/pca data    
    output_df_filename = 'pca_scaled_predict_df.csv'
    output_df.to_csv(data_dir + output_df_filename, index=False)
    
    kf_counter += 1
    
    if save_files:
        
        # save k-fold dataframe with prediction
        output_df_filename = f'rf_validation_output_kfold{kf_counter}.csv'
        output_df.to_csv(validation_dir + output_df_filename, index=False)
        
        # save k-fold random forest model
        rf_model_filename = f'{optimal_model_name}_kfold{kf_counter}.pkl'
        pickle.dump(rf_model, open(validation_dir + rf_model_filename, 'wb'))


results_df.index = range(len(results_df))

if save_files:
    
    # save ROC curve plots train/test
    ROC_train_plot_filename = f'ROC Curve -- Train -- KFolds{n_splits}.png'
    fig_train.savefig(validation_dir + ROC_train_plot_filename, dpi=dpi)

    # save ROC curve plot
    ROC_test_plot_filename = f'ROC Curve -- Test -- KFolds{n_splits}.png'
    fig_test.savefig(validation_dir + ROC_test_plot_filename, dpi=dpi)
    
    # save metrics
    results_df_filename = f'train_test_metrics_KFolds{n_splits}.csv'
    results_df.to_csv(validation_dir + results_df_filename, index=False)
