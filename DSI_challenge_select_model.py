#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: DSI Challenge - Breast Cancer Data Set - Select Model
Date: 13JUN2021
Author: John Koenig
Purpose: Select the optimal model and visualize a selected estimator from it

Inputs: pca_scaled_df.csv
        train_test_results.csv
        
Outputs: true_positive_rate.csv
         true_negative_rate.csv
         false_positive_rate.csv
         false_negative_rate.csv
         accuracy.csv
         precision.csv
         recall.csv
         F1_score.csv
         ROC_area.csv
         confusion_heatmaps.png
         potential_models{max_table_rows}.csv
         potential_models_table{max_table_rows}.png
         {optimal_model_name}.pkl
         dtreeviz_estimator{estimator_num}.dot
         
Notes: I had some problems saving dtreeviz programatically

'''


#%%
# import libraries

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

from dtreeviz.trees import dtreeviz, tree

project_dir = os.getcwd()
data_dir = project_dir + r'/data/'
models_dir = project_dir + r'/models/'
plots_dir = project_dir + r'/plots/'
validation_dir = project_dir + r'/validation/'

# set dpi
dpi = 300

# save files or not
save_files = True


#%%
# Load Random Forest Model Input

# load processed data
df_filename = 'pca_scaled_df.csv'
df = pd.read_csv(data_dir + df_filename)

# isolate principal components
pca_output = df.iloc[:,-3:]
pca_columns = list(pca_output.columns)

# separate data
target = df['target']
patient_IDs = pd.DataFrame(df['patient_ID'])

# calculate % of dataset that is positive class for target variable
percent_target = np.round(target.sum() / target.size, 2)


#%%
# Load Random Forest Train/Test Metrics DataFrame

# load results data
results_df_filename = 'train_test_results.csv'
results_df = pd.read_csv(data_dir + results_df_filename)
results_df_head = results_df.iloc[:100,:]
results_df_columns = list(results_df.columns)

# hyperparameter grid search options
samples_percent_options = [s/100 for s in range(5, 100, 5)] + [.99]
depth_options = list(range(1,21))

# prepare metrics
test_metrics_list = ['true_positive_rate',
                     'true_negative_rate',
                     'false_positive_rate',
                     'false_negative_rate',
                     'accuracy',
                     'precision',
                     'recall',
                     'F1_score',
                     'ROC_area']

samples_percent_columns = [str(d) for d in depth_options]
depth_index = [f'{str(int(s * 100))}%' for s in samples_percent_options]


#%%
# Reshape Metrics Data

# filter for only test records
results_df_test = results_df[results_df['train_test'] == 'Test']
results_df_test.sort_values(['sample_percent', 'depth'], inplace=True)
results_df_test.index = range(len(results_df_test))

# create metric dataframes
true_positive_rate_df = pd.DataFrame(columns=samples_percent_columns)
true_negative_rate_df = pd.DataFrame(columns=samples_percent_columns)
false_positive_rate_df = pd.DataFrame(columns=samples_percent_columns)
false_negative_rate_df = pd.DataFrame(columns=samples_percent_columns)
accuracy_df = pd.DataFrame(columns=samples_percent_columns)
precision_df = pd.DataFrame(columns=samples_percent_columns)
recall_df = pd.DataFrame(columns=samples_percent_columns)
F1_score_df = pd.DataFrame(columns=samples_percent_columns)
ROC_area_df = pd.DataFrame(columns=samples_percent_columns)

# iterate and build individual metrics dataframes
for samples in samples_percent_options:
    
    sample_df_tmp = results_df_test[results_df_test['sample_percent'] == samples]
    
    true_positive_rate_tmp = pd.DataFrame([sample_df_tmp['true_positive_rate'].values], columns=samples_percent_columns)
    true_positive_rate_df = pd.concat([true_positive_rate_df, true_positive_rate_tmp], axis=0)
    true_negative_rate_tmp = pd.DataFrame([sample_df_tmp['true_negative_rate'].values], columns=samples_percent_columns)
    true_negative_rate_df = pd.concat([true_negative_rate_df, true_negative_rate_tmp], axis=0)
    false_positive_rate_tmp = pd.DataFrame([sample_df_tmp['false_positive_rate'].values], columns=samples_percent_columns)
    false_positive_rate_df = pd.concat([false_positive_rate_df, false_positive_rate_tmp], axis=0)
    false_negative_rate_tmp = pd.DataFrame([sample_df_tmp['false_negative_rate'].values], columns=samples_percent_columns)
    false_negative_rate_df = pd.concat([false_negative_rate_df, false_negative_rate_tmp], axis=0)
    accuracy_tmp = pd.DataFrame([sample_df_tmp['accuracy'].values], columns=samples_percent_columns)
    accuracy_df = pd.concat([accuracy_df, accuracy_tmp], axis=0)
    precision_tmp = pd.DataFrame([sample_df_tmp['precision'].values], columns=samples_percent_columns)
    precision_df = pd.concat([precision_df, precision_tmp], axis=0)
    recall_tmp = pd.DataFrame([sample_df_tmp['recall'].values], columns=samples_percent_columns)
    recall_df = pd.concat([recall_df, recall_tmp], axis=0)
    F1_score_tmp = pd.DataFrame([sample_df_tmp['F1_score'].values], columns=samples_percent_columns)
    F1_score_df = pd.concat([F1_score_df, F1_score_tmp], axis=0)
    ROC_area_tmp = pd.DataFrame([sample_df_tmp['ROC_area'].values], columns=samples_percent_columns)
    ROC_area_df = pd.concat([ROC_area_df, ROC_area_tmp], axis=0)

# set indexes of metrics dataframes
true_positive_rate_df.index = depth_index
true_negative_rate_df.index = depth_index
false_positive_rate_df.index = depth_index
false_negative_rate_df.index = depth_index
accuracy_df.index = depth_index
precision_df.index = depth_index
recall_df.index = depth_index
F1_score_df.index = depth_index
ROC_area_df.index = depth_index

if save_files:
    
    # save metrics dataframes
    true_positive_rate_df.to_csv(data_dir + 'true_positive_rate.csv')
    true_negative_rate_df.to_csv(data_dir + 'true_negative_rate.csv')
    false_positive_rate_df.to_csv(data_dir + 'false_positive_rate.csv')
    false_negative_rate_df.to_csv(data_dir + 'false_negative_rate.csv')
    accuracy_df.to_csv(data_dir + 'accuracy.csv')
    precision_df.to_csv(data_dir + 'precision.csv')
    recall_df.to_csv(data_dir + 'recall.csv')
    F1_score_df.to_csv(data_dir + 'F1_score.csv')
    ROC_area_df.to_csv(data_dir + 'ROC_area.csv')


#%%
# Create Heatmap Multiplots

x_ticks = [0, 5, 10, 15, 20]
x_tick_labels = ['1', '5', '10', '15', '20']
y_ticks = [0, 5, 10, 15, 20]
y_tick_labels = ['0%', '25%', '50%', '75%', '100%']
x_axis_label = 'Max Tree Depth'
y_axis_label = 'Bootstrap Sample Percentage'

y_tick_labels.reverse()

fig, ax = plt.subplots(2, 4, figsize=(20, 12), sharex=True, sharey=True)
fig.suptitle('Test Performance Heatmaps', x=.525, fontsize=42)

sns.heatmap(np.flip(true_positive_rate_df.values, 0), cmap='Greens', ax=ax[0,0])
sns.heatmap(np.flip(true_negative_rate_df.values, 0), cmap='Reds_r', ax=ax[0,1])
sns.heatmap(np.flip(accuracy_df.values, 0), cmap='Oranges', ax=ax[0,2])
sns.heatmap(np.flip(precision_df.values, 0), cmap='Purples', ax=ax[0,3])
sns.heatmap(np.flip(false_positive_rate_df.values, 0), cmap='Reds_r', ax=ax[1,0])
sns.heatmap(np.flip(false_negative_rate_df.values, 0), cmap='Greens', ax=ax[1,1])
sns.heatmap(np.flip(recall_df.values, 0), cmap='Blues', ax=ax[1,2])
sns.heatmap(np.flip(F1_score_df.values, 0), cmap='Greys', ax=ax[1,3])

title_fontsize = 16
tick_fontsize = 14
cbar_kws = {'fontsize':tick_fontsize}

ax[0,0].set_title('true_positive_rate', fontsize=title_fontsize)
ax[0,1].set_title('true_negative_rate', fontsize=title_fontsize)
ax[0,2].set_title('accuracy', fontsize=title_fontsize)
ax[0,3].set_title('precision', fontsize=title_fontsize)
ax[1,0].set_title('false_positive_rate', fontsize=title_fontsize)
ax[1,1].set_title('false_negative_rate', fontsize=title_fontsize)
ax[1,2].set_title('recall', fontsize=title_fontsize)
ax[1,3].set_title('F1_score', fontsize=title_fontsize)
ax[1,0].set_xticks(x_ticks)
ax[1,0].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)
ax[1,1].set_xticks(x_ticks)
ax[1,1].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)
ax[1,2].set_xticks(x_ticks)
ax[1,2].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)
ax[1,3].set_xticks(x_ticks)
ax[1,3].set_xticklabels(x_tick_labels, fontsize=tick_fontsize)
ax[0,0].set_yticks(y_ticks)
ax[0,0].set_yticklabels(y_tick_labels, fontsize=tick_fontsize, rotation=90)
ax[1,0].set_yticks(y_ticks)
ax[1,0].set_yticklabels(y_tick_labels, fontsize=tick_fontsize, rotation=90)

fig.text(.44, .07, x_axis_label, fontsize=24)
fig.text(.09, .31, y_axis_label, fontsize=24, rotation=90)
plt.subplots_adjust(left=.13, top=.89)

if save_files:
    
    # save heatmap plot
    confusion_heatmaps_filename = 'confusion_heatmaps.png'
    fig.savefig(plots_dir + confusion_heatmaps_filename, dpi=dpi)


#%%
# Select Top 10 Potential Models and Build Table

results_df_test_sort_order = ['false_negative_rate',
                              'true_positive_rate',
                              'true_negative_rate',
                              'false_positive_rate']

results_df_test_ascending_order = [True, False, False, True] 

results_df_test.sort_values(by=results_df_test_sort_order,
                            axis=0,
                            ascending=results_df_test_sort_order,
                            inplace=True)

results_df_test.index = range(len(results_df_test))

table_columns = [#'model_name',
                 #'train_test',
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

# create potential models table
max_table_rows = 10

potential_models10 = results_df_test.iloc[:max_table_rows,:]
potential_models10.index = range(len(potential_models10))

table_fontsize = 9
column_widths = [.1, .15, .1, .2, .1, .15, .15, .15, .15, .2, .2, .2, .2, .1, .1, .1, .1, .1]
row_labels = potential_models10['model_name'].iloc[:max_table_rows]

cell_colors = [['white' for c in range(len(table_columns))] for r in range(max_table_rows)]
cell_colors[0] = ['mistyrose' for c in range(len(table_columns))]
potential_models10 = potential_models10[table_columns]
potential_models10_table_strings = [list(r) for _,r in potential_models10.astype(str).iterrows()]

# plot potential models table
fig, ax = plt.subplots(figsize=(10,4))
fig.suptitle('Top 10 Potential Random Forest Models', x=1, fontsize=28)
ax.set_axis_off()

model_table = ax.table(potential_models10_table_strings,
                       cellColours=cell_colors,
                       colWidths=column_widths,
                       rowColours=['lightgrey'] * len(potential_models10.columns),
                       rowLabels=row_labels,
                       colColours=['lightblue'] * len(potential_models10.columns),
                       colLabels=table_columns,
                       cellLoc='center',
                       loc='upper left')

model_table.scale(1,2)
model_table.auto_set_font_size(False)
model_table.set_fontsize(table_fontsize)

# select optimal model
optimal_model_name = results_df_test.iloc[0,0]
optimal_model_results = pd.DataFrame([results_df_test.iloc[0,:]])

if save_files:
    
    # save potential_models dataframe
    potential_models_filename = f'potential_models{max_table_rows}.csv'
    potential_models10.to_csv(data_dir + potential_models_filename)
    
    # save potential models table
    potential_models_table_filename = f'potential_models_table{max_table_rows}.png'
    fig.savefig(plots_dir + potential_models_table_filename)  # this isn't saving properly


#%%
# Select Optimal Model and Visualize Tree

# load optimal model
optimal_model_filename = f'{optimal_model_name}.pkl'
optimal_model = pickle.load(open(models_dir + optimal_model_filename, 'rb'))

x = pca_output
y = target

estimator_num = 7474

tree_visualization = dtreeviz(optimal_model.estimators_[estimator_num], 
                              x_data=x,
                              y_data=y,
                              target_name='Predict',
                              feature_names=pca_columns, 
                              class_names=['Benign', 'Malignant'], 
                              title=f'Selected Estimator from Optimal Random Forest [{estimator_num}:10,000]')

# show tree
tree_visualization.view()   # I have to save svg manually by right click save

if save_files:
    
    # save optimal model
    pickle.dump(optimal_model, open(validation_dir + optimal_model_filename, 'wb'))
    
    # save optimal model results to file
    optimal_model_name_filename = f'optimal_model_results.csv'
    optimal_model_results.to_csv(validation_dir + optimal_model_name_filename, index=False)

