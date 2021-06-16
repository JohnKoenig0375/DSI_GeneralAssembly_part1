#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Title: DSI Challenge - Breast Cancer Data Set - EDA
Date: 13JUN2021
Author: John Koenig
Purpose: Conduct Exploratory Data Analysis (EDA) and data visualization
         of the given breast cancer data set

Inputs: breast_cancer.csv (clean)
Outputs: corr_heatmap.png
         histogram_6plots_part{p}.png  (6 files)
         boxplots30.png
         boxplots_no_outliers.png
         
Notes: 
    
'''

#%%
# import libraries

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns;

project_dir = os.getcwd()
data_dir = project_dir + r'/data/'
plots_dir = project_dir + r'/plots/'

# set dpi
dpi = 300

# save files or not
save_files = True

#%%
# Load Cleaned Breast Cancer Dataset

# load cleaned dataframe
df_filename = 'breast_cancer.csv'
df = pd.read_csv(data_dir + df_filename)
df_columns = list(df.columns)

# extract input variable labels
input_variables_labels = df_columns[2:]
input_variables = df.iloc[:,2:]


#%%
# Correlation Heatmap

# create mask for upper right of triangle
mask = np.triu(np.ones_like(input_variables.corr(), dtype=bool))

# create correlation heatmap for input variables
corr_min = -1
corr_max = 1

fig, ax = plt.subplots(figsize=(9.5, 8))

sns.heatmap(input_variables.corr(),
            vmin=corr_min,
            vmax=corr_max,
            cmap='coolwarm_r',
            cbar_kws={"ticks": [-1, 0, 1]},
            mask=mask,
            ax=ax)

fig.suptitle('Correlation Between Breast Cancer Input Variables', x=.46, fontsize=18)

plt.subplots_adjust(top=.92)

if save_files:
    
    # save heatmap plot
    corr_filename = 'corr_heatmap.png'
    fig.savefig(plots_dir + corr_filename, dpi=dpi)


#%%
# Small Multiples of Histograms

# create a histogram for each variable
axes_per_fig = 6

bins = 100
x_limit = (0, 5000)
y_limit = (0, 50)
hist_color = 'mediumblue'

# iterate over input variables and create custom histograms for each
for p in range(int(len(input_variables_labels) / axes_per_fig)):
    
    start_index = p * 6
    end_index = p * 6 + 6
    var_labels_tmp = input_variables_labels[start_index:end_index]
    
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    
    ax_list = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2]]
    
    for i in range(len(ax_list)):    
        ax_list[i].hist(input_variables[var_labels_tmp[i]], bins=bins, color=hist_color)
        ax_list[i].set_title(var_labels_tmp[i])
        ax_list[i].set_ylim(y_limit)
    
    suptitle = f'Histograms of Breast Cancer Input Variables {var_labels_tmp[0]} - {var_labels_tmp[5]}'
    suptitle += '\nNote: X-axis is not constant between plots'
    fig.suptitle(suptitle, x=.55, fontsize=18)
    fig.subplots_adjust(top=0.87)
    
    # add axis labels
    x_axis_label = f'Values [bins={bins}]'
    y_axis_label = f'Value Counts'
    fig.text(.44, .05, x_axis_label, fontsize=14)
    fig.text(.08, .45, y_axis_label, fontsize=14, rotation=90)
    
    # add vertical lines for mean/median
    lines_labels = ['Mean', 'Median']
    colors = ['limegreen', 'tomato']
    line_styles = ['--', '--']
    alpha = .8
    
    for i in range(len(ax_list)):
        mean_tmp = input_variables[var_labels_tmp[i]].mean()
        med_tmp = input_variables[var_labels_tmp[i]].median()
        
        ax_list[i].axvline(x=mean_tmp, color=colors[0], linestyle=line_styles[0], alpha=alpha)
        ax_list[i].axvline(x=med_tmp, color=colors[1], linestyle=line_styles[1], alpha=alpha)
        
    # create custom legend
    handles = [mlines.Line2D([], [], color=colors[0], linestyle=line_styles[0], alpha=alpha),
               mlines.Line2D([], [], color=colors[1], linestyle=line_styles[1], alpha=alpha)]
    
    fig.legend(handles, lines_labels, loc='right', fontsize=12)
    fig.subplots_adjust(right=.88)
    
    if save_files:
        
        # save histogram multi-plot
        histogram_6plots_filename = f'histogram_6plots_part{p}.png'
        fig.savefig(plots_dir + histogram_6plots_filename, dpi=dpi)


#%%
# 30 Variable Box Plot

# reshape input data
input_variables_labels_reversed = input_variables_labels[::-1]
box_input = input_variables[input_variables_labels_reversed].T

# create box plot
fig, ax = plt.subplots(figsize=(6, 10))

ax.boxplot(box_input, vert=False)

ax.set_yticklabels(input_variables_labels_reversed)
ax.set_title('Breast Cancer Input Variables - Box Plot', x=.45)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Variable Name', fontsize=12)

if save_files:
    
    # save box plot
    boxplots30_filename = f'boxplots30.png'
    fig.savefig(plots_dir + boxplots30_filename, dpi=dpi)


#%%
# 30 Variable Box Plot - Outliers Removed

# remove outliers
outlier_cutoff = 1
no_outliers_list = []

for v in input_variables_labels:
    if input_variables[v].max() <= outlier_cutoff:
        no_outliers_list.append(v)

input_variables_no_outliers = input_variables[no_outliers_list]

# reshape input data
input_variables_no_outliers_labels_reversed = input_variables_no_outliers.columns.to_list()[::-1]
box_input = input_variables[input_variables_no_outliers_labels_reversed].T

# create box plot
fig, ax = plt.subplots(figsize=(6, 10))

ax.boxplot(box_input, vert=False)

ax.set_yticklabels(input_variables_no_outliers_labels_reversed)
ax.set_title('Breast Cancer Input Variables - Box Plot\nNo Outliers', x=.44)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Variable Name', fontsize=12)

if save_files:
    
    # save box plot - no outliers
    boxplots_no_outliers_filename = f'boxplots_no_outliers.png'
    fig.savefig(plots_dir + boxplots_no_outliers_filename, dpi=dpi)

