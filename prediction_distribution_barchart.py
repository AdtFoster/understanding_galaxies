"""
Created on Thurs Mar 31 2022
@author: adamfoster
"""

import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf
import os
import glob


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-data-dir', dest='prediction_data_dir', type=str)

    args = parser.parse_args()
    
    prediction_data_dir = args.prediction_data_dir

    logging.info('Beginning Morphology Plots')

    # The data
    debiased_df_chunks = [pd.read_csv(loc) for loc in glob.glob(os.path.join(prediction_data_dir, '*.csv'))]
    merged_dataframe = pd.concat(debiased_df_chunks, axis=0).reset_index(drop=True)

    #plot the DE-BIASED plots
    plt.subplots(2, 2, figsize=(12, 8))
    plt.suptitle('Prediction Distribution for N={0} De-biased Galaxies'.format(len(merged_dataframe)), fontsize=20, wrap=True)

    plt.subplot(221)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="debiased_smooth_pred", binwidth=0.1, binrange=[0, 1], color='r', alpha=0.5, label='Smooth')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='r')
    plt.xlabel('Zoobot Smooth Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(222)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="debiased_featured_pred", binwidth=0.1, binrange=[0, 1], color='g', alpha=0.5, label='Featured')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='g')
    plt.xlabel('Zoobot Featured Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(223)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="debiased_artifact_pred", binwidth=0.1, binrange=[0, 1], color='b', alpha=0.5, label='Artifact')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='b')
    plt.xlabel('Zoobot Artifact Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(224)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="debiased_smooth_pred", binwidth=0.1, binrange=[0, 1], color='r', alpha=0.2, label='Smooth')
    sns.histplot(data=merged_dataframe, x="debiased_featured_pred", binwidth=0.1, binrange=[0, 1], color='g', alpha=0.2, label='Featured')
    sns.histplot(data=merged_dataframe, x="debiased_artifact_pred", binwidth=0.1, binrange=[0, 1], color='b', alpha=0.2, label='Artifact')
    plt.xlabel('Zoobot Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor=['r', 'g', 'b'])

    plt.savefig('other_plots/prediction_distribution_debiased_bar.jpeg')
    plt.close()

    #plot the NON-SIMULATED plots
    plt.subplots(2, 2, figsize=(12, 8))
    plt.suptitle('Prediction Distribution for N={0} Non-Simulated Galaxies'.format(len(merged_dataframe)), fontsize=20, wrap=True)


    plt.subplot(221)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="actual_smooth_pred", binwidth=0.1, binrange=[0, 1], color='r', alpha=0.5, label='Smooth')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='r')
    plt.xlabel('Zoobot Smooth Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(222)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="actual_featured_pred", binwidth=0.1, binrange=[0, 1], color='g', alpha=0.5, label='Featured')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='g')
    plt.xlabel('Zoobot Featured Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(223)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="actual_artifact_pred", binwidth=0.1, binrange=[0, 1], color='b', alpha=0.5, label='Artifact')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='b')
    plt.xlabel('Zoobot Artifact Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(224)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="actual_smooth_pred", binwidth=0.1, binrange=[0, 1], color='r', alpha=0.2, label='Smooth')
    sns.histplot(data=merged_dataframe, x="actual_featured_pred", binwidth=0.1, binrange=[0, 1], color='g', alpha=0.2, label='Featured')
    sns.histplot(data=merged_dataframe, x="actual_artifact_pred", binwidth=0.1, binrange=[0, 1], color='b', alpha=0.2, label='Artifact')
    plt.xlabel('Zoobot Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor=['r', 'g', 'b'])

    plt.savefig('other_plots/prediction_distribution_nonsim_bar.jpeg')
    plt.close()

    #plot the SIMULATED plots
    plt.subplots(2, 2, figsize=(12, 8))
    plt.suptitle('Prediction Distribution for N={0} High-z Simulated Galaxies'.format(len(merged_dataframe)), fontsize=20, wrap=True)


    plt.subplot(221)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="sim_smooth_pred", binwidth=0.1, binrange=[0, 1], color='r', alpha=0.5, label='Smooth')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='r')
    plt.xlabel('Zoobot Smooth Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(222)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="sim_featured_pred", binwidth=0.1, binrange=[0, 1], color='g', alpha=0.5, label='Featured')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='g')
    plt.xlabel('Zoobot Featured Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(223)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="sim_artifact_pred", binwidth=0.1, binrange=[0, 1], color='b', alpha=0.5, label='Artifact')
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor='b')
    plt.xlabel('Zoobot Artifact Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)

    plt.subplot(224)
    sns.set_style('whitegrid')  
    sns.histplot(data=merged_dataframe, x="sim_smooth_pred", binwidth=0.1, binrange=[0, 1], color='r', alpha=0.2, label='Smooth')
    sns.histplot(data=merged_dataframe, x="sim_featured_pred", binwidth=0.1, binrange=[0, 1], color='g', alpha=0.2, label='Featured')
    sns.histplot(data=merged_dataframe, x="sim_artifact_pred", binwidth=0.1, binrange=[0, 1], color='b', alpha=0.2, label='Artifact')
    plt.xlabel('Zoobot Predictions', fontsize = 16)
    plt.ylabel('Count', fontsize = 16)
    plt.xlim((0, 1))
    plt.legend(fontsize=12, labelcolor=['r', 'g', 'b'])

    plt.savefig('other_plots/prediction_distribution_simhighz_bar.jpeg')
    plt.close()

   
