"""
Created on Tue Mar 22 2022
@author: adamfoster
"""
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

    smooth_pred_non_biased = merged_dataframe['actual_smooth_pred']
    featured_pred_non_biased = merged_dataframe['actual_featured_pred']

    smooth_pred_debiased = merged_dataframe['debiased_smooth_pred']
    featured_pred_debiased = merged_dataframe['debiased_featured_pred']

    smooth_sim_pred = merged_dataframe['sim_smooth_pred']
    featured_sim_pred = merged_dataframe['sim_featured_pred']

    #plot the non-sim spread
    plt.figure(figsize=(10,6))
    plt.suptitle('Morphology Distribution for N={0} Galaxies'.format(len(smooth_pred_non_biased)), fontsize=20, wrap=True)

    plt.scatter(smooth_pred_non_biased, featured_pred_non_biased, marker='x', s=2, alpha=0.5)
    plt.plot([0, 1], [1, 0], marker='', color='r')
    plt.xlabel('Smooth Likelihood', fontsize = 16)
    plt.ylabel('Featured Likelihood', fontsize = 16)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('other_plots/morphology_distribution_nonsim.jpeg')
    plt.close()

    #plot the high z sim spread
    plt.figure(figsize=(10,6))
    plt.suptitle('Morphology Distribution for N={0} High Redshift Simulated Galaxies'.format(len(smooth_sim_pred)), fontsize=20, wrap=True)

    plt.scatter(smooth_sim_pred, featured_sim_pred, marker='x', s=2, alpha=0.5)
    plt.plot([0, 1], [1, 0], marker='', color='r')
    plt.xlabel('Smooth Likelihood', fontsize = 16)
    plt.ylabel('Featured Likelihood', fontsize = 16)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('other_plots/morphology_distribution_sim.jpeg')
    plt.close()

    #plot the de-biased spread
    plt.figure(figsize=(10,6))
    plt.suptitle('Morphology Distribution for N={0} Debiased Galaxies'.format(len(smooth_pred_debiased)), fontsize=20, wrap=True)

    plt.scatter(smooth_pred_debiased, featured_pred_debiased, marker='x', s=2, alpha=0.5)
    plt.plot([0, 1], [1, 0], marker='', color='r')
    plt.xlabel('Smooth Likelihood', fontsize = 16)
    plt.ylabel('Featured Likelihood', fontsize = 16)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('other_plots/morphology_distribution_debiased.jpeg')
    plt.close()

    #plot superposition of each
    plt.figure(figsize=(10,6))
    plt.suptitle('Morphology Distribution for N={0} Galaxies'.format(len(smooth_pred_debiased)), fontsize=20, wrap=True)
    plt.scatter(smooth_pred_non_biased, featured_pred_non_biased, marker='v', s=2, alpha=0.25, color='r', label='Non-simulated predictions')
    plt.scatter(smooth_pred_debiased, featured_pred_debiased, marker='x', s=2, alpha=0.25, color='b', label='De-biased predictions')
    plt.plot([0, 1], [1, 0], marker='', color='black')
    plt.xlabel('Smooth Likelihood', fontsize = 16)
    plt.ylabel('Featured Likelihood', fontsize = 16)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15, labelcolor=['r', 'b'])
    plt.savefig('other_plots/morphology_distribution_debiased_superimposed.jpeg')
    plt.close() 

    logging.info('Finished')

 

