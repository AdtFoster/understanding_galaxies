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


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logging.info('Beginning')
    # The data
    full_data = pd.read_csv('output_csvs/full_data.csv') #index_col=0

    non_simulated_gals=full_data.query(f'scale_factor == 1')

    smooth_pred = non_simulated_gals['smooth-or-featured-dr5_smooth_prob']
    featured_pred = non_simulated_gals['smooth-or-featured-dr5_featured-or-disk_prob']

    plt.figure(figsize=(10,6))
    plt.suptitle('Morphology Distribution for N={0} Galaxies'.format(len(smooth_pred)), fontsize=20, wrap=True)

    plt.scatter(smooth_pred, featured_pred, marker='x', s=2, alpha=0.5)
    plt.plot([0, 1], [1, 0], marker='', color='r')
    plt.xlabel('Smooth Likelihood')
    plt.ylabel('Featured Likelihood')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig('other_plots/morphology_distribution.jpeg')
    plt.close()

    logging.info('Finished')

 

