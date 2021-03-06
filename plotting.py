# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:56:09 2022

@author: r41331jc
"""
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    logging.info('Beginning plotting')

    count_array_perm = pd.read_csv('output_csvs/count_array_perm.csv', index_col=0)
    count_array_perm = count_array_perm.to_numpy()
    
    error_array_perm = pd.read_csv('output_csvs/error_array_perm.csv', index_col=0)
    error_array_perm = error_array_perm.to_numpy()
    
    residual_array = pd.read_csv('output_csvs/residual_array.csv', index_col=0)
    residual_array = residual_array.to_numpy()
    
    standard_deviation_array = pd.read_csv('output_csvs/standard_deviation_array.csv', index_col=0)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    count = int(len(count_array_perm[:,0])/9)
    
    count_array_perm = np.split(count_array_perm, count)
    count_array_perm = np.asarray(count_array_perm)
    
    error_array_perm = np.split(error_array_perm, count)
    error_array_perm = np.asarray(error_array_perm)
    
    residual_array = np.split(residual_array, count)
    residual_array = np.asarray(residual_array)
    
    for i in range(len(count_array_perm[:,0,0])):
        plt.plot(count_array_perm[i,:,0],count_array_perm[i,:,1], linestyle = "-", alpha = 0.7) #label = "{0:.3f} {1:.3f} {2:.3f}".format(delta_z, delta_p, delta_mag),
    
    plt.plot(error_array_perm[0,:,0],error_array_perm[0,:,1], linestyle = "--", color = "purple") #label= 'Expected number',
    plt.xlabel("Percentage", fontsize = 15)
    plt.ylabel("Number of galaxies", fontsize = 15)
    plt.xlim([0, 100])
    plt.ylim([0, error_array_perm[0,0,1]*10])
    #plt.legend()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("Number of galaxies with percentage", fontsize = 16, wrap = True)
    plt.savefig("other_plots/Total linear.png")
    plt.close()
    
    for i in range(len(residual_array[:,0,0])):
        plt.plot(residual_array[i,:,0],residual_array[i,:,1],linestyle = "-", alpha = 0.7)
    
    plt.xlabel("Percentage", fontsize = 15)
    plt.ylabel("Differnece", fontsize = 15, wrap = True)
    plt.xlim([0, 100])
    plt.ylim([np.min(residual_array[:,:,1])-0.5, np.max(residual_array[:,:,1])+0.5])
    plt.axhline(0, color='black')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.legend()
    plt.title("Difference in galaxies from percentage", fontsize = 16, wrap = True)
    plt.savefig("other_plots/Total linear residual.png")
    plt.close()
    
    temp = pd.DataFrame(standard_deviation_array)
    temp = temp.loc[[temp[4].astype(float).idxmin()]]
    temp = temp.reset_index(drop=True)
    delta_z = temp[0][0]
    delta_p = temp[1][0]
    delta_mag = temp[2][0]
    delta_mass = temp[3][0]
    delta_conc = temp[4][0]
    
    temp.to_csv('output_csvs/deltas.csv')

    logging.info('Plotting complete')

    logging.info(f'The best delta values are delta z = {delta_z}, delta p =  {delta_p}, delta mag = {delta_mag}, delta mass = {delta_mass} and delta conc = {delta_conc}')