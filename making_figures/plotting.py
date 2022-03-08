# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:56:09 2022

@author: r41331jc
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    count_array_perm = pd.read_csv('count_array_perm.csv', index_col=0)
    count_array_perm = count_array_perm.to_numpy()
    
    error_array_perm = pd.read_csv('error_array_perm.csv', index_col=0)
    error_array_perm = error_array_perm.to_numpy()
    
    residual_array = pd.read_csv('residual_array.csv', index_col=0)
    residual_array = residual_array.to_numpy()
    
    standard_deviation_array = pd.read_csv('standard_deviation_array.csv', index_col=0)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    count = int(len(count_array_perm[:,0])/17)
    
    count_array_perm = np.split(count_array_perm, count)
    count_array_perm = np.asarray(count_array_perm)
    
    error_array_perm = np.split(error_array_perm, count)
    error_array_perm = np.asarray(error_array_perm)
    
    residual_array = np.split(residual_array, count)
    residual_array = np.asarray(residual_array)
    
    standard_deviation_array = np.split(standard_deviation_array, count)
    standard_deviation_array = np.asarray(standard_deviation_array)
    
    for i in range(len(count_array_perm[:,0,0])):
        plt.plot(count_array_perm[i,:,0],count_array_perm[i,:,1], linestyle = "-", alpha = 0.7) #label = "{0:.3f} {1:.3f} {2:.3f}".format(delta_z, delta_p, delta_mag),
    
    plt.plot(error_array_perm[0,:,0],error_array_perm[0,:,1], linestyle = "--", color = "purple") #label= 'Expected number',
    plt.xlabel("Percentage", fontsize = 14)
    plt.ylabel("Number of galaxies", fontsize = 14)
    plt.xlim([0, 100])
    plt.ylim([0, error_array_perm[0,0,1]*10])
    #plt.legend()
    plt.title("Number of galaxies with percentage", fontsize = 14, wrap = True)
    plt.savefig("Total linear")
    plt.close()
    
    for i in range(len(residual_array[:,0,0])):
        plt.plot(residual_array[i,:,0],residual_array[i,:,1],linestyle = "-", alpha = 0.7)
    
    plt.xlabel("Percentage", fontsize = 14)
    plt.ylabel("Differnece", fontsize = 14, wrap = True)
    plt.xlim([0, 100])
    plt.ylim([np.min(residual_array[:,:,1])-0.5, np.max(residual_array[:,:,1])+0.5])
    plt.axhline(0, color='black')
    #plt.legend()
    plt.title("Difference in galaxies from percentage", fontsize = 14, wrap = True)
    plt.savefig("Total linear residual")
    plt.close()
    
    temp = pd.DataFrame(standard_deviation_array[:,0,:])
    temp = temp.loc[[temp[3].astype(float).idxmin()]]
    temp = temp.reset_index(drop=True)
    delta_z = temp[0][0]
    delta_p = temp[1][0]
    delta_mag = temp[2][0]
    
    print("The best delta values are delta z =  ", delta_z, ", delta p = ", delta_p, " and delta mag = ", delta_mag)