# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:39:56 2022

@author: r41331jc
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf


if __name__ == '__main__':
    
    print('\nStart')

    parser = argparse.ArgumentParser()
    parser.add_argument('--min-gal', dest='min_gal', type=int)
    parser.add_argument('--max-gal', dest='max_gal', type=int)
    parser.add_argument('--delta-z', dest='delta_z', type=float)
    parser.add_argument('--delta-p', dest='delta_p', type=float)
    parser.add_argument('--delta-mag', dest='delta_mag', type=float)
    parser.add_argument('--delta-mass', dest='delta_mass', type=float)
    parser.add_argument('--min-z', dest='max_z', type=float)
    parser.add_argument('--percent', dest='percent', type=float)
    
    args = parser.parse_args()
    
    # min_gal = 120
    # max_gal = 130
    # delta_z = 0.006 #sets width of sample box - Default optimised = 0.008
    # delta_p = 0.017 #sets height of smaple box - Default optimised = 0.016
    # delta_mag = 0.4 #Vary to find better base value - Default optimised = 0.5
    # delta_mass = 0.1
    # min_z = 0.05
    # percent = 66
    
    min_gal = args.min_gal
    max_gal = args.max_gal
    delta_z = args.delta_z #sets width of sample box - Default optimised = 0.008
    delta_p = args.delta_p #sets height of smaple box - Default optimised = 0.016
    delta_mag = args.delta_mag #Vary to find better base value - Default optimised = 0.5
    delta_mass = args.delta_mass #Vary to find better base value - Default optimised = 0.5
    min_z = args.min_z
    percent = args.percent
    
    count_array = []

    # The data
    full_data = pd.read_csv('full_data.csv', index_col=0)
    full_data = full_data.to_numpy()
    
    full_data_var = pd.read_csv('full_data_var.csv', index_col=0)
    full_data_var = full_data_var.to_numpy()

    print('Files appended, removing test sample')
    #Remove the test sample
    test_sample_names = full_data[min_gal:max_gal, 0] 

    full_dataframe = pd.DataFrame(full_data)
    full_dataframe_var = pd.DataFrame(full_data_var)
    test_sample = pd.DataFrame(columns=full_dataframe.columns)

    for name in test_sample_names:
        cond = full_dataframe[0] == name
        rows = full_dataframe.loc[cond, :]
        test_sample = test_sample.append(rows ,ignore_index=True)
        full_dataframe.drop(rows.index, inplace=True)
        full_dataframe_var.drop(rows.index, inplace=True)

    print('Beginning predictions')
    #If we want to operate over multiple galaxies, start a for loop here
    for test_name in test_sample_names:
    
        test_galaxy = test_sample[test_sample[0] == test_name]
        gal_max_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmax()]]
        gal_min_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmin()]]
        test_z = gal_max_z[4].astype(float).to_numpy()[0]
        test_p = gal_max_z[1].astype(float).to_numpy()[0]
        pred_z = min_z
        actual_p = gal_min_z[1].astype(float).to_numpy()[0]
        test_mag = gal_max_z[5].astype(float).to_numpy()[0]
        test_mass = gal_max_z[6].astype(float).to_numpy()[0]

        #Set values for smapling 
        upper_z = test_z + delta_z
        lower_z = test_z - delta_z
        upper_p = test_p + delta_p
        lower_p =test_p - delta_p

        immediate_sub_sample = full_dataframe[(full_dataframe[4].astype(float) < upper_z) & (full_dataframe[4].astype(float) >= lower_z) & (full_dataframe[1].astype(float) >= lower_p) & (full_dataframe[1].astype(float) <= upper_p)]
        unique_names = pd.unique(immediate_sub_sample[0])
            
        sim_sub_set = pd.DataFrame()
        sim_sub_set_var = pd.DataFrame()
        for name in unique_names:
            sim_sub_set = sim_sub_set.append(full_dataframe[full_dataframe[0] == name])
            sim_sub_set_var = sim_sub_set_var.append(full_dataframe_var[full_dataframe_var[0] == name])
        
        #Let's make some predictions

        prediction_list=[]
        weight_list = []
        sd_list=[]
    
        for name in unique_names:
            galaxy_data = sim_sub_set[sim_sub_set[0] == name]
            galaxy_data_var = sim_sub_set_var[sim_sub_set_var[0] == name]

            abs_diff_pred_z = abs(galaxy_data[4].astype(float) - pred_z)
            min_pos_pred = abs_diff_pred_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            next_min_pos_pred = abs_diff_pred_z.nsmallest(2).index[1]

            abs_diff_test_z = abs(galaxy_data[4].astype(float) - test_z)
            min_pos_test = abs_diff_test_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            nextmin_pos_test = abs_diff_test_z.nsmallest(2).index[1]

            estimate_predictions = galaxy_data.loc[[min_pos_pred]]
            estimate_predictions_var = galaxy_data_var.loc[[min_pos_pred]]
            grad_reference = galaxy_data.loc[[next_min_pos_pred]]

            diff_y = estimate_predictions[1].astype(float).to_numpy()[0] - grad_reference[1].astype(float).to_numpy()[0]
            diff_x = estimate_predictions[4].astype(float).to_numpy()[0] - grad_reference[4].astype(float).to_numpy()[0] #the astype and to numpy are to extract numbers from dataframe
            gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value

            minimum_point_seperation = pred_z - estimate_predictions[4].astype(float).to_numpy()[0]
            grad_correction = gradient * minimum_point_seperation
            grad_corrected_prediction = estimate_predictions[1].astype(float).to_numpy()[0] + grad_correction

            closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
            
            gaussain_p_variable = closest_vals[1].astype(float).to_numpy()[0]
            gaussian_z_variable = closest_vals[4].astype(float).to_numpy()[0]
            gaussian_mag_variable = closest_vals[5].astype(float).to_numpy()[0]
            gaussian_mass_variable = closest_vals[6].astype(float).to_numpy()[0]

            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p, test_z, delta_p/2, delta_z/2)
            mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)
            mass_weight = frf.mass_gaussian_weightings(gaussian_mass_variable, test_mass, delta_mass)
            
            weight = proximity_weight * mag_weight * mass_weight

            #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)

            prediction_list.append(grad_corrected_prediction)
            sd_list.append(estimate_predictions_var[2].astype(float).to_numpy()[0])
            weight_list.append(weight)
        
        mean_prediction = np.mean(prediction_list)
        mean_std = np.std(prediction_list)

        weighted_mean_numerator = np.sum(np.array(weight_list) * np.array(prediction_list))
        weighted_mean_denominator = np.sum(np.array(weight_list))
        weighted_mean = weighted_mean_numerator/weighted_mean_denominator

        weighted_std_numerator = np.sum(np.array(weight_list)*((np.array(prediction_list) - weighted_mean)**2))
        weighted_std_denominator = np.sum(np.array(weight_list))
        weighted_std = np.sqrt(weighted_std_numerator/weighted_std_denominator)

        plt.figure(figsize=(10,6))
        plt.suptitle('{3} Morphology Near Test Value Parameters z={0:.3f} p={1:.3f} with N={2} Galaxies. % = {4}\n'.format(test_z, test_p, len(unique_names), test_name, percent), fontsize=20, wrap=True)

        #Manipulate the weight list to turn into usable alphas
        weight_list_np = np.array(weight_list)
        #transform to interval [0, 1] using -1/log10(weight/10)
        logged_weights = np.log10(weight_list_np/10)
        alpha_per_gal = -1/logged_weights
        #Normalise the alphas to max at 0.8
        max_alpha = alpha_per_gal.max()
        norm_factor = 0.5/max_alpha
        norm_alphas_per_gal = alpha_per_gal * norm_factor

        plt.subplot(121)
        weight_index=0
        for name in unique_names:
            data_to_plot = sim_sub_set[sim_sub_set[0] == name]
            var_to_plot = sim_sub_set_var[sim_sub_set_var[0] == name]
            x_data = np.asarray(data_to_plot[4]).astype(float)
            y_data = np.asarray(data_to_plot[1]).astype(float)
            y_err = np.sqrt(np.asarray(var_to_plot[1]).astype(float))
            
            plt.errorbar(x_data, y_data, marker ='x', alpha=norm_alphas_per_gal[weight_index])
            weight_index+=1

        plt.errorbar(pred_z, weighted_mean, weighted_std, marker ='x', color = 'red', alpha=1, label='Weighted mean = {0:.3f}\nWeighted std = {1:.3f}\nTarget redshift = {2:.3f}\nActual liklihood = {3:.3f}'.format(weighted_mean, weighted_std, pred_z, actual_p)) #plotting average weighted by 2D gaussian
        plt.errorbar(pred_z, actual_p, marker = 'v', alpha = 0.75,  color = 'black', label='Actual Test prediction for new redshift')
        plt.errorbar(test_z, test_p, marker = 's', alpha = 0.75,  color = 'black', label='Original redshift prediction')
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('Prediction of Smoothness Liklihood', fontsize=15)
        plt.xlim([0, 0.25])
        plt.ylim([0, 1])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(framealpha=0)

        """
        Copy from here
        
        Change x_val, sd and weights
        """
        
        x_val = prediction_list
        sd = np.sqrt(sd_list)
        weights = weight_list

        kern_sum_val = np.zeros(0)
        area_kern_sum_val = np.zeros(0)
        width_val = np.zeros(0)
        allowed_val = np.zeros((0,2))
        sum_value = 0
        
        x_range = np.arange (0,1,0.001)
        for x in x_range:
           
            a = frf.kernal_create(x, x_val, sd, weights)
            kern_sum_val = np.append(kern_sum_val, a)
            area_kern_sum_val = np.append(area_kern_sum_val, a * 0.001)
        
        area_norm = np.sum(area_kern_sum_val)
        norm_kern_sum = kern_sum_val/area_norm
        
        for x in x_range:
           
            temp_array = np.array([[x, norm_kern_sum[int(x*1000)]]])
            allowed_val = np.vstack((allowed_val, temp_array))
        
        allowed_val = allowed_val[allowed_val[:, 1].argsort()]
        
        while np.sum(allowed_val[:,1]) > (percent * 10):
            allowed_val = np.delete(allowed_val, 0, axis = 0)
        
        allowed_val = allowed_val[allowed_val[:, 0].argsort()]
        
        width_val = np.append(width_val, allowed_val[0,0])
        
        for i in range(len(allowed_val)-1):
            if (bool((allowed_val[i+1,0] - allowed_val[i,0])<0.0015)  != bool((allowed_val[i,0]- allowed_val[i-1,0])<0.0015)):
                width_val = np.append(width_val, allowed_val[i,0])
        
        width_val = np.append(width_val, allowed_val[len(allowed_val)-1,0])
        round_act_p = int(1000*round(actual_p,3))
        
        sum_value += norm_kern_sum[round_act_p]
        
        plt.subplot(122)

        plt.plot(norm_kern_sum, x_range, label= 'Kerneled pdf')
        plt.ylabel("Smooth probability", fontsize=15)
        plt.xlabel("Normalised value", fontsize=15)
        plt.axhline(actual_p, label='Original prob = {0:.3f}'.format(actual_p), color='black')
        plt.ylim([0, 1])
        plt.xlim(left=0)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()

        for i in range((len(width_val)//2)):
    
            midrange = (width_val[i*2]+width_val[i*2+1])/2
            width_range = width_val[i*2+1]-width_val[i*2]
            plt.fill_betweenx(x_range,norm_kern_sum, where=(abs(x_range-midrange)<=width_range/2), color ='blue', alpha = 0.4)
        
        plt.legend(fontsize=12, framealpha=0)
        plt.gca().invert_xaxis()
        plt.savefig('grad_corr_{0}_with_kernels_adjusted.png'.format(test_name))
        plt.close()

    plt.close('all')
    print('End')