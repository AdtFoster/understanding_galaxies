# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:35:48 2022

@author: r41331jc
"""

import numpy as np
import pandas as pd
import argparse

import functions_for_redshifting_figures as frf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-gal', dest='min_gal', type=int)
    parser.add_argument('--max-gal', dest='max_gal', type=int)
    parser.add_argument('--min-delta-z', dest='min_delta_z', type=float)
    parser.add_argument('--max-delta-z', dest='max_delta_z', type=float)
    parser.add_argument('--step-delta-z', dest='step_delta_z', type=float)
    parser.add_argument('--min-delta-p', dest='min_delta_p', type=float)
    parser.add_argument('--max-delta-p', dest='max_delta_p', type=float)
    parser.add_argument('--step-delta-p', dest='step_delta_p', type=float)
    parser.add_argument('--min-delta-mag', dest='min_delta_mag', type=float)
    parser.add_argument('--max-delta-mag', dest='max_delta_mag', type=float)
    parser.add_argument('--step-delta-mag', dest='step_delta_mag', type=float)
    parser.add_argument('--min-delta-mass', dest='min_delta_mass', type=float)
    parser.add_argument('--max-delta-mass', dest='max_delta_mass', type=float)
    parser.add_argument('--step-delta-mass', dest='step_delta_mass', type=float)
    
    args = parser.parse_args()
    
    min_gal = 120
    max_gal = 130
    
    #min_gal = args.min_gal
    #max_gal = args.max_gal
    
    min_delta_z = 0.005
    max_delta_z = 0.007
    step_delta_z = 0.001
    min_delta_p = 0.015
    max_delta_p = 0.017
    step_delta_p = 0.001
    min_delta_mag = 0.4
    max_delta_mag = 0.6
    step_delta_mag = 0.1
    min_delta_mass = 0.1
    max_delta_mass = 0.3
    step_delta_mass = 0.1
    
    #min_delta_z = args.min_delta_z
    #max_delta_z = args.max_delta_z
    #step_delta_z = args.step_delta_z
    #min_delta_p = args.min_delta_p
    #max_delta_p = args.max_delta_p
    #step_delta_p = args.step_delta_p
    #min_delta_mag = args.min_delta_mag
    #max_delta_mag = args.max_delta_mag
    #step_delta_mag = args.step_delta_mag
    #min_delta_mass = args.min_delta_mass
    #max_delta_mass = args.max_delta_mass
    #step_delta_mass = args.step_delta_mass
    
    full_data = pd.read_csv('full_data.csv', index_col=0)
    full_data = full_data.to_numpy()
    
    full_data_var = pd.read_csv('full_data_var.csv', index_col=0)
    full_data_var = full_data_var.to_numpy()
    
    test_sample_names = full_data[min_gal:max_gal, 0] 
    
    full_data = pd.DataFrame(full_data)
    full_data_var = pd.DataFrame(full_data_var)
    test_sample = pd.DataFrame(columns=full_data.columns)
    
    for name in test_sample_names:
        cond = full_data[0] == name
        rows = full_data.loc[cond, :]
        test_sample = test_sample.append(rows ,ignore_index=True)
        full_data.drop(rows.index, inplace=True)
        full_data_var.drop(rows.index, inplace=True)
        
    standard_deviation_array = np.zeros((0,8))
    residual_array = np.zeros((0,2))
    count_array_perm = np.zeros((0,2))
    error_array_perm = np.zeros((0,2))
    
    count = 0
    
    print('Beginning predictions')
    for delta_z in np.arange(min_delta_z,max_delta_z,step_delta_z):
        for delta_p in np.arange(min_delta_p,max_delta_p,step_delta_p):
            for delta_mag in np.arange(min_delta_mag,max_delta_mag,step_delta_mag):
                for delta_mass in np.arange(min_delta_mass,max_delta_mass,step_delta_mass):
                    print(delta_z, delta_p, delta_mag, delta_mass)
                    
                    number_of_galaxies = 0
                    count_array = []
                    for x in range(10,95,5):
                        b = [x,0]
                        count_array.append(b)
                    count_array = np.asarray(count_array)
                    sum_value = 0
                                
                    #If we want to operate over multiple galaxies, start a for loop here
                    for name in test_sample_names:
                    
                        test_galaxy = test_sample[test_sample[0] == name]
                        gal_max_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmax()]]
                        gal_min_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmin()]]
                        test_z = gal_max_z[4].astype(float).to_numpy()[0]
                        test_p = gal_max_z[1].astype(float).to_numpy()[0]
                        pred_z = gal_min_z[4].astype(float).to_numpy()[0]
                        actual_p = gal_min_z[1].astype(float).to_numpy()[0]
                        test_mag = gal_max_z[5].astype(float).to_numpy()[0]
                        test_mass = gal_max_z[6].astype(float).to_numpy()[0]
                    
                        #Set values for smapling 
                        upper_z = test_z + delta_z
                        lower_z = test_z - delta_z
                        upper_p = test_p + delta_p
                        lower_p = test_p - delta_p
                    
                        immediate_sub_sample = full_data[(full_data[4].astype(float) < upper_z) & (full_data[4].astype(float) >= lower_z) & (full_data[1].astype(float) >= lower_p) & (full_data[1].astype(float) <= upper_p)]
                        unique_names = pd.unique(immediate_sub_sample[0])
                            
                        sim_sub_set = pd.DataFrame()
                        sim_sub_set_var = pd.DataFrame()
                        for unique_name in unique_names:
                            sim_sub_set = sim_sub_set.append(full_data[full_data[0] == unique_name])
                            sim_sub_set_var = sim_sub_set_var.append(full_data_var[full_data_var[0] == unique_name])
                        
                        
                        #Let's make some predictions
                    
                        prediction_list=[]
                        weight_list = []
                        sd_list = []
                    
                    
                        for unique_name in unique_names:
                            galaxy_data = sim_sub_set[sim_sub_set[0] == unique_name]
                            galaxy_data_var = sim_sub_set_var[sim_sub_set_var[0] == unique_name]
                    
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
                            sd_list.append(estimate_predictions_var[1].astype(float).to_numpy()[0])
                            weight_list.append(weight)
                        
                        mean_prediction = np.mean(prediction_list)
                        mean_std = np.std(prediction_list)
                    
                        weighted_mean_numerator = np.sum(np.array(weight_list) * np.array(prediction_list))
                        weighted_mean_denominator = np.sum(np.array(weight_list))
                        weighted_mean = weighted_mean_numerator/weighted_mean_denominator
                    
                        weighted_std_numerator = np.sum(np.array(weight_list)*((np.array(prediction_list) - weighted_mean)**2))
                        weighted_std_denominator = np.sum(np.array(weight_list))
                        weighted_std = np.sqrt(weighted_std_numerator/weighted_std_denominator)
                        
                        """
                        Copy from here
                        
                        Change x_val, sd and weights
                        """
                        
                        for percent in range(10,95, 5):
                        
                            x_val = prediction_list
                            sd = np.sqrt(sd_list)
                            weights = weight_list
                        
                            kern_sum_val = np.zeros(0)
                            area_kern_sum_val = np.zeros(0)
                            width_val = np.zeros(0)
                            allowed_val = np.zeros((0,2))
                            
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
                            
                            for i in range((len(width_val)//2)):
                        
                                midrange = (width_val[i*2]+width_val[i*2+1])/2
                                width_range = width_val[i*2+1]-width_val[i*2]
                                #plt.fill_between(x_range, norm_kern_sum, where=(abs(x_range-midrange)<=width_range/2), color ='blue', alpha = 0.4)
                                if (abs(actual_p-midrange)<=width_range/2):
                    
                                    if percent == percent:
                                        count_array[int((percent-10)/5),1] +=1
        
                            
                        number_of_galaxies +=1
                        print(number_of_galaxies)
                        
                    error_array = []
                    
                    for x in range(10,95,5):
                        a = [x,(number_of_galaxies * x / 100)]
                        error_array.append(a)
                    error_array = np.asarray(error_array)
                    
                    difference_array = error_array
                    difference_array = count_array[:,1] - error_array[:,1]
                    difference_array = np.dstack((error_array[:,0],difference_array))
                    difference_array = difference_array[0,:,:]
                    
                    standard_deviation = np.sqrt(np.sum(difference_array[:,1]**2)/(len(difference_array)-2))
                    
                    sum_x = np.sum(count_array[:,0])
                    sum_x_sqr = np.sum(count_array[:,0]**2)
                    sum_y = np.sum(count_array[:,1])
                    sum_y_sqr = np.sum(count_array[:,1]**2)
                    sum_xy = np.sum(count_array[:,0]*count_array[:,1])
                    n = len(count_array[:,0])
                    sxx = sum_x_sqr - ((sum_x**2)/n)
                    syy = sum_y_sqr - ((sum_y**2)/n)
                    sxy = sum_xy - ((sum_x*sum_y)/n)
                    y_mean = sum_y/n
                    
                    pearson = sxy/np.sqrt(sxx*syy)
                    
                    expec = np.sum(error_array[:,1])
                    expec_sqr = np.sum(error_array[:,1]**2)
                    
                    ssr = np.sum((count_array[:,1] - error_array[:,1])**2)
                    sst = np.sum((count_array[:,1] - y_mean)**2)
                    r_sqr = 1 - (ssr/sst)
                    temp = [delta_z, delta_p, delta_mag, delta_mass, standard_deviation, pearson, r_sqr, sum_value]
                    residual_array = np.vstack((residual_array,difference_array))
                    count_array_perm = np.vstack((count_array_perm,count_array))
                    error_array_perm = np.vstack((error_array_perm, error_array))
                    standard_deviation_array = np.vstack((standard_deviation_array,temp))
                    count +=1
    
    standard_deviation_array = pd.DataFrame(standard_deviation_array)
    standard_deviation_array = standard_deviation_array.sort_values(by=4)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    pd.DataFrame(count_array_perm).to_csv('count_array_perm.csv')
    pd.DataFrame(error_array_perm).to_csv('error_array_perm.csv')
    pd.DataFrame(residual_array).to_csv('residual_array.csv')
    pd.DataFrame(standard_deviation_array).to_csv('standard_deviation_array.csv')