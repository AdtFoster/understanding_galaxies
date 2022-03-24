# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:25:14 2022
@author: r41331jc
"""

import logging
#import string

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)

    logging.info('Begin creating squid diagrams')

    parser = argparse.ArgumentParser()
    parser.add_argument('--min-gal', dest='min_gal', type=int)
    parser.add_argument('--max-gal', dest='max_gal', type=int)
    parser.add_argument('--delta-z', dest='delta_z', type=float)
    parser.add_argument('--delta-p', dest='delta_p', type=float)
    parser.add_argument('--delta-mag', dest='delta_mag', type=float)
    parser.add_argument('--delta-mass', dest='delta_mass', type=float)
    parser.add_argument('--delta-conc', dest='delta_conc', type=float)
    parser.add_argument('--min-z', dest='min_z', type=float)
    parser.add_argument('--percent', dest='percent', type=float)
    parser.add_argument('--morphology', dest='morphology', type=str)
    parser.add_argument('--max-z', dest='max_z', type=float)

    args = parser.parse_args()
    
    min_gal = args.min_gal
    max_gal = args.max_gal
    delta_z = args.delta_z #sets width of sample box - Default optimised = 0.008
    delta_p = args.delta_p #sets height of smaple box - Default optimised = 0.016
    delta_mag = args.delta_mag #Vary to find better base value - Default optimised = 0.5
    delta_mass = args.delta_mass #Vary to find better base value - Default optimised = 0.5
    delta_conc = args.delta_conc #Vary to find better base value - Default optimised = 0.1
    min_z = args.min_z
    percent = args.percent
    morphology = args.morphology
    max_z = args.max_z
    
    count_array = []

    # The data
    full_data = pd.read_csv('output_csvs/full_data_1m_with_resizing.csv') #index_col=0
    full_data['elpetro_mass'] = np.log10(full_data['elpetro_mass'])
    
    #form a numpy array of the first [min_gal:max_gal] galaxy names
    test_sample_names = pd.unique(full_data['iauname'])[min_gal:max_gal] 
    #test_sample_names = full_data.loc[min_gal:max_gal, 'iauname'] 

    test_sample = full_data[full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)

    full_data = full_data[~full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)

    print('Beginning predictions')
    #If we want to operate over multiple galaxies, start a for loop here
    for test_name in test_sample_names:
    
        test_galaxy = test_sample[test_sample['iauname'] == test_name]
        gal_max_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmax()]]
        gal_min_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmin()]]
        test_z = gal_max_z['redshift'].values[0]
        test_p = gal_max_z[f'smooth-or-featured-dr5_{morphology}_prob'].values[0]
        pred_z = gal_min_z['redshift'].values[0]
        actual_p = gal_min_z[f'smooth-or-featured-dr5_{morphology}_prob'].values[0]
        test_mag = gal_max_z['elpetro_absmag_r'].values[0]
        test_mass = gal_max_z['elpetro_mass'].values[0]
        test_conc = gal_max_z['concentration'].values[0]

        #Set values for smapling 
        upper_z = test_z + delta_z
        lower_z = test_z - delta_z
        upper_p = test_p + delta_p
        lower_p =test_p - delta_p
        
        #Sets values for magnitude
        upper_mag = test_mag + delta_mag #sets upper box mag limit
        lower_mag = test_mag - delta_mag #sets lower box mag limit
        
        #Sets values for mass
        upper_mass = test_mass + delta_mass #sets upper box mass limit
        lower_mass = test_mass - delta_mass #sets lower box mass limit
        
        #Sets values for conc
        upper_conc = test_conc + delta_conc #sets upper box mass limit
        lower_conc = test_conc - delta_conc #sets lower box mass limit

        immediate_sub_sample = full_data[
                            (full_data['redshift'].astype(float) < upper_z) &
                            (full_data['redshift'].astype(float) >= lower_z) &
                            (full_data[f'smooth-or-featured-dr5_{morphology}_prob'].astype(float) >= lower_p) &
                            (full_data[f'smooth-or-featured-dr5_{morphology}_prob'].astype(float) <= upper_p) &
                            (full_data['elpetro_absmag_r'].astype(float) <= upper_mag) &
                            (full_data['elpetro_absmag_r'].astype(float) >= lower_mag) &
                            (full_data['elpetro_mass'].astype(float) <= upper_mass) &
                            (full_data['elpetro_mass'].astype(float) >= lower_mass) &
                            (full_data['concentration'].astype(float) <= upper_conc) &
                            (full_data['concentration'].astype(float) >= lower_conc)
                            ]

        galaxy_names_in_box = pd.unique(immediate_sub_sample['iauname'])
            
        sim_sub_set = full_data[full_data['iauname'].isin(galaxy_names_in_box)]
        assert len(sim_sub_set) > 0  
        #Let's make some predictions

        prediction_list=[]
        weight_list = []
        sd_list=[]
    
        for galaxy in galaxy_names_in_box:

            # df of all simulated instances of this particular examplar galaxy
            galaxy_data = sim_sub_set.query(f'iauname == "{galaxy}"').reset_index(drop=True)
                        
            abs_diff_pred_z = abs(galaxy_data['redshift'].astype(float) - pred_z)  # simulated redshift - target redshift

            # pick the 2 smallest simulated version of this examplar galaxy, and define as min and next_min
            min_pos_pred = abs_diff_pred_z.nsmallest(2).index[0] #nsmallest picks the n smallest values and puts them in df
            next_min_pos_pred = abs_diff_pred_z.nsmallest(2).index[1]
                        
            abs_diff_test_z = abs(galaxy_data['redshift'].astype(float) - test_z)
            min_pos_test = abs_diff_test_z.nsmallest(2).index[0] #nsmallest picks the n smallest values and puts them in df
            nextmin_pos_test = abs_diff_test_z.nsmallest(2).index[1]
                        
            estimate_predictions = galaxy_data.loc[[min_pos_pred]]
            grad_reference = galaxy_data.loc[[next_min_pos_pred]]

            #Calculating the difference between the closest and second closest prob and redshifts to the target prediction redshift
            diff_y = estimate_predictions[f'smooth-or-featured-dr5_{morphology}_prob'].values[0] - grad_reference[f'smooth-or-featured-dr5_{morphology}_prob'].values[0]
            diff_x = estimate_predictions['redshift'].values[0] - grad_reference['redshift'].values[0] 
            gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value
                        
            minimum_point_seperation = pred_z - estimate_predictions['redshift'].values[0]
            grad_correction = gradient * minimum_point_seperation
            grad_corrected_prediction = estimate_predictions[f'smooth-or-featured-dr5_{morphology}_prob'].values[0] + grad_correction
                                    
            closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
            
            gaussain_p_variable = closest_vals[f'smooth-or-featured-dr5_{morphology}_prob'].values[0]
            gaussian_z_variable = closest_vals['redshift'].values[0]
            gaussian_mag_variable = closest_vals['elpetro_absmag_r'].values[0]
            gaussian_mass_variable = closest_vals['elpetro_mass'].values[0]
            gaussian_conc_variable = closest_vals['concentration'].values[0]
                        
            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p, test_z, delta_p/2, delta_z/2)
            mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)
            mass_weight = frf.mass_gaussian_weightings(gaussian_mass_variable, test_mass, delta_mass)
            conc_weight = frf.conc_gaussian_weightings(gaussian_conc_variable, test_conc, delta_conc)
                        
            weight = proximity_weight * mag_weight * mass_weight * conc_weight
                        
            #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)
                        
            prediction_list.append(grad_corrected_prediction)
            sd_list.append(estimate_predictions[f'smooth-or-featured-dr5_{morphology}_var'].values[0])
            weight_list.append(weight)

        weighted_mean_numerator = np.sum(np.array(weight_list) * np.array(prediction_list))
        weighted_mean_denominator = np.sum(np.array(weight_list))
        weighted_mean = weighted_mean_numerator/weighted_mean_denominator
                        
        weighted_std_numerator = np.sum(np.array(weight_list)*((np.array(prediction_list) - weighted_mean)**2))
        weighted_std_denominator = np.sum(np.array(weight_list))
        weighted_std = np.sqrt(weighted_std_numerator/weighted_std_denominator)
        

        plt.figure(figsize=(10,6))
        plt.suptitle('{3} Morphology Near Test Value Parameters z={0:.3f} p={1:.3f} with N={2} Galaxies. % = {4}\n'.format(test_z, test_p, len(galaxy_names_in_box), test_name, percent), fontsize=20, wrap=True)

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
        
        for galaxy in galaxy_names_in_box:
            galaxy_data = sim_sub_set.query(f'iauname == "{galaxy}"').reset_index(drop=True)
            x_data = np.asarray(galaxy_data['redshift']).astype(float)
            y_data = np.asarray(galaxy_data[f'smooth-or-featured-dr5_{morphology}_prob']).astype(float)
            y_err = np.sqrt(np.asarray(galaxy_data[f'smooth-or-featured-dr5_{morphology}_var']).astype(float))
            
            plt.errorbar(x_data, y_data, marker ='x', alpha=norm_alphas_per_gal[weight_index])
            weight_index+=1

        plt.errorbar(pred_z, weighted_mean, weighted_std, marker ='x', color = 'red', alpha=1, label='Weighted mean = {0:.3f}\nWeighted std = {1:.3f}\nTarget redshift = {2:.3f}\nActual liklihood = {3:.3f}'.format(weighted_mean, weighted_std, pred_z, actual_p)) #plotting average weighted by 2D gaussian
        plt.errorbar(pred_z, actual_p, marker = 'v', alpha = 0.75,  color = 'black', label='Actual Test prediction for new redshift')
        plt.errorbar(test_z, test_p, marker = 's', alpha = 0.75,  color = 'black', label='Original redshift prediction')
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel(f'Prediction of {morphology} Liklihood', fontsize=15)
        plt.xlim([0, max_z])
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
            if (bool((allowed_val[i+1,0] - allowed_val[i,0])<0.0015) != bool((allowed_val[i,0] - allowed_val[i-1,0])<0.0015)):
                width_val = np.append(width_val, allowed_val[i,0])
        
        width_val = np.append(width_val, allowed_val[len(allowed_val)-1,0])
        round_act_p = int(1000*round(actual_p,3))
        
        sum_value += norm_kern_sum[round_act_p]
        
        plt.subplot(122)

        plt.plot(norm_kern_sum, x_range, label= 'Kerneled pdf')
        plt.ylabel(f'{morphology} probability', fontsize=15)
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
        plt.savefig('squid_plots/grad_corr_{0}_with_kernels_adjusted_{1}.jpeg'.format(test_name, morphology))
        plt.close()

    plt.close('all')
    logging.info('Finished producing squid diagrams')