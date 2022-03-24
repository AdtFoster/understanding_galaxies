# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:35:48 2022
@author: r41331jc
"""
import logging

import numpy as np
import pandas as pd
import argparse

import functions_for_redshifting_figures as frf

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

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
    parser.add_argument('--min-delta-conc', dest='min_delta_conc', type=float)
    parser.add_argument('--max-delta-conc', dest='max_delta_conc', type=float)
    parser.add_argument('--step-delta-conc', dest='step_delta_conc', type=float)
    parser.add_argument('--initial-delta-p', dest='initial_delta_p', type=float)
    parser.add_argument('--initial-delta-mag', dest='initial_delta_mag', type=float)
    parser.add_argument('--initial-delta-mass', dest='initial_delta_mass', type=float)
    parser.add_argument('--initial-delta-conc', dest='initial_delta_conc', type=float)
    
    args = parser.parse_args()
    
    min_gal = args.min_gal
    max_gal = args.max_gal
    
    min_delta_z = args.min_delta_z
    max_delta_z = args.max_delta_z
    step_delta_z = args.step_delta_z
    min_delta_p = args.min_delta_p
    max_delta_p = args.max_delta_p
    step_delta_p = args.step_delta_p
    min_delta_mag = args.min_delta_mag
    max_delta_mag = args.max_delta_mag
    step_delta_mag = args.step_delta_mag
    min_delta_mass = args.min_delta_mass
    max_delta_mass = args.max_delta_mass
    step_delta_mass = args.step_delta_mass
    min_delta_conc = args.min_delta_conc
    max_delta_conc = args.max_delta_conc
    step_delta_conc = args.step_delta_conc
    initial_delta_p = args.initial_delta_p
    initial_delta_mag = args.initial_delta_mag
    initial_delta_mass = args.initial_delta_mass
    initial_delta_conc = args.initial_delta_conc   
    
    # created by create_dataframe.py
    full_data = pd.read_csv('full_data_1m_with_resizing.csv') #index_col=0
    full_data['elpetro_mass'] = np.log10(full_data['elpetro_mass'])
    
    #form a numpy array of the first [min_gal:max_gal] galaxy names
    test_sample_names = pd.unique(full_data['iauname'])[min_gal:max_gal] 
    #test_sample_names = full_data.loc[min_gal:max_gal, 'iauname'] 

    test_sample = full_data[full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)

    full_data = full_data[~full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)
    
    standard_deviation_array = np.zeros((0,9))
    residual_array = np.zeros((0,2))
    count_array_perm = np.zeros((0,2))
    error_array_perm = np.zeros((0,2))
    
    count = 0
    
    def optimise (delta_z, delta_p, delta_mag, delta_mass, delta_conc, standard_deviation_array, residual_array, count_array_perm, error_array_perm, count):
        logging.info('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(delta_z, delta_p, delta_mag, delta_mass, delta_conc))
        
        number_of_galaxies = 0
        count_array = []
        for x in range(10,95,5):
            b = [x,0]
            count_array.append(b)
        count_array = np.asarray(count_array)
        sum_value = 0
                    
        #If we want to operate over multiple galaxies, start a for loop here
        for name in test_sample_names:  # rename test_galaxy_name
        
            test_galaxy = test_sample[test_sample['iauname'] == name]
            gal_max_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmax()]]
            gal_min_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmin()]]
            test_z = gal_max_z['redshift'].values[0]
            test_p = gal_max_z['smooth-or-featured-dr5_smooth_prob'].values[0]
            pred_z = gal_min_z['redshift'].values[0]
            actual_p = gal_min_z['smooth-or-featured-dr5_smooth_prob'].values[0]
            test_mag = gal_max_z['elpetro_absmag_r'].values[0]
            test_mass = gal_max_z['elpetro_mass'].values[0]
            test_conc = gal_max_z['concentration'].values[0]
        
            #Set values for smapling 
            upper_z = test_z + delta_z
            lower_z = test_z - delta_z
            upper_p = test_p + delta_p
            lower_p = test_p - delta_p
            
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
                (full_data['smooth-or-featured-dr5_smooth_prob'].astype(float) >= lower_p) &
                (full_data['smooth-or-featured-dr5_smooth_prob'].astype(float) <= upper_p) &
                (full_data['elpetro_absmag_r'].astype(float) <= upper_mag) &
                (full_data['elpetro_absmag_r'].astype(float) >= lower_mag) &
                (full_data['elpetro_mass'].astype(float) <= upper_mass) &
                (full_data['elpetro_mass'].astype(float) >= lower_mass) &
                (full_data['concentration'].astype(float) <= upper_conc) &
                (full_data['concentration'].astype(float) >= lower_conc)
                ]
                # mag and mass are extra weightings, don't change num galaxies in the box
            
            #find unique galaxy names within catchment area
            galaxy_names_in_box = pd.unique(immediate_sub_sample['iauname'])
            logging.info(len(galaxy_names_in_box))
            if len(galaxy_names_in_box) > 10: #could change 10 to variable?
                # begin debiasing

                # select all galaxies within specified box
                sim_sub_set = full_data[full_data['iauname'].isin(galaxy_names_in_box)]
                assert len(sim_sub_set) > 0
                
                #Let's make some predictions
            
                prediction_list=[]
                weight_list = []
                sd_list = []
            
                for galaxy in galaxy_names_in_box:

                    # df of all simulated instances of this particular examplar galaxy
                    galaxy_data = sim_sub_set.query(f'iauname == "{galaxy}"').reset_index(drop=True)
            
                    abs_diff_pred_z = abs(galaxy_data['redshift'].astype(float) - pred_z)  # simulated redshift - target redshift

                    # pick the 2 smallest simulated version of this examplar galaxy, and define as min and next_min
                    min_pos_pred = abs_diff_pred_z.nsmallest(2).index[0] #nsmallest picks the n smallest values and puts them in df
                    next_min_pos_pred = abs_diff_pred_z.nsmallest(2).index[1]
            
                    abs_diff_test_z = abs(galaxy_data['redshift'].astype(float) - test_z)
                    min_pos_test = abs_diff_test_z.nsmallest(2).index[0] #nsmallest picks the n smallest values and puts them in df
            
                    estimate_predictions = galaxy_data.loc[[min_pos_pred]]
                    grad_reference = galaxy_data.loc[[next_min_pos_pred]]

                    #Calculating the difference between the closest and second closest prob and redshifts to the target prediction redshift
                    diff_y = estimate_predictions['smooth-or-featured-dr5_smooth_prob'].values[0] - grad_reference['smooth-or-featured-dr5_smooth_prob'].values[0]
                    diff_x = estimate_predictions['redshift'].values[0] - grad_reference['redshift'].values[0] 
                    gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value
            
                    minimum_point_seperation = pred_z - estimate_predictions['redshift'].values[0]
                    grad_correction = gradient * minimum_point_seperation
                    grad_corrected_prediction = estimate_predictions['smooth-or-featured-dr5_smooth_prob'].values[0] + grad_correction
                        
                    closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
                    
                    gaussain_p_variable = closest_vals['smooth-or-featured-dr5_smooth_prob'].values[0]
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
                    sd_list.append(estimate_predictions['smooth-or-featured-dr5_smooth_var'].values[0])
                    weight_list.append(weight)
            
                
            
                """
                Copy from here
                
                Change x_val, sd and weights
                """
                
                for percent in range(10,95,5):
                
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
                
            else:
                logging.warning(f'Skipping galaxy {name} as not enough examplars in box')
                
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
        
        ssr = np.sum((count_array[:,1] - error_array[:,1])**2)
        sst = np.sum((count_array[:,1] - y_mean)**2)
        r_sqr = 1 - (ssr/sst)
        temp = [delta_z, delta_p, delta_mag, delta_mass, delta_conc, standard_deviation, pearson, r_sqr, sum_value]
        logging.info(temp)
        residual_array = np.vstack((residual_array,difference_array))
        count_array_perm = np.vstack((count_array_perm,count_array))
        error_array_perm = np.vstack((error_array_perm, error_array))
        standard_deviation_array = np.vstack((standard_deviation_array,temp))
        return residual_array, count_array_perm, error_array_perm, standard_deviation_array
        
        count +=1

        logging.info(count)
    
    logging.info('Beginning predictions')
    
    # try many options for box size
    for delta_z in np.arange(min_delta_z,max_delta_z,step_delta_z):
        residual_array, count_array_perm, error_array_perm, standard_deviation_array = optimise(delta_z, initial_delta_p, initial_delta_mag, initial_delta_mass, initial_delta_conc, standard_deviation_array, residual_array, count_array_perm, error_array_perm, count)
    
    standard_deviation_array = pd.DataFrame(standard_deviation_array)
    standard_deviation_array = standard_deviation_array.sort_values(by=4)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    for delta_p in np.arange(min_delta_p,max_delta_p,step_delta_p):
        residual_array, count_array_perm, error_array_perm, standard_deviation_array = optimise(standard_deviation_array[0][0], delta_p, initial_delta_mag, initial_delta_mass, initial_delta_conc, standard_deviation_array, residual_array, count_array_perm, error_array_perm, count)
    
    standard_deviation_array = pd.DataFrame(standard_deviation_array)
    standard_deviation_array = standard_deviation_array.sort_values(by=4)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    for delta_mag in np.arange(min_delta_mag,max_delta_mag,step_delta_mag):
        residual_array, count_array_perm, error_array_perm, standard_deviation_array = optimise(standard_deviation_array[0][0], standard_deviation_array[0][1], delta_mag, initial_delta_mass, initial_delta_conc, standard_deviation_array, residual_array, count_array_perm, error_array_perm, count)
    
    standard_deviation_array = pd.DataFrame(standard_deviation_array)
    standard_deviation_array = standard_deviation_array.sort_values(by=4)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    for delta_mass in np.arange(min_delta_mass,max_delta_mass,step_delta_mass):
        residual_array, count_array_perm, error_array_perm, standard_deviation_array = optimise(standard_deviation_array[0][0], standard_deviation_array[0][1], standard_deviation_array[0][2], delta_mass, initial_delta_conc, standard_deviation_array, residual_array, count_array_perm, error_array_perm, count)
    
    standard_deviation_array = pd.DataFrame(standard_deviation_array)
    standard_deviation_array = standard_deviation_array.sort_values(by=4)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    for delta_conc in np.arange(min_delta_conc,max_delta_conc,step_delta_conc):
        residual_array, count_array_perm, error_array_perm, standard_deviation_array = optimise(standard_deviation_array[0][0], standard_deviation_array[0][1], standard_deviation_array[0][2], standard_deviation_array[0][3], delta_conc, standard_deviation_array, residual_array, count_array_perm, error_array_perm, count)
            
    standard_deviation_array = pd.DataFrame(standard_deviation_array)
    standard_deviation_array = standard_deviation_array.sort_values(by=4)
    standard_deviation_array = standard_deviation_array.to_numpy()
    
    pd.DataFrame(count_array_perm).to_csv('output_csvs/count_array_perm.csv')
    pd.DataFrame(error_array_perm).to_csv('output_csvs/error_array_perm.csv')
    pd.DataFrame(residual_array).to_csv('output_csvs/residual_array.csv')
    pd.DataFrame(standard_deviation_array).to_csv('output_csvs/standard_deviation_array.csv')

    logging.info('Sucessfully saved debiased values - exiting')