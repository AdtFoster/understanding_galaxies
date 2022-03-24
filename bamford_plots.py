# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:16:53 2022

@author: r41331jc
"""

import numpy as np
import functions_for_redshifting_figures as frf
import pandas as pd
import logging
import argparse


if __name__ == '__main__':
    print('Begin \n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-interval', dest='update_interval', type=int)
    parser.add_argument('--threshold-val', dest='threshold_val', type=float)
    parser.add_argument('--delta-z', dest='delta_z', type=float)
    parser.add_argument('--delta-p', dest='delta_p', type=float)
    parser.add_argument('--delta-mag', dest='delta_mag', type=float)
    parser.add_argument('--delta-mass', dest='delta_mass', type=float)
    parser.add_argument('--delta-conc', dest='delta_conc', type=float)
    parser.add_argument('--rounding', dest='rounding', type=float)
    
    args = parser.parse_args()
    
    full_data = pd.read_csv('full_data_1m_with_resizing.csv') #index_col=0
    full_data = full_data.sort_values(by = 'redshift')
    full_data['elpetro_mass'] = np.log10(full_data['elpetro_mass'])
    
    #delta_z = args.delta_z #sets width of sample box - Default optimised = 0.008
    #delta_p = args.delta_p #sets height of smaple box - Default optimised = 0.016
    #delta_mag = args.delta_mag #Vary to find better base value - Default optimised = 0.5
    #delta_mass = args.delta_mass #Vary to find better base value - Default optimised = 1.0
    #delta_conc = args.delta_conc #Vary to find better base value - Default optimised = 0.1
    
    rounding = args.rounding
    cut_threshold = args.thrshold_val
    update_interval = args.update_interval
    
    full_data_array_first_cut=np.zeros((0, 9))
    full_data_array_second_cut=np.zeros((0, 9))
    full_data_array_third_cut=np.zeros((0, 9))
    proportions_by_redshift_by_cut = []
    
    first_mag_cut = full_data[(full_data["elpetro_absmag_r"] < -18 ) & (full_data["elpetro_absmag_r"] >= -20)]
    second_mag_cut = full_data[(full_data["elpetro_absmag_r"] < -20 ) & (full_data["elpetro_absmag_r"] >= -21)]
    third_mag_cut = full_data[(full_data["elpetro_absmag_r"] < -21 ) & (full_data["elpetro_absmag_r"] >= -24)]
    
    figure_mag_lims = ['-18 to -20',
                        '-20 to -21',
                        '-21 to -24']
    
    figure_save_names = ['smoothness_cut_18_20_{0}_graph_redshift_certain_classification_extended_data_remade.png'.format(cut_threshold), 
                        'smoothness_cut_20_21_{0}_graph_redshift_certain_classification_extended_data_remade.png'.format(cut_threshold),
                        'smoothness_cut_21_24_{0}_graph_redshift_certain_classification_extended_data_remade.png'.format(cut_threshold)]
    
    i = 0
    for cut in [first_mag_cut, second_mag_cut, third_mag_cut]:
        #full_data_array[:, 4]=np.round(full_data_array[:, 4].astype(float), 2) #rounds the redshift values to 2 dp for binning
        cut["redshift"] = np.round(cut["redshift"] * (1/rounding))/(1/rounding)
    
        cut = cut.sort_values(by = "redshift").reset_index(drop = True)#sorts all data based on ascending redshift
        
        df = pd.DataFrame(columns=['redshift','smooth','featured','artifact'])
        for redshift in np.arange(cut["redshift"].min(), 0.12+rounding, rounding):
            
            redshift = np.round(redshift * (1/rounding))/(1/rounding)
            df_temp = cut[cut["redshift"] == redshift]
            confident_smooth = df_temp[(df_temp["smooth-or-featured-dr5_smooth_prob"] > cut_threshold)].count()[0]
            confident_featured = df_temp[(df_temp["smooth-or-featured-dr5_featured-or-disk_prob"] > cut_threshold)].count()[0]
            confident_artifact= df_temp[(df_temp["smooth-or-featured-dr5_artifact_prob"] > cut_threshold)].count()[0]
            total = confident_smooth + confident_featured + confident_artifact
            df_redshift = pd.DataFrame([[redshift, confident_smooth/total, confident_featured/total, confident_artifact/total]], columns=df.columns)
            df = pd.concat([df,df_redshift])
        
        frf.error_bar_smoothness_3(df['redshift'].values, df['smooth'].values, df['featured'].values, df['artifact'].values, save_name=figure_save_names[i], title='Galaxy Morphology with Redshift for Magnitudes {0}'.format(figure_mag_lims[i]), xlabel='Redshift', ylabel='Proportion of expected predictions', ylimits=[0, 1], xlimits=[0, 0.12])
        i+=1
    
    print("Original plots plotted")
    
    figure_save_names = ['smoothness_cut_18_20_{0}_graph_redshift_certain_classification_extended_data_remade_debiased.png'.format(cut_threshold), 
                        'smoothness_cut_20_21_{0}_graph_redshift_certain_classification_extended_data_remade_debiased.png'.format(cut_threshold),
                        'smoothness_cut_21_24_{0}_graph_redshift_certain_classification_extended_data_remade_debiased.png'.format(cut_threshold)]
    
    j = 0
    for cut in [first_mag_cut, second_mag_cut, third_mag_cut]:
        df = pd.DataFrame(columns=['redshift','smooth','featured','artifact'])
        redshifts = np.arange(cut["redshift"].min(), 0.12+rounding, rounding)
        for pred_redshift in redshifts:
            weighted_means_list_smooth = [] #list for each weighted mean prediction (smooth prediction)
            weighted_means_list_featured = [] #list for each weighted mean prediction (featured prediction)
            weighted_means_list_artifact = [] #list for each weighted mean prediction (artifact prediction)
            
            count_smooth = [] #list for each weighted mean prediction (smooth prediction)
            count_featured = [] #list for each weighted mean prediction (featured prediction)
            count_artifact = [] #list for each weighted mean prediction (artifact prediction)
            
            cut_names = pd.unique(full_data['iauname'])
            
            for i in np.arange(0, len(cut_names), len(cut_names)/10):
                
                logging.info('Extracting test sample')
                # Remove the test sample
                test_sample_names = pd.unique(full_data['iauname'])[int(np.floor(i)):int(np.floor(i+len(cut_names)/10))] 
                assert len(test_sample_names) > 0
                
                test_sample = full_data[full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)
                
                full_data_no_test = full_data[~full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)
                
                logging.info('Beginning predictions on {} galaxies'.format(len(test_sample_names)))
                #If we want to operate over multiple galaxies, start a for loop here
                test_gal_number = 0 #count number of gals which have been processed
                skipped_gal = 0        
            
                for test_name in test_sample_names:
                    
                    if (test_gal_number % update_interval == 0):
                        logging.info('completed {0} of {1} galaxy debias predictions'.format(test_gal_number, len(test_sample_names))) #prints progress every {number} galaxy debias predictions
                    
                    test_galaxy = test_sample[test_sample['iauname'] == test_name]
                    gal_max_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmax()]]
                    gal_min_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmin()]]
                    test_z = gal_max_z['redshift'].values[0]
                    pred_z = pred_redshift
                    test_mag = gal_max_z['elpetro_absmag_r'].values[0]
                    test_mass = gal_max_z['elpetro_mass'].values[0]
                    test_conc = gal_max_z['concentration'].values[0]
                                    
                    #identify the 3 prob variables for the simulated image (could function)
                    test_p_smooth = gal_max_z['smooth-or-featured-dr5_smooth_prob'].values[0] #converts prob at max z to useable float format
                    test_p_featured = gal_max_z['smooth-or-featured-dr5_featured-or-disk_prob'].values[0] #converts prob at max z to useable float format
                    test_p_artifact = gal_max_z['smooth-or-featured-dr5_artifact_prob'].values[0] #converts prob at max z to useable float format
                
                    #identify the 3 prob variables for non-simulated image for comparison 
                    actual_p_smooth = gal_min_z['smooth-or-featured-dr5_smooth_prob'].values[0] #prob of smooth
                    actual_p_featured = gal_min_z['smooth-or-featured-dr5_featured-or-disk_prob'].values[0] #prob of featured
                    actual_p_artifact = gal_min_z['smooth-or-featured-dr5_artifact_prob'].values[0] #prob of artifact
                    
                    #Set values for smapling 
                    upper_z = test_z + delta_z #sets upper box z limit
                    lower_z = test_z - delta_z #sets lower box z limit
                    
                    #set limits for each of the 3 morphologies
                    upper_p_smooth = test_p_smooth + delta_p #sets upper p box limit for smooth prediction
                    upper_p_featured = test_p_featured + delta_p #sets upper p box limit for featured prediction
                    upper_p_artifact = test_p_artifact + delta_p #sets upper p box limit for artifact prediction
                    
                    lower_p_smooth = test_p_smooth - delta_p #sets lower p box limit for smooth
                    lower_p_featured = test_p_featured - delta_p #sets lower p box limit for featured
                    lower_p_artifact = test_p_artifact - delta_p #sets lower p box limit for artifact
                    
                    #Sets values for magnitude
                    upper_mag = test_mag + delta_mag #sets upper box mag limit
                    lower_mag = test_mag - delta_mag #sets lower box mag limit
                    
                    #Sets values for mass
                    upper_mass = test_mass + delta_mass #sets upper box mass limit
                    lower_mass = test_mass - delta_mass #sets lower box mass limit
                    
                    #Sets values for conc
                    upper_conc = test_conc + delta_conc #sets upper box mass limit
                    lower_conc = test_conc - delta_conc #sets lower box mass limit
                
                    #sub sample for each morphology
                    immediate_sub_sample_smooth = full_data_no_test[
                                        (full_data_no_test['redshift'].astype(float) <= upper_z) &
                                        (full_data_no_test['redshift'].astype(float) >= lower_z) &
                                        (full_data_no_test['smooth-or-featured-dr5_smooth_prob'].astype(float) >= lower_p_smooth) &
                                        (full_data_no_test['smooth-or-featured-dr5_smooth_prob'].astype(float) <= upper_p_smooth) &
                                        (full_data_no_test['elpetro_absmag_r'].astype(float) <= upper_mag) &
                                        (full_data_no_test['elpetro_absmag_r'].astype(float) >= lower_mag) &
                                        (full_data_no_test['elpetro_mass'].astype(float) <= upper_mass) &
                                        (full_data_no_test['elpetro_mass'].astype(float) >= lower_mass) &
                                        (full_data_no_test['concentration'].astype(float) <= upper_conc) &
                                        (full_data_no_test['concentration'].astype(float) >= lower_conc)
                                        ] #samples galaxies within box limits
                    
                    immediate_sub_sample_featured = full_data_no_test[
                                        (full_data_no_test['redshift'].astype(float) <= upper_z) &
                                        (full_data_no_test['redshift'].astype(float) >= lower_z) &
                                        (full_data_no_test['smooth-or-featured-dr5_featured-or-disk_prob'].astype(float) >= lower_p_featured) &
                                        (full_data_no_test['smooth-or-featured-dr5_featured-or-disk_prob'].astype(float) <= upper_p_featured) &
                                        (full_data_no_test['elpetro_absmag_r'].astype(float) <= upper_mag) &
                                        (full_data_no_test['elpetro_absmag_r'].astype(float) >= lower_mag) &
                                        (full_data_no_test['elpetro_mass'].astype(float) <= upper_mass) &
                                        (full_data_no_test['elpetro_mass'].astype(float) >= lower_mass) &
                                        (full_data_no_test['concentration'].astype(float) <= upper_conc) &
                                        (full_data_no_test['concentration'].astype(float) >= lower_conc)
                                        ] #samples galaxies within box limits
                    immediate_sub_sample_artifact = full_data_no_test[
                                        (full_data_no_test['redshift'].astype(float) <= upper_z) &
                                        (full_data_no_test['redshift'].astype(float) >= lower_z) &
                                        (full_data_no_test['smooth-or-featured-dr5_artifact_prob'].astype(float) >= lower_p_artifact) &
                                        (full_data_no_test['smooth-or-featured-dr5_artifact_prob'].astype(float) <= upper_p_artifact) &
                                        (full_data_no_test['elpetro_absmag_r'].astype(float) <= upper_mag) &
                                        (full_data_no_test['elpetro_absmag_r'].astype(float) >= lower_mag) &
                                        (full_data_no_test['elpetro_mass'].astype(float) <= upper_mass) &
                                        (full_data_no_test['elpetro_mass'].astype(float) >= lower_mass) &
                                        (full_data_no_test['concentration'].astype(float) <= upper_conc) &
                                        (full_data_no_test['concentration'].astype(float) >= lower_conc)
                                        ] #samples galaxies within box limits
                    
                    print(len(immediate_sub_sample_smooth), len(immediate_sub_sample_featured), len(immediate_sub_sample_artifact))
                    
                    #requires at least 10 nearby galaxies for debiasing
                    if len(immediate_sub_sample_smooth)<10:
                        skipped_gal+=1
                        continue
                    elif len(immediate_sub_sample_featured)<10:
                        skipped_gal+=1
                        continue
                    elif len(immediate_sub_sample_artifact)<10:
                        skipped_gal+=1
                        continue
                    
                    #unique names for each sub sample
                    galaxy_names_in_box_smooth = pd.unique(immediate_sub_sample_smooth['iauname']) #names of subset galaxies for smooth
                    galaxy_names_in_box_featured = pd.unique(immediate_sub_sample_featured['iauname']) #names of subset galaxies for featured
                    galaxy_names_in_box_artifact = pd.unique(immediate_sub_sample_artifact['iauname']) #names of subset galaxies for artifact
                
                    sim_sub_set_smooth = full_data_no_test[full_data_no_test['iauname'].isin(galaxy_names_in_box_smooth)]
                    sim_sub_set_featured = full_data_no_test[full_data_no_test['iauname'].isin(galaxy_names_in_box_featured)]
                    sim_sub_set_artifact = full_data_no_test[full_data_no_test['iauname'].isin(galaxy_names_in_box_artifact)]
                    
                    """
                    The key elements in paly here are the 3 actual_p variables (_smooth, _featured and _artifact) which give the non simulated
                    probabilites for the galaxy and the sim_sub_set dataframes which we want to use for predicting the debiased value for each
                    of the smooth (1) featured (2) and artifact (3). 
                    """
                    #Let's make some predictions
                
                    prediction_list_smooth = []
                    prediction_list_featured = []
                    prediction_list_artifact = []
                    
                    weight_list_smooth = []
                    weight_list_featured = []
                    weight_list_artifact = []
                    
                    sd_list_smooth = []
                    sd_list_featured = []
                    sd_list_artifact = []
                    
                    #for loop for the smooth gradient corrections 
                    for galaxy in galaxy_names_in_box_smooth:
                
                        # df of all simulated instances of this particular examplar galaxy
                        galaxy_data = sim_sub_set_smooth.query(f'iauname == "{galaxy}"').reset_index(drop=True)
                                    
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
                
                        proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_smooth, test_z, delta_p/2, delta_z/2)
                        mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)
                        mass_weight = frf.mass_gaussian_weightings(gaussian_mass_variable, test_mass, delta_mass)
                        conc_weight = frf.conc_gaussian_weightings(gaussian_conc_variable, test_conc, delta_conc)
                                    
                        weight = proximity_weight * mag_weight * mass_weight * conc_weight
                                    
                        #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)
                                    
                        prediction_list_smooth.append(grad_corrected_prediction)
                        sd_list_smooth.append(estimate_predictions['smooth-or-featured-dr5_smooth_var'].values[0])
                        weight_list_smooth.append(weight)
                    
                    #for loop for the featured gradient predictions
                    for galaxy in galaxy_names_in_box_featured:
                
                        # df of all simulated instances of this particular examplar galaxy
                        galaxy_data = sim_sub_set_featured.query(f'iauname == "{galaxy}"').reset_index(drop=True)
                                    
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
                        diff_y = estimate_predictions['smooth-or-featured-dr5_featured-or-disk_prob'].values[0] - grad_reference['smooth-or-featured-dr5_featured-or-disk_prob'].values[0]
                        diff_x = estimate_predictions['redshift'].values[0] - grad_reference['redshift'].values[0] 
                        gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value
                                    
                        minimum_point_seperation = pred_z - estimate_predictions['redshift'].values[0]
                        grad_correction = gradient * minimum_point_seperation
                        grad_corrected_prediction = estimate_predictions['smooth-or-featured-dr5_featured-or-disk_prob'].values[0] + grad_correction
                                                
                        closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
                                            
                        gaussain_p_variable = closest_vals['smooth-or-featured-dr5_featured-or-disk_prob'].values[0]
                        gaussian_z_variable = closest_vals['redshift'].values[0]
                        gaussian_mag_variable = closest_vals['elpetro_absmag_r'].values[0]
                        gaussian_mass_variable = closest_vals['elpetro_mass'].values[0]
                        gaussian_conc_variable = closest_vals['concentration'].values[0]
                
                        proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_smooth, test_z, delta_p/2, delta_z/2)
                        mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)
                        mass_weight = frf.mass_gaussian_weightings(gaussian_mass_variable, test_mass, delta_mass)
                        conc_weight = frf.conc_gaussian_weightings(gaussian_conc_variable, test_conc, delta_conc)
                                    
                        weight = proximity_weight * mag_weight * mass_weight * conc_weight
                                    
                        #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)
                                    
                        prediction_list_featured.append(grad_corrected_prediction)
                        sd_list_featured.append(estimate_predictions['smooth-or-featured-dr5_featured-or-disk_var'].values[0])
                        weight_list_featured.append(weight)
                        
                    #for loop for the artifact gradient predictions
                    for galaxy in galaxy_names_in_box_artifact:
                
                        # df of all simulated instances of this particular examplar galaxy
                        galaxy_data = sim_sub_set_artifact.query(f'iauname == "{galaxy}"').reset_index(drop=True)
                                    
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
                        diff_y = estimate_predictions['smooth-or-featured-dr5_artifact_prob'].values[0] - grad_reference['smooth-or-featured-dr5_artifact_prob'].values[0]
                        diff_x = estimate_predictions['redshift'].values[0] - grad_reference['redshift'].values[0] 
                        gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value
                                    
                        minimum_point_seperation = pred_z - estimate_predictions['redshift'].values[0]
                        grad_correction = gradient * minimum_point_seperation
                        grad_corrected_prediction = estimate_predictions['smooth-or-featured-dr5_artifact_prob'].values[0] + grad_correction
                                                
                        closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
                                            
                        gaussain_p_variable = closest_vals['smooth-or-featured-dr5_artifact_prob'].values[0]
                        gaussian_z_variable = closest_vals['redshift'].values[0]
                        gaussian_mag_variable = closest_vals['elpetro_absmag_r'].values[0]
                        gaussian_mass_variable = closest_vals['elpetro_mass'].values[0]
                        gaussian_conc_variable = closest_vals['concentration'].values[0]
                
                        proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_smooth, test_z, delta_p/2, delta_z/2)
                        mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)
                        mass_weight = frf.mass_gaussian_weightings(gaussian_mass_variable, test_mass, delta_mass)
                        conc_weight = frf.conc_gaussian_weightings(gaussian_conc_variable, test_conc, delta_conc)
                                    
                        weight = proximity_weight * mag_weight * mass_weight * conc_weight
                                    
                        #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)
                                    
                        prediction_list_artifact.append(grad_corrected_prediction)
                        sd_list_artifact.append(estimate_predictions['smooth-or-featured-dr5_artifact_var'].values[0])
                        weight_list_artifact.append(weight)
                        
                    #3 mean predictions
                    mean_prediction_smooth = np.mean(prediction_list_smooth)
                    mean_prediction_featured = np.mean(prediction_list_featured)
                    mean_prediction_artifact = np.mean(prediction_list_artifact)
                    
                    mean_std_smooth = np.std(prediction_list_smooth)
                    mean_std_featured = np.std(prediction_list_featured)
                    mean_std_artifact = np.std(prediction_list_artifact)
                    
                    #smooth
                    weighted_mean_numerator_smooth = np.sum(np.array(weight_list_smooth) * np.array(prediction_list_smooth))
                    weighted_mean_denominator_smooth = np.sum(np.array(weight_list_smooth))
                    weighted_mean_smooth = weighted_mean_numerator_smooth/weighted_mean_denominator_smooth #the weighted mean linear prediction using all sub set galaxies
                    
                    #featured
                    weighted_mean_numerator_featured = np.sum(np.array(weight_list_featured) * np.array(prediction_list_featured))
                    weighted_mean_denominator_featured = np.sum(np.array(weight_list_featured))
                    weighted_mean_featured = weighted_mean_numerator_featured/weighted_mean_denominator_featured #the weighted mean linear prediction using all sub set galaxies
                
                    #artifact
                    weighted_mean_numerator_artifact = np.sum(np.array(weight_list_artifact) * np.array(prediction_list_artifact))
                    weighted_mean_denominator_artifact = np.sum(np.array(weight_list_artifact))
                    weighted_mean_artifact = weighted_mean_numerator_artifact/weighted_mean_denominator_artifact #the weighted mean linear prediction using all sub set galaxies
                
                    #store all the weighted means for each morphology as list
                    weighted_means_list_smooth.append(weighted_mean_smooth)
                    weighted_means_list_featured.append(weighted_mean_featured)
                    weighted_means_list_artifact.append(weighted_mean_artifact)
                        
                    #add 1 to galaxy counter for print statement at start of loop
                    test_gal_number+=1
                    
                    sum_of_morph_predictions = np.asarray(weighted_means_list_featured) + np.asarray(weighted_means_list_smooth) + np.asarray(weighted_means_list_artifact)
                    
                    weighted_means_list_smooth_norm = weighted_means_list_smooth / sum_of_morph_predictions
                    weighted_means_list_artifact_norm = weighted_means_list_artifact / sum_of_morph_predictions
                    weighted_means_list_featured_norm = weighted_means_list_featured / sum_of_morph_predictions
                    
                    for i in range(len(weighted_means_list_smooth)):
                        if weighted_means_list_smooth_norm[i]>=cut_threshold:
                            count_smooth.append("smooth")
                        elif weighted_means_list_featured_norm[i]>=cut_threshold:
                            count_featured.append("featured")
                        elif weighted_means_list_artifact_norm[i]>=cut_threshold:
                            count_artifact.append("artifact")
                            
            confident_smooth = len(count_smooth) 
            confident_featured = len(count_featured)
            confident_artifact = len(count_artifact)
            total = confident_smooth + confident_featured + confident_artifact
            df_redshift = pd.DataFrame([[pred_redshift, confident_smooth/total, confident_featured/total, confident_artifact/total]], columns=df.columns)
            df = pd.concat([df,df_redshift])
            
        frf.error_bar_smoothness_3(df['redshift'].values, df['smooth'].values, df['featured'].values, df['artifact'].values, save_name=figure_save_names[i], title='Galaxy Morphology with Redshift for Magnitudes {0}'.format(figure_mag_lims[i]), xlabel='Redshift', ylabel='Proportion of expected predictions', ylimits=[0, 1], xlimits=[0, 0.12])
        j += 1
        print("new cut")
        
    print('\n end')