"""
Created on Tue Mar  8 15:09:08 2022
@author: adamfoster
"""
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import argparse
#frf for file opening and plotting functions
import functions_for_redshifting_figures as frf


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--min-gal', dest='min_gal', type=int)
    parser.add_argument('--max-gal', dest='max_gal', type=int)
    parser.add_argument('--update-interval', dest='update_interval', type=int)
    parser.add_argument('--pred-z', dest='pred_z', type=float)
    parser.add_argument('--threshold-val', dest='threshold_val', type=float)
    parser.add_argument('--delta-z', dest='delta_z', type=float)
    parser.add_argument('--delta-p', dest='delta_p', type=float)
    parser.add_argument('--delta-mag', dest='delta_mag', type=float)
    parser.add_argument('--delta-mass', dest='delta_mass', type=float)
    parser.add_argument('--delta-conc', dest='delta_conc', type=float)

    args = parser.parse_args()
    
    min_gal = args.min_gal
    max_gal = args.max_gal
    
    delta_z = args.delta_z #sets width of sample box - Default optimised = 0.008
    delta_p = args.delta_p #sets height of smaple box - Default optimised = 0.016
    delta_mag = args.delta_mag #Vary to find better base value - Default optimised = 0.5
    delta_mass = args.delta_mass #Vary to find better base value - Default optimised = 0.5
    delta_conc = args.delta_conc #Vary to find better base value - Default optimised = 0.5
    
    logging.info('CONFUSION MATRIX CODE COMMENCING')

    full_data = pd.read_csv('output_csvs/full_data_1m_with_resizing.csv')
    full_data['elpetro_mass'] = np.log10(full_data['elpetro_mass'])
    #full_data = pd.read_csv('output_csvs/full_data.csv')
    assert len(full_data) > 0
    
    logging.info('Extracting test sample')
    # Remove the test sample
    test_sample_names = pd.unique(full_data['iauname'])[min_gal:max_gal] 
    assert len(test_sample_names) > 0

    test_sample = full_data[full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)

    full_data = full_data[~full_data['iauname'].isin(test_sample_names)].reset_index(drop=True)

    logging.info('Beginning predictions on {} galaxies'.format(len(test_sample_names)))
    #If we want to operate over multiple galaxies, start a for loop here
    test_gal_number = 0 #count number of gals which have been processed
    skipped_gal = 0
    
    weighted_means_list_smooth = [] #list for each weighted mean prediction (smooth prediction)
    weighted_means_list_featured = [] #list for each weighted mean prediction (featured prediction)
    weighted_means_list_artifact = [] #list for each weighted mean prediction (artifact prediction)
    
    actual_p_list_smooth = [] #list for each actual_p value (smooth)
    actual_p_list_featured = [] #list for each actual_p value (featured)
    actual_p_list_artifact= [] #list for each actual_p value (artifact)
    
    test_p_list_smooth = [] #list for each test_p value per galaxy (smooth)
    test_p_list_featured = [] #list for each test_p value per galaxy (featured)
    test_p_list_artifact = [] #list for each test_p value per galaxy (artifact)
    
    for test_name in test_sample_names:
        
        if (test_gal_number % args.update_interval == 0):
            logging.info('completed {0} of {1} galaxy debias predictions'.format(test_gal_number, len(test_sample_names))) #prints progress every {number} galaxy debias predictions
        
        test_galaxy = test_sample[test_sample['iauname'] == test_name]
        gal_max_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmax()]]
        gal_min_z = test_galaxy.loc[[test_galaxy['redshift'].astype(float).idxmin()]]
        test_z = gal_max_z['redshift'].values[0]
        pred_z = gal_min_z['redshift'].values[0]
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
        immediate_sub_sample_smooth = full_data[
                            (full_data['redshift'].astype(float) <= upper_z) &
                            (full_data['redshift'].astype(float) >= lower_z) &
                            (full_data['smooth-or-featured-dr5_smooth_prob'].astype(float) >= lower_p_smooth) &
                            (full_data['smooth-or-featured-dr5_smooth_prob'].astype(float) <= upper_p_smooth) &
                            (full_data['elpetro_absmag_r'].astype(float) <= upper_mag) &
                            (full_data['elpetro_absmag_r'].astype(float) >= lower_mag) &
                            (full_data['elpetro_mass'].astype(float) <= upper_mass) &
                            (full_data['elpetro_mass'].astype(float) >= lower_mass) &
                            (full_data['concentration'].astype(float) <= upper_conc) &
                            (full_data['concentration'].astype(float) >= lower_conc)
                            ] #samples galaxies within box limits
        
        immediate_sub_sample_featured = full_data[
                            (full_data['redshift'].astype(float) <= upper_z) &
                            (full_data['redshift'].astype(float) >= lower_z) &
                            (full_data['smooth-or-featured-dr5_featured-or-disk_prob'].astype(float) >= lower_p_featured) &
                            (full_data['smooth-or-featured-dr5_featured-or-disk_prob'].astype(float) <= upper_p_featured) &
                            (full_data['elpetro_absmag_r'].astype(float) <= upper_mag) &
                            (full_data['elpetro_absmag_r'].astype(float) >= lower_mag) &
                            (full_data['elpetro_mass'].astype(float) <= upper_mass) &
                            (full_data['elpetro_mass'].astype(float) >= lower_mass) &
                            (full_data['concentration'].astype(float) <= upper_conc) &
                            (full_data['concentration'].astype(float) >= lower_conc) 
                            ] #samples galaxies within box limits

        immediate_sub_sample_artifact = full_data[
                            (full_data['redshift'].astype(float) <= upper_z) &
                            (full_data['redshift'].astype(float) >= lower_z) &
                            (full_data['smooth-or-featured-dr5_artifact_prob'].astype(float) >= lower_p_artifact) &
                            (full_data['smooth-or-featured-dr5_artifact_prob'].astype(float) <= upper_p_artifact) &
                            (full_data['elpetro_absmag_r'].astype(float) <= upper_mag) &
                            (full_data['elpetro_absmag_r'].astype(float) >= lower_mag) &
                            (full_data['elpetro_mass'].astype(float) <= upper_mass) &
                            (full_data['elpetro_mass'].astype(float) >= lower_mass) &
                            (full_data['concentration'].astype(float) <= upper_conc) &
                            (full_data['concentration'].astype(float) >= lower_conc)
                            ] #samples galaxies within box limits
        
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

        sim_sub_set_smooth = full_data[full_data['iauname'].isin(galaxy_names_in_box_smooth)]
        sim_sub_set_featured = full_data[full_data['iauname'].isin(galaxy_names_in_box_featured)]
        sim_sub_set_artifact = full_data[full_data['iauname'].isin(galaxy_names_in_box_artifact)]
        
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

            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_featured, test_z, delta_p/2, delta_z/2)
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

            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_artifact, test_z, delta_p/2, delta_z/2)
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

        #store all the true p vals for each morphology in list
        actual_p_list_smooth.append(actual_p_smooth)
        actual_p_list_featured.append(actual_p_featured)
        actual_p_list_artifact.append(actual_p_artifact)
        
        #store the test_p vals for plotting as the non de-biased matrice in a list
        test_p_list_smooth.append(test_p_smooth)
        test_p_list_featured.append(test_p_featured)
        test_p_list_artifact.append(test_p_artifact)

        #add 1 to galaxy counter for print statement at start of loop
        test_gal_number+=1
        

    """
    From here we have the weighted_mean as the average prediction per galaxy, which we want for
    each test galaxy operating over. So need in a list.
    
    We also need the true value for each test galaxy, since we are predicting from point of lowest
    redshift, will simply be the lowest redshift input for that galaxy, actual_p.
    
    The prediction redshift given by pred_z, the redshift predicted from is test_z, the initial 
    prediction prob is test_p.
    """
    
    logging.info('{0} predictions complete, {1} galaxies skipped, plotting matrix'.format(test_gal_number, skipped_gal))
    
    morphology_names = ['smooth', 'featured', 'artifact', 'unclassified'] 
    threshold_p = args.threshold_val
    
        #normalising the de-biased predictions
    sum_of_morph_predictions = np.asarray(weighted_means_list_featured) + np.asarray(weighted_means_list_smooth) + np.asarray(weighted_means_list_artifact)
    
    weighted_means_list_smooth_norm = weighted_means_list_smooth / sum_of_morph_predictions
    weighted_means_list_artifact_norm = weighted_means_list_artifact / sum_of_morph_predictions
    weighted_means_list_featured_norm = weighted_means_list_featured / sum_of_morph_predictions

    dominant_morphology_expected = []
    for i in range(len(actual_p_list_smooth)):
        if actual_p_list_smooth[i]>=threshold_p:
            dominant_morphology_expected.append("smooth")
        elif actual_p_list_featured[i]>=threshold_p:
            dominant_morphology_expected.append("featured")
        elif actual_p_list_artifact[i]>=threshold_p:
            dominant_morphology_expected.append("artifact")
        else:
            dominant_morphology_expected.append("unclassified") 
    
    dominant_morphology_predicted = []
    for i in range(len(weighted_means_list_smooth)):
        if weighted_means_list_smooth_norm[i]>=threshold_p:
            dominant_morphology_predicted.append("smooth")
        elif weighted_means_list_featured_norm[i]>=threshold_p:
            dominant_morphology_predicted.append("featured")
        elif weighted_means_list_artifact_norm[i]>=threshold_p:
            dominant_morphology_predicted.append("artifact")
        else:
            dominant_morphology_predicted.append("unclassified")

    dominant_morphology_simulated = []
    for i in range(len(test_p_list_smooth)):
        if test_p_list_smooth[i]>=threshold_p:
            dominant_morphology_simulated.append("smooth")
        elif test_p_list_featured[i]>=threshold_p:
            dominant_morphology_simulated.append("featured")
        elif test_p_list_artifact[i]>=threshold_p:
            dominant_morphology_simulated.append("artifact") 
        else:
            dominant_morphology_simulated.append("unclassified")  
    
    confident_locs_smooth = np.argwhere(np.asarray(weighted_means_list_smooth_norm) > 0.7) #find arguements for confident debiased predictions (predictions > 0.7)
    confident_locs_featured = np.argwhere(np.asarray(weighted_means_list_featured_norm) > 0.7) #find arguements for confident debiased predictions (predictions > 0.7)
    confident_locs_artifact = np.argwhere(np.asarray(weighted_means_list_artifact_norm) > 0.7) #find arguements for confident debiased predictions (predictions > 0.7)

    confident_locs=np.vstack((confident_locs_smooth, confident_locs_featured))
    confident_locs=np.vstack((confident_locs, confident_locs_artifact))

    logging.info(confident_locs)
    if len(confident_locs) == 0:
        raise ValueError(f'No confident (i.e. non-unclassified i.e. with predicted debiased p > {threshold_p} found - cannot make CMs or metrics')
    
    dominant_morphology_expected = np.asarray(dominant_morphology_expected)[confident_locs] #remove non-confident debiased predictions from expected list
    dominant_morphology_predicted = np.asarray(dominant_morphology_predicted)[confident_locs] #remove non-confident debiased predictions from predicted list
    dominant_morphology_simulated =np.asarray(dominant_morphology_simulated)[confident_locs] #remove non-confident debiased predictions from simulated list
    
    #Creating the confusion matrices
    expected = dominant_morphology_expected #list of the true values in order of prediction - rounded actual_p vals to nearest 0.5
    predicted = dominant_morphology_predicted #list of predicted vals in order of prediction
    simulated = dominant_morphology_simulated
    
    results = confusion_matrix(expected, predicted, labels=morphology_names) #converting inputs into confusion matrix format (debiased on x-axis (top) and true on y-axis (side))
    results = results[0:4, 0:4] #(remove the unclassified column for debiased prediction as it will never be filled) changed to include to avoid confusion about the diagonal - mike request
    comparison_results = confusion_matrix(expected, simulated, labels=morphology_names) #converting inputs into confusion matrix format (debiased on x-axis (top) and true on y-axis (side))
    
    #find the precision, recall and f1 scores
    precision_results = precision_score(expected, predicted, labels=['smooth', 'featured', 'artifact'], average=None, zero_division=1)
    recall_results = recall_score(expected, predicted, labels=['smooth', 'featured', 'artifact'], average=None, zero_division=1)
    f1_results = f1_score(expected, predicted, labels=['smooth', 'featured', 'artifact'], average=None, zero_division=1)
    
    precision_comp = precision_score(expected, simulated, labels=['smooth', 'featured', 'artifact'], average=None, zero_division=1)
    recall_comp = recall_score(expected, simulated, labels=['smooth', 'featured', 'artifact'], average=None, zero_division=1)
    f1_comp = f1_score(expected, simulated, labels=['smooth', 'featured', 'artifact'], average=None, zero_division=1)
    
    #calculate normal accuracies
    accuracy_results = np.trace(results) / np.sum(results)
    accuracy_comp = np.trace(comparison_results) / np.sum(comparison_results)
    
    #calculate kappa accuracies
    kappa_results = cohen_kappa_score(expected, predicted, labels=['smooth', 'featured', 'artifact', 'unclassified'])
    kappa_comp = cohen_kappa_score(expected, simulated, labels=['smooth', 'featured', 'artifact', 'unclassified'])

    #table the precision, rcall, f1 and accuracy
    table_of_scores_results = pd.DataFrame((precision_results, recall_results, f1_results)).T
    table_of_scores_comp = pd.DataFrame((precision_comp, recall_comp, f1_comp)).T
    #rename table headers and indicies
    table_of_scores_results = table_of_scores_results.rename(columns={0:'Precision', 1:'Recall', 2:'F1-Score'}, index={0:'Smooth', 1:'Featured', 2:'Artifact'})
    table_of_scores_comp = table_of_scores_comp.rename(columns={0:'Precision', 1:'Recall', 2:'F1-Score'}, index={0:'Smooth', 1:'Featured', 2:'Artifact'})
    
    #rename matrix entries
    dataframe_results = pd.DataFrame(results) #converting confuson matrix format in dataframe for seaborn
    dataframe_comparative_results = pd.DataFrame(comparison_results)
    
    for i in dataframe_results.index:
        dataframe_results = dataframe_results.rename(columns={dataframe_results.index[i]:morphology_names[i]}) #renaming all the column headers to correct ranges (round to remove additional decimal place errors)
        dataframe_results = dataframe_results.rename(index={dataframe_results.index[i]:morphology_names[i]}) #renaming all the row headers to correct ranges (round to remove additional decimal place errors)
        
    for i in dataframe_comparative_results.index:
        dataframe_comparative_results = dataframe_comparative_results.rename(columns={dataframe_comparative_results.index[i]:morphology_names[i]}) #renaming all the column headers to correct ranges (round to remove additional decimal place errors)
        dataframe_comparative_results = dataframe_comparative_results.rename(index={dataframe_comparative_results.index[i]:morphology_names[i]}) #renaming all the row headers to correct ranges (round to remove additional decimal place errors)

    print('De-biased predction metrics:\n\n', table_of_scores_results, '\nAccuarcy:', accuracy_results, '\nKappa:', kappa_results) #print for the debiased precision, recall and f1
    print('\nHigh redshift predction metrics:\n\n', table_of_scores_comp, '\nAccuarcy:', accuracy_comp, '\nKappa:', kappa_comp) #print for the simulated precision, recall and f1

    #plot the de-biased matrix
    plot_1 = plt.figure(figsize=(10,6))
   
    sb.heatmap(dataframe_results, annot=True, fmt='d', annot_kws={'fontsize':10}, cmap=sb.cubehelix_palette(as_cmap=True, reverse=True)) #plotting heatmap for use as confusion matrix plot (annot shows the number within the box)
    sb.set(font_scale=1.5)

    #sb.heatmap(dataframe_results, annot=True, cmap=sb.color_palette("YlOrBr_r", as_cmap=True)) #plotting heatmap for use as confusion matrix plot (annot shows the number within the box)
    #sb.cubehelix_palette(start=.5, rot=-.75, reverse=True, as_cmap=True)
    
    plt.title('Comparative predictions with de-biasing method (N={0} with {1} above p={2})'.format(len(test_sample_names) - skipped_gal, len(predicted), threshold_p), fontsize=20, wrap=True)
    plt.xlabel('De-biased high redshift prediction (Predicted)', fontsize = 16) # x-axis label with fontsize 15
    plt.ylabel('Non-simulated (low redshift) prediction (Actual)', fontsize = 16) # y-axis label with fontsize 15
    plt.savefig('matrix_plots/De_biased_predictions_confusion_matrix.jpeg')
    plt.close()
    
    
    #plot the non de-biased matrix
    plot_2 = plt.figure(figsize=(10,6))
   
    sb.heatmap(dataframe_comparative_results, annot=True, fmt='d', annot_kws={'fontsize':14}, cmap=sb.cubehelix_palette(as_cmap=True, reverse=True)) #plotting heatmap for use as confusion matrix plot (annot shows the number within the box)
    sb.set(font_scale=1.5)

    plt.title('Comparative predictions with non de-biasing method (N={0:} with {1} above p={2})'.format(len(test_sample_names) - skipped_gal, len(simulated), threshold_p), fontsize=20, wrap=True)
    plt.xlabel('High redshift prediction (Prediction)', fontsize = 16) # x-axis label with fontsize 15
    plt.ylabel('Non-simulated (low redshift) prediction (Actual)', fontsize = 16) # y-axis label with fontsize 15
    plt.savefig('matrix_plots/Non_de_biased_predictions_confusion_matrix.jpeg')
    plt.close()

    logging.info('Confusion matrix plots complete - exiting')