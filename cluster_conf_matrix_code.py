#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

    args = parser.parse_args()
    
    delta_z = args.delta_z #sets width of sample box - Default optimised = 0.008
    delta_p = args.delta_p #sets height of smaple box - Default optimised = 0.016
    delta_mag = args.delta_mag #Vary to find better base value - Default optimised = 0.5
    
    full_data_array_first_cut = pd.read_csv('full_data.csv', index_col=0)
    full_data_array_first_cut = full_data_array_first_cut.to_numpy()
    
    logging.info('Extracting test sample')
    #Remove the test sample
    test_sample_names = full_data_array_first_cut[args.min_gal:args.max_gal, 0] #define the test galaxies we want to use

    full_dataframe = pd.DataFrame(full_data_array_first_cut) #convert numpy arrays to dataframes
    test_sample = pd.DataFrame(columns=full_dataframe.columns) #form new array of correct column length and labels

    for name in test_sample_names: #iterate over each galaxy in test sample
        cond = full_dataframe[0] == name #identify each galaxy by classification name
        rows = full_dataframe.loc[cond, :]
        test_sample = test_sample.append(rows ,ignore_index=True) #append correct galaxy rows to blank array
        full_dataframe.drop(rows.index, inplace=True) #reset indexing

    logging.info('Beginning predictions')
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
        
        test_galaxy = test_sample[test_sample[0] == test_name] #selects only the current test galaxy from dataframe
        gal_max_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmax()]] #finds the maximum redshfit value for all the galaxies simulations
        gal_min_z = test_galaxy.loc[[test_galaxy[4].astype(float).idxmin()]] #finds the minimum redshift value for all the galaxies simulations
        test_z = gal_max_z[4].astype(float).to_numpy()[0] #converts max z to useable float format
        #test_p = gal_max_z[1].astype(float).to_numpy()[0] #converts prob at max z to useable float format
        pred_z = args.pred_z
        #actual_p = gal_min_z[1].astype(float).to_numpy()[0] #converts the prob at min z to useable format - corrosponds to non-simulated prediction
        test_mag = gal_max_z[5].astype(float).to_numpy()[0] #finds the magnitude of the galaxy (unchanged in simulation)
        
        #identify the 3 prob variables for the simulated image
        test_p_smooth = gal_max_z[1].astype(float).to_numpy()[0] #converts prob at max z to useable float format
        test_p_featured = gal_max_z[2].astype(float).to_numpy()[0] #converts prob at max z to useable float format
        test_p_artifact = gal_max_z[3].astype(float).to_numpy()[0] #converts prob at max z to useable float format

        #identify the 3 prob variables for non-simulated image for comparison 
        actual_p_smooth = gal_min_z[1].astype(float).to_numpy()[0] #prob of smooth
        actual_p_featured = gal_min_z[2].astype(float).to_numpy()[0] #prob of featured
        actual_p_artifact = gal_min_z[3].astype(float).to_numpy()[0] #prob of artifact
        
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

        #sub sample for each morphology
        immediate_sub_sample_smooth = full_dataframe[(full_dataframe[4].astype(float) < upper_z) & (full_dataframe[4].astype(float) >= lower_z) & (full_dataframe[1].astype(float) >= lower_p_smooth) & (full_dataframe[1].astype(float) <= upper_p_smooth)] #samples galaxies within box limits
        immediate_sub_sample_featured = full_dataframe[(full_dataframe[4].astype(float) < upper_z) & (full_dataframe[4].astype(float) >= lower_z) & (full_dataframe[2].astype(float) >= lower_p_featured) & (full_dataframe[2].astype(float) <= upper_p_featured)] #samples galaxies within box limits
        immediate_sub_sample_artifact = full_dataframe[(full_dataframe[4].astype(float) < upper_z) & (full_dataframe[4].astype(float) >= lower_z) & (full_dataframe[3].astype(float) >= lower_p_artifact) & (full_dataframe[3].astype(float) <= upper_p_artifact)] #samples galaxies within box limits
        
        if len(immediate_sub_sample_smooth)==0:
            skipped_gal+=1
            continue
        elif len(immediate_sub_sample_featured)==0:
            skipped_gal+=1
            continue
        elif len(immediate_sub_sample_artifact)==0:
            skipped_gal+=1
            continue
        
        #unique names for each sub sample
        unique_names_smooth = pd.unique(immediate_sub_sample_smooth[0]) #names of subset galaxies for smooth
        unique_names_featured = pd.unique(immediate_sub_sample_featured[0]) #names of subset galaxies for featured
        unique_names_artifact = pd.unique(immediate_sub_sample_artifact[0]) #names of subset galaxies for artifact

        sim_sub_set_smooth = pd.DataFrame() #define empty dataframe
        sim_sub_set_var_smooth = pd.DataFrame() #define empty dataframe
        sim_sub_set_featured = pd.DataFrame() #define empty dataframe
        sim_sub_set_var_featured = pd.DataFrame() #define empty dataframe
        sim_sub_set_artifact = pd.DataFrame() #define empty dataframe
        sim_sub_set_var_artifact = pd.DataFrame() #define empty dataframe
        
        for name in unique_names_smooth:
            sim_sub_set_smooth = sim_sub_set_smooth.append(full_dataframe[full_dataframe[0] == name]) #appends nearby galaxies to dataframe
        
        for name in unique_names_featured:
            sim_sub_set_featured = sim_sub_set_featured.append(full_dataframe[full_dataframe[0] == name]) #appends nearby galaxies to dataframe

        for name in unique_names_artifact:
            sim_sub_set_artifact = sim_sub_set_artifact.append(full_dataframe[full_dataframe[0] == name]) #appends nearby galaxies to dataframe
        
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
        for name in unique_names_smooth:
            galaxy_data = sim_sub_set_smooth[sim_sub_set_smooth[0] == name] #selects each galaxy iteratively from the larger sample
            galaxy_data_var = sim_sub_set_var_smooth[sim_sub_set_var_smooth[0] == name]

            abs_diff_pred_z = abs(galaxy_data[4].astype(float) - pred_z) #finds the difference between each simulated image z and the target z
            min_pos_pred = abs_diff_pred_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            next_min_pos_pred = abs_diff_pred_z.nsmallest(2).index[1]

            abs_diff_test_z = abs(galaxy_data[4].astype(float) - test_z)
            min_pos_test = abs_diff_test_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            nextmin_pos_test = abs_diff_test_z.nsmallest(2).index[1]

            estimate_predictions = galaxy_data.loc[[min_pos_pred]]
            estimate_predictions_var = galaxy_data_var.loc[[min_pos_pred]]
            grad_reference = galaxy_data.loc[[next_min_pos_pred]]
            
            #smooth
            diff_y = estimate_predictions[1].astype(float).to_numpy()[0] - grad_reference[1].astype(float).to_numpy()[0]
            diff_x = estimate_predictions[4].astype(float).to_numpy()[0] - grad_reference[4].astype(float).to_numpy()[0] #the astype and to numpy are to extract numbers from dataframe
            gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value
            
            minimum_point_seperation = pred_z - estimate_predictions[4].astype(float).to_numpy()[0]
            
            #the 3 corrections to gradient
            grad_correction = gradient * minimum_point_seperation
            
            #post grad correction
            grad_corrected_prediction = estimate_predictions[1].astype(float).to_numpy()[0] + grad_correction
        
            closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
            
            #prob varaibles for morphology
            gaussain_p_variable = closest_vals[1].astype(float).to_numpy()[0]
            gaussian_z_variable = closest_vals[4].astype(float).to_numpy()[0]
            gaussian_mag_variable = closest_vals[5].astype(float).to_numpy()[0]
            
            #proximity weights (weightings related to distance from max z sim point in z-p space)
            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_smooth, test_z, delta_p/2, delta_z/2) #

            mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)

            weight = proximity_weight * mag_weight

            #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)

            prediction_list_smooth.append(grad_corrected_prediction) #prediction list is the prediction for each sample galaxy
            sd_list_smooth.append(estimate_predictions_var[2].astype(float).to_numpy()[0])
            weight_list_smooth.append(weight)
            
        #for loop for the featured gradient predictions
        for name in unique_names_featured:
            galaxy_data = sim_sub_set_featured[sim_sub_set_featured[0] == name] #selects each galaxy iteratively from the larger sample
            galaxy_data_var = sim_sub_set_var_featured[sim_sub_set_var_featured[0] == name]

            abs_diff_pred_z = abs(galaxy_data[4].astype(float) - pred_z) #finds the difference between each simulated image z and the target z
            min_pos_pred = abs_diff_pred_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            next_min_pos_pred = abs_diff_pred_z.nsmallest(2).index[1]

            abs_diff_test_z = abs(galaxy_data[4].astype(float) - test_z)
            min_pos_test = abs_diff_test_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            nextmin_pos_test = abs_diff_test_z.nsmallest(2).index[1]

            estimate_predictions = galaxy_data.loc[[min_pos_pred]]
            estimate_predictions_var = galaxy_data_var.loc[[min_pos_pred]]
            grad_reference = galaxy_data.loc[[next_min_pos_pred]]
            
            #featured
            diff_y = estimate_predictions[2].astype(float).to_numpy()[0] - grad_reference[2].astype(float).to_numpy()[0]
            diff_x = estimate_predictions[4].astype(float).to_numpy()[0] - grad_reference[4].astype(float).to_numpy()[0] #the astype and to numpy are to extract numbers from dataframe
            gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value
            
            minimum_point_seperation = pred_z - estimate_predictions[4].astype(float).to_numpy()[0]
            
            #the 3 corrections to gradient
            grad_correction = gradient * minimum_point_seperation
            
            #post grad correction
            grad_corrected_prediction = estimate_predictions[2].astype(float).to_numpy()[0] + grad_correction
        
            closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
            
            #prob varaibles for morphology
            gaussain_p_variable = closest_vals[2].astype(float).to_numpy()[0]
            gaussian_z_variable = closest_vals[4].astype(float).to_numpy()[0]
            gaussian_mag_variable = closest_vals[5].astype(float).to_numpy()[0]
            
            #proximity weights (weightings related to distance from max z sim point in z-p space)
            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_featured, test_z, delta_p/2, delta_z/2) #

            mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)

            weight = proximity_weight * mag_weight

            #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)

            prediction_list_featured.append(grad_corrected_prediction) #prediction list is the prediction for each sample galaxy
            sd_list_featured.append(estimate_predictions_var[2].astype(float).to_numpy()[0])
            weight_list_featured.append(weight)
            
        #for loop for the artifact gradient predictions
        for name in unique_names_artifact:
            galaxy_data = sim_sub_set_artifact[sim_sub_set_artifact[0] == name] #selects each galaxy iteratively from the larger sample
            galaxy_data_var = sim_sub_set_var_artifact[sim_sub_set_var_artifact[0] == name]

            abs_diff_pred_z = abs(galaxy_data[4].astype(float) - pred_z) #finds the difference between each simulated image z and the target z
            min_pos_pred = abs_diff_pred_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            next_min_pos_pred = abs_diff_pred_z.nsmallest(2).index[1]

            abs_diff_test_z = abs(galaxy_data[4].astype(float) - test_z)
            min_pos_test = abs_diff_test_z.nsmallest(2).index[0] #nsmallest pickest the n smallest values and puts them in df
            nextmin_pos_test = abs_diff_test_z.nsmallest(2).index[1]

            estimate_predictions = galaxy_data.loc[[min_pos_pred]]
            estimate_predictions_var = galaxy_data_var.loc[[min_pos_pred]]
            grad_reference = galaxy_data.loc[[next_min_pos_pred]]
            
            #artifact
            diff_y = estimate_predictions[3].astype(float).to_numpy()[0] - grad_reference[3].astype(float).to_numpy()[0]
            diff_x = estimate_predictions[4].astype(float).to_numpy()[0] - grad_reference[4].astype(float).to_numpy()[0] #the astype and to numpy are to extract numbers from dataframe
            gradient = diff_y / diff_x #Finding the gradient between the two points closest to the test value
            
            minimum_point_seperation = pred_z - estimate_predictions[4].astype(float).to_numpy()[0]
            
            #the 3 corrections to gradient
            grad_correction = gradient * minimum_point_seperation
            
            #post grad correction
            grad_corrected_prediction = estimate_predictions[3].astype(float).to_numpy()[0] + grad_correction
        
            closest_vals = galaxy_data.loc[[min_pos_test]] #Possible could encounter edge case issues
            
            #prob varaibles for morphology
            gaussain_p_variable = closest_vals[3].astype(float).to_numpy()[0]
            
            gaussian_z_variable = closest_vals[4].astype(float).to_numpy()[0]
            gaussian_mag_variable = closest_vals[5].astype(float).to_numpy()[0]
            
            #proximity weights (weightings related to distance from max z sim point in z-p space)
            proximity_weight = frf.gaussian_weightings(gaussain_p_variable, gaussian_z_variable, test_p_artifact, test_z, delta_p/2, delta_z/2) #

            mag_weight = frf.gaussian_weightings(gaussian_mag_variable, 0, test_mag, 0, delta_mag, 1)

            weight = proximity_weight * mag_weight

            #print('mag_weight is:', mag_weight, '\nprox_wieght is:', proximity_weight, '\nTotal Weight is:', weight)

            prediction_list_artifact.append(grad_corrected_prediction) #prediction list is the prediction for each sample galaxy
            sd_list_artifact.append(estimate_predictions_var[2].astype(float).to_numpy()[0])
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
    
    morphology_names = ['smooth', 'featured', 'artifact', 'NULL'] #add NULL to morphology names
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
            dominant_morphology_expected.append("NULL") 
    
    dominant_morphology_predicted = []
    for i in range(len(weighted_means_list_smooth)):
        if weighted_means_list_smooth_norm[i]>=threshold_p:
            dominant_morphology_predicted.append("smooth")
        elif weighted_means_list_featured_norm[i]>=threshold_p:
            dominant_morphology_predicted.append("featured")
        elif weighted_means_list_artifact_norm[i]>=threshold_p:
            dominant_morphology_predicted.append("artifact")
        else:
            dominant_morphology_predicted.append("NULL")

    dominant_morphology_simulated = []
    for i in range(len(test_p_list_smooth)):
        if test_p_list_smooth[i]>=threshold_p:
            dominant_morphology_simulated.append("smooth")
        elif test_p_list_featured[i]>=threshold_p:
            dominant_morphology_simulated.append("featured")
        elif test_p_list_artifact[i]>=threshold_p:
            dominant_morphology_simulated.append("artifact") 
        else:
            dominant_morphology_simulated.append("NULL")  
    
    confident_locs = np.argwhere(np.asarray(dominant_morphology_predicted)!='NULL') #find arguements for confident debiased predictions
    logging.info(confident_locs)
    if len(confident_locs) == 0:
        raise ValueError(f'No confident (i.e. non-NULL i.e. with predicted debiased p > {threshold_p} found - cannot make CMs or metrics')
    
    dominant_morphology_expected = np.asarray(dominant_morphology_expected)[confident_locs] #remove non-confident debiased predictions from expected list
    dominant_morphology_predicted = np.asarray(dominant_morphology_predicted)[confident_locs] #remove non-confident debiased predictions from predicted list
    dominant_morphology_simulated =np.asarray(dominant_morphology_simulated)[confident_locs] #remove non-confident debiased predictions from simulated list
    
    #Creating the confusion matrices
    expected = dominant_morphology_expected #list of the true values in order of prediction - rounded actual_p vals to nearest 0.5
    predicted = dominant_morphology_predicted #list of predicted vals in order of prediction
    simulated = dominant_morphology_simulated
    
    logging.info(expected)
    logging.info(predicted)
    results = confusion_matrix(expected, predicted, labels=morphology_names) #converting inputs into confusion matrix format (debiased on x-axis (top) and true on y-axis (side))
    results = results[0:4, 0:4] #(remove the NULL column for debiased prediction as it will never be filled) changed to include to avoid confusion about the diagonal - mike request
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
    kappa_results = cohen_kappa_score(expected, predicted, labels=['smooth', 'featured', 'artifact', 'NULL'])
    kappa_comp = cohen_kappa_score(expected, simulated, labels=['smooth', 'featured', 'artifact', 'NULL'])

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
    
    #sb.heatmap(dataframe_results, annot=True, cmap=sb.color_palette("YlOrBr_r", as_cmap=True)) #plotting heatmap for use as confusion matrix plot (annot shows the number within the box)
    #sb.cubehelix_palette(start=.5, rot=-.75, reverse=True, as_cmap=True)
    
    plt.title('Comparative predictions with de-biasing method (N={0} with {1} above p={2})'.format(len(test_sample_names) - skipped_gal, len(predicted), threshold_p), fontsize=20, wrap=True)
    plt.xlabel('De-biased high redshift prediction (Predicted)', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Non-simulated (low redshift) prediction (Actual)', fontsize = 15) # y-axis label with fontsize 15
    plt.savefig('De_biased_predictions_confusion_matrix.png')
    plt.close()
    
    
    #plot the non de-biased matrix
    plot_2 = plt.figure(figsize=(10,6))
   
    sb.heatmap(dataframe_comparative_results, annot=True, fmt='d', annot_kws={'fontsize':10}, cmap=sb.cubehelix_palette(as_cmap=True, reverse=True)) #plotting heatmap for use as confusion matrix plot (annot shows the number within the box)
    
    plt.title('Comparative predictions with non de-biasing method (N={0:} with {1} above p={2})'.format(len(test_sample_names) - skipped_gal, len(simulated), threshold_p), fontsize=20, wrap=True)
    plt.xlabel('High redshift prediction (Prediction)', fontsize = 15) # x-axis label with fontsize 15
    plt.ylabel('Non-simulated (low redshift) prediction (Actual)', fontsize = 15) # y-axis label with fontsize 15
    plt.savefig('Non_de_biased_predictions_confusion_matrix.png')
    plt.close()

    logging.info('Confusion matrix plots complete - existing')
    
