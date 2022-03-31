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
import glob
import os
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
    parser.add_argument('--morphology-dir', dest='morphology_dir', type=str)


    args = parser.parse_args()
    
    morphology_dir = args.morphology_dir
    
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
    debiased_df_chunks = [frf.file_reader(loc) for loc in glob.glob(os.path.join(morphology_dir, '*.csv'))]
    merged_dataframe = pd.concat(debiased_df_chunks, axis=0).reset_index(drop=True)

    dominant_morphology_expected = []
    for i in range(len(merged_dataframe['actual_smooth_pred'])):
        if merged_dataframe['actual_smooth_pred'][i]>=threshold_p:
            dominant_morphology_expected.append("smooth")
        elif merged_dataframe['actual_featured_pred'][i]>=threshold_p:
            dominant_morphology_expected.append("featured")
        elif merged_dataframe['actual_artifact_pred'][i]>=threshold_p:
            dominant_morphology_expected.append("artifact")
        else:
            dominant_morphology_expected.append("unclassified") 
    
    dominant_morphology_predicted = []
    for i in range(len(merged_dataframe['actual_smooth_pred'])):
        if merged_dataframe['debiased_smooth_pred'][i]>=threshold_p:
            dominant_morphology_predicted.append("smooth")
        elif merged_dataframe['debiased_featured_pred'][i]>=threshold_p:
            dominant_morphology_predicted.append("featured")
        elif merged_dataframe['debiased_artifact_pred'][i]>=threshold_p:
            dominant_morphology_predicted.append("artifact")
        else:
            dominant_morphology_predicted.append("unclassified")

    dominant_morphology_simulated = []
    for i in range(len(merged_dataframe['actual_smooth_pred'])):
        if merged_dataframe['sim_smooth_pred'][i]>=threshold_p:
            dominant_morphology_simulated.append("smooth")
        elif merged_dataframe['sim_featured_pred'][i]>=threshold_p:
            dominant_morphology_simulated.append("featured")
        elif merged_dataframe['sim_artifact_pred'][i]>=threshold_p:
            dominant_morphology_simulated.append("artifact") 
        else:
            dominant_morphology_simulated.append("unclassified")  
    
    confident_locs_smooth = np.argwhere(np.asarray(merged_dataframe['debiased_smooth_pred']) > 0.7) #find arguements for confident debiased predictions (predictions > 0.7)
    confident_locs_featured = np.argwhere(np.asarray(merged_dataframe['debiased_featured_pred']) > 0.7) #find arguements for confident debiased predictions (predictions > 0.7)
    confident_locs_artifact = np.argwhere(np.asarray(merged_dataframe['debiased_artifact_pred']) > 0.7) #find arguements for confident debiased predictions (predictions > 0.7)

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
    
    plt.title('Comparative predictions with de-biasing method (N={0} with {1} above p={2})'.format(len(test_sample_names) - skipped_gal, len(predicted), 0.7), fontsize=20, wrap=True)
    plt.xlabel('De-biased high redshift prediction (Predicted)', fontsize = 16) # x-axis label with fontsize 15
    plt.ylabel('Non-simulated (low redshift) prediction (Actual)', fontsize = 16) # y-axis label with fontsize 15
    plt.savefig('matrix_plots/De_biased_predictions_confusion_matrix.jpeg')
    plt.close()
    
    
    #plot the non de-biased matrix
    plot_2 = plt.figure(figsize=(10,6))
   
    sb.heatmap(dataframe_comparative_results, annot=True, fmt='d', annot_kws={'fontsize':14}, cmap=sb.cubehelix_palette(as_cmap=True, reverse=True)) #plotting heatmap for use as confusion matrix plot (annot shows the number within the box)
    sb.set(font_scale=1.5)

    plt.title('Comparative predictions with non de-biasing method (N={0:} with {1} above p={2})'.format(len(test_sample_names) - skipped_gal, len(simulated), 0.7), fontsize=20, wrap=True)
    plt.xlabel('High redshift prediction (Prediction)', fontsize = 16) # x-axis label with fontsize 15
    plt.ylabel('Non-simulated (low redshift) prediction (Actual)', fontsize = 16) # y-axis label with fontsize 15
    plt.savefig('matrix_plots/Non_de_biased_predictions_confusion_matrix.jpeg')
    plt.close()

    logging.info('Confusion matrix plots complete - exiting')