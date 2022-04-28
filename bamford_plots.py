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
import glob
import os

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold-val', dest='threshold_val', type=float)
    parser.add_argument('--rounding', dest='rounding', type=float)
    parser.add_argument('--predictions-dir', dest='predictions_dir', type=str)
    
    args = parser.parse_args()
    
    logging.info('MORPHOLOGY WITH REDSHIFT CODE COMMENCING')
    
    predictions_dir = args.predictions_dir
    
    full_data = pd.read_csv('output_csvs/full_data_1m_with_resizing.csv') #index_col=0
    full_data = full_data.sort_values(by = 'redshift')
    full_data['elpetro_mass'] = np.log10(full_data['elpetro_mass'])
    
    rounding = args.rounding
    cut_threshold = args.threshold_val
    
    full_data_array_first_cut=np.zeros((0, 9))
    full_data_array_second_cut=np.zeros((0, 9))
    full_data_array_third_cut=np.zeros((0, 9))
    proportions_by_redshift_by_cut = []
    
    first_mag_cut = full_data[(full_data["elpetro_absmag_r"] < -18 ) & (full_data["elpetro_absmag_r"] >= -20)]
    first_mag_cut['redshift'] = np.round(first_mag_cut["redshift"] * (1/rounding))/(1/rounding)
    second_mag_cut = full_data[(full_data["elpetro_absmag_r"] < -20 ) & (full_data["elpetro_absmag_r"] >= -21)]
    second_mag_cut['redshift'] = np.round(second_mag_cut["redshift"] * (1/rounding))/(1/rounding)
    third_mag_cut = full_data[(full_data["elpetro_absmag_r"] < -21 ) & (full_data["elpetro_absmag_r"] >= -24)]
    third_mag_cut['redshift'] = np.round(third_mag_cut["redshift"] * (1/rounding))/(1/rounding)
    
    figure_mag_lims = ['-18 to -20',
                        '-20 to -21',
                        '-21 to -24']
    
    figure_save_names = ['other_plots/smoothness_cut_18_20_{0}_graph_redshift_certain_classification_extended_data_remade.png'.format(cut_threshold), 
                        'other_plots/smoothness_cut_20_21_{0}_graph_redshift_certain_classification_extended_data_remade.png'.format(cut_threshold),
                        'other_plots/smoothness_cut_21_24_{0}_graph_redshift_certain_classification_extended_data_remade.png'.format(cut_threshold)]
    
    i = 0
    for cut in [first_mag_cut, second_mag_cut, third_mag_cut]:
        #full_data_array[:, 4]=np.round(full_data_array[:, 4].astype(float), 2) #rounds the redshift values to 2 dp for binning
    
        cut = cut.sort_values(by = "redshift").reset_index(drop = True)#sorts all data based on ascending redshift
        
        df = pd.DataFrame(columns=['redshift','smooth','featured','artifact'])
        for redshift in np.arange(cut["redshift"].min(), 0.12+rounding, rounding):
            
            redshift = np.round(redshift * (1/rounding))/(1/rounding) #necessary?
            df_temp = cut[cut["redshift"] == redshift]
            confident_smooth = df_temp[(df_temp["smooth-or-featured-dr5_smooth_prob"] > cut_threshold)].count()[0]
            confident_featured = df_temp[(df_temp["smooth-or-featured-dr5_featured-or-disk_prob"] > cut_threshold)].count()[0]
            confident_artifact= df_temp[(df_temp["smooth-or-featured-dr5_artifact_prob"] > cut_threshold)].count()[0]
            total = confident_smooth + confident_featured + confident_artifact
            df_redshift = pd.DataFrame([[redshift, confident_smooth/total, confident_featured/total, confident_artifact/total]], columns=df.columns)
            df = pd.concat([df,df_redshift])
        
        frf.error_bar_smoothness_3(df['redshift'].values, df['smooth'].values, df['featured'].values, df['artifact'].values, save_name=figure_save_names[i], title='Galaxy Morphology with Redshift for Magnitudes {0}'.format(figure_mag_lims[i]), xlabel='Redshift', ylabel='Proportion of expected predictions', ylimits=[0, 1], xlimits=[0, 0.12])
        i+=1
    
    logging.info("Original plots plotted")
    
    figure_save_names = ['other_plots/smoothness_cut_18_20_{0}_graph_redshift_certain_classification_extended_data_remade_debiased.png'.format(cut_threshold), 
                        'other_plots/smoothness_cut_20_21_{0}_graph_redshift_certain_classification_extended_data_remade_debiased.png'.format(cut_threshold),
                        'other_plots/smoothness_cut_21_24_{0}_graph_redshift_certain_classification_extended_data_remade_debiased.png'.format(cut_threshold)]
    
    debiased_df_chunks = [pd.read_csv(loc) for loc in glob.glob(os.path.join(predictions_dir, '*.csv'))]
    debiased_df = pd.concat(debiased_df_chunks, axis=0).reset_index(drop=True)
    
    if os.path.isdir('/share/nas2'):
        catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/catalogs/nsa_v1_0_1_mag_cols.parquet'
    else:
        catalog_loc = '/Users/adamfoster/Documents/MPhysProject/understanding_galaxies/making_figures/nsa_v1_0_1_mag_cols.parquet'
    
    nsa_catalog = pd.read_parquet(catalog_loc, columns=['iauname', 'elpetro_absmag_r'])
    
    merged_dataframe = debiased_df.merge(nsa_catalog, on='iauname', how='inner')
    
    first_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -18 ) & (merged_dataframe["elpetro_absmag_r"] >= -20)]
    first_mag_cut['redshift'] = np.round(first_mag_cut["redshift"] * (1/rounding))/(1/rounding)
    second_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -20 ) & (merged_dataframe["elpetro_absmag_r"] >= -21)]
    second_mag_cut['redshift'] = np.round(second_mag_cut["redshift"] * (1/rounding))/(1/rounding)
    third_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -21 ) & (merged_dataframe["elpetro_absmag_r"] >= -24)]
    third_mag_cut['redshift'] = np.round(third_mag_cut["redshift"] * (1/rounding))/(1/rounding)
    
    j = 0
    for cut in [first_mag_cut, second_mag_cut, third_mag_cut]:
        df = pd.DataFrame(columns=['redshift','smooth','featured','artifact'])
        redshifts = pd.unique(cut['redshift'])
        for pred_redshift in redshifts:
            
            count_smooth = [] #list for each weighted mean prediction (smooth prediction)
            count_featured = [] #list for each weighted mean prediction (featured prediction)
            count_artifact = [] #list for each weighted mean prediction (artifact prediction)
            
            for i in range(len(cut)):
                if cut['debiased_smooth_pred'][i]>=cut_threshold:
                    count_smooth.append("smooth")
                elif cut['debiased_featured_pred'][i]>=cut_threshold:
                    count_featured.append("featured")
                elif cut['debiased_artifact_pred'][i]>=cut_threshold:
                    count_artifact.append("artifact")
                            
            confident_smooth = len(count_smooth) 
            confident_featured = len(count_featured)
            confident_artifact = len(count_artifact)
            total = confident_smooth + confident_featured + confident_artifact
            df_redshift = pd.DataFrame([[pred_redshift, confident_smooth/total, confident_featured/total, confident_artifact/total]], columns=df.columns)
            df = pd.concat([df,df_redshift])
            
        frf.error_bar_smoothness_3(df['redshift'].values, df['smooth'].values, df['featured'].values, df['artifact'].values, save_name=figure_save_names[j], title='Galaxy Morphology with Redshift for Magnitudes {0}'.format(figure_mag_lims[j]), xlabel='Redshift', ylabel='Proportion of expected predictions', ylimits=[0, 1], xlimits=[0, 0.12])
        j += 1
        logging.info("new cut")
        
    logging.info('\n MORPHOLOGY WITH REDSHIFT CODE COMPLETE')