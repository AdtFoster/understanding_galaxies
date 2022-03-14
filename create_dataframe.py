# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:32:45 2022

@author: r41331jc
"""
import os
import logging

import numpy as np
import pandas as pd
import argparse

import functions_for_redshifting_figures as frf

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    scale_factor_data={}
    full_data_array_first_cut=np.zeros((0, 10))
    full_data_array_first_cut_var=np.zeros((0, 10))
    
    # The data
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name', dest='file_name', type=str, help='ML predictions e.g. scaled_image_predictions_1.csv')
    parser.add_argument('--min-allow-z', dest='min_allow_z', type=float)
    parser.add_argument('--max-allow-z', dest='max_allow_z', type=float)
    args = parser.parse_args()

    file_name = args.file_name
    
    max_allow_z = args.max_allow_z
    min_allow_z = args.min_allow_z
    
    # max_allow_z = 0.25
    # min_allow_z = 0.02
    
    # file_name_list = ['scaled_image_predictions_1.csv', 'scaled_image_predictions_1.2.csv', 'scaled_image_predictions_1.4.csv', 'scaled_image_predictions_1.6.csv', 'scaled_image_predictions_1.8.csv', 'scaled_image_predictions_2.csv', 'scaled_image_predictions_2.2.csv', 'scaled_image_predictions_2.4.csv', 'scaled_image_predictions_2.6.csv', 'scaled_image_predictions_2.8.csv', 'scaled_image_predictions_3.csv', 'scaled_image_predictions_4.csv', 'scaled_image_predictions_5.csv', 'scaled_image_predictions_6.csv']
    # scale_factor_multiplier=[1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 4, 5, 6] #index used for scale facotr multiplication
    # i=0 

    if os.path.isdir('/share/nas2'):
        catalog_loc = '/share/nas2/walml/repos/gz-decals-classifiers/data/catalogs/nsa_v1_0_1_mag_cols.parquet'
    else:
        catalog_loc = 'nsa_v1_0_1_mag_cols.parquet'

    nsa_catalog = pd.read_parquet(catalog_loc, columns= ['iauname', 'redshift', 'elpetro_absmag_r','elpetro_mass','petro_th50','petro_th90'])
    nsa_catalog['concentration'] = nsa_catalog['petro_th50'] / nsa_catalog['petro_th90']

    # logging.info(pd.read_csv(file_name).columns.values)
    scale_factor_data[file_name] = frf.file_reader(file_name)

    scale_factor_df = pd.DataFrame(scale_factor_data[file_name])
    scale_factor_df.rename(columns=
        {
            'id_str': 'iauname', 
        },
    inplace=True) #rename the headers of the dataframe
    

    logging.info(scale_factor_df['iauname'])
    # scale_factor_df['iauname'] = scale_factor_df.iauname.str.replace('/share/nas/walml/repos/understanding_galaxies/scaled_{0}/'.format(scale_factor_multiplier[i]), '', regex=False)
    # scale_factor_df['iauname'] = scale_factor_df.iauname.str.replace('.png', '', regex=False)
    scale_factor_df['iauname'] = scale_factor_df.iauname.str.replace(scale_factor_df.iauname.str.split('_')[1], '', regex=False)
    scale_factor_df['iauname'] = scale_factor_df.iauname.str.replace('_', '', regex=False)
    
    logging.info(scale_factor_df['iauname'])
    # merged_dataframe = scale_factor_df.merge(nsa_catalog, left_on='iauname', right_on='iauname', how='left') #in final version make the larger dataframe update itsefl to be smaller?
    # merged_dataframe['redshift']=merged_dataframe['redshift'].clip(lower=1e-10) #removes any negative errors
    # merged_dataframe['redshift']=merged_dataframe['redshift'].mul(scale_factor_multiplier[i]) #Multiplies the redshift by the scalefactor
    merged_dataframe = scale_factor_df.merge(nsa_catalog, left_on='iauname', right_on='iauname', how='inner')
    logging.info(len(merged_dataframe))
    assert len(merged_dataframe) > 0

    merged_dataframe['redshift'] = merged_dataframe['redshift'].clip(lower=1e-10) #removes any negative errors should be unnecessary

    first_mag_cut = merged_dataframe[(merged_dataframe["elpetro_absmag_r"] < -18 ) & (merged_dataframe["elpetro_absmag_r"] >= -24) & (merged_dataframe["redshift"] <= max_allow_z) & (merged_dataframe["redshift"] >= min_allow_z)]

    merged_dataframe['redshift']=merged_dataframe['redshift'].mul(float(scale_factor_df.iauname.str.split('_')[1])) #Multiplies the redshift by the scalefactor
    
    merged_numpy_first_cut = first_mag_cut.to_numpy(dtype=str) #converts dataframe to numpy array for manipulation
    
    numpy_merged_probs_first_cut = frf.prob_maker(merged_numpy_first_cut)
    numpy_merged_var_first_cut = frf.variance_from_beta(merged_numpy_first_cut)

    numpy_merged_probs_first_cut = np.hstack((numpy_merged_probs_first_cut, merged_numpy_first_cut[:, -1:]))
    numpy_merged_var_first_cut = np.hstack((numpy_merged_var_first_cut, merged_numpy_first_cut[:, -1:]))

    full_data_array_first_cut=np.vstack((full_data_array_first_cut, numpy_merged_probs_first_cut)) #stacks all data from current redshift to cumulative array
    full_data_array_first_cut_var=np.vstack((full_data_array_first_cut_var, numpy_merged_var_first_cut))
        # i+=1
    
    pd.DataFrame(full_data_array_first_cut).to_csv('full_data.csv', index=False)
    pd.DataFrame(full_data_array_first_cut_var).to_csv('full_data_var.csv', index=False)
    
    print('Files appended, removing test sample')
